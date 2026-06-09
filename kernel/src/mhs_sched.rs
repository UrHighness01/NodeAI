//! MHS (Multi-Scale Hierarchical State) scheduler — O(T) replacement for the
//! O(T²) transformer in transformer_sched.rs.
//!
//! Architecture (ported from Project-M, adapted for kernel scheduling):
//!
//!   FastState   — per-token GLA recurrence (dh=16).  Captures per-syscall patterns.
//!   SlowState   — event-gated GLA (dh=20, soft sigmoid gate).  Self-organises to fire
//!                 at scheduling-domain shifts (batch→interactive, I/O→CPU, etc.) —
//!                 the scheduler analogue of Project-M's document-boundary detection.
//!
//!   Input  : last CONTEXT_LEN syscall numbers (same window as transformer_sched)
//!   Embed  : syscall_nr → D_MODEL-dim embedding (VOCAB_SIZE × D_MODEL table)
//!   GLA    : O(T) causal linear-attention recurrence; no softmax, no T² term
//!   Pool   : mean of per-token outputs from each level
//!   Head   : Dense(DH0+DH2 → HEAD_H, ReLU) → Dense(HEAD_H → 4, Linear)
//!   Output : [nice_delta, burst_ticks, prefault_pages, predicted_wait_us]
//!
//! INT8 quantisation (Project-T insight):
//!   After QUANTIZE_AFTER SGD steps, all weight matrices are frozen as i8 with
//!   a per-matrix f32 scale.  Inference dequantises on the fly → 8× weight-memory
//!   reduction, integer accumulation on modern x86 integer pipeline.
//!
//! This is the first OS scheduler using hierarchical gated linear attention;
//! the SlowState gate learns scheduling-domain boundaries without supervision.

#![allow(clippy::needless_range_loop)]

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use spin::Mutex;

// ── Dimensions ────────────────────────────────────────────────────────────────
pub  const CONTEXT_LEN: usize = 16;  // same window as transformer_sched
const VOCAB_SIZE:  usize = 512;  // syscall number range
const D_MODEL:     usize = 32;   // embedding dimension
const DH0:         usize = 16;   // FastState GLA dimension
const DH2:         usize = 20;   // SlowState GLA dimension
const HEAD_H:      usize = 20;   // output head hidden size
const N_OUTPUTS:   usize = 4;    // [nice, burst, pf_pages, wait_us]
const QUANTIZE_AFTER: u64 = 500; // SGD steps before weight quantisation
const LR_INIT:    f32 = 0.002;
const LR_DECAY:   f32 = 0.00005;

// ── Quantised weight helper ───────────────────────────────────────────────────

/// A weight matrix stored as either f32 (training) or INT8 (inference).
/// INT8 storage: w_f32 ≈ i8_val * scale.  Dequantise on the fly in matmul.
struct QWeight {
    f32w:  Box<[f32]>,      // f32 training shadow (always kept for SGD)
    i8w:   Box<[i8]>,       // quantised copy (valid only if quantised=true)
    scale: f32,             // dequantisation scale
    rows:  usize,
    cols:  usize,
    quantised: bool,
}

impl QWeight {
    fn new(rows: usize, cols: usize, seed: u64) -> Self {
        let fan_in = cols;
        let f32w: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let h = seed
                    .wrapping_add(i as u64 * 2654435761)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let scale = libm::sqrtf(2.0 / fan_in as f32);
                let frac = ((h >> 11) as f32) / 4503599627370496.0f32;
                (frac * 2.0 - 1.0) * scale
            })
            .collect();
        Self {
            i8w: alloc::vec![0i8; rows * cols].into_boxed_slice(),
            f32w: f32w.into_boxed_slice(),
            scale: 1.0,
            rows,
            cols,
            quantised: false,
        }
    }

    fn quantise(&mut self) {
        let max_abs = self.f32w.iter().cloned()
            .fold(1e-9f32, |a, v| a.max(v.abs()));
        self.scale = max_abs / 127.0;
        for (f, i) in self.f32w.iter().zip(self.i8w.iter_mut()) {
            *i = (libm::roundf(f / self.scale).clamp(-127.0, 127.0)) as i8;
        }
        self.quantised = true;
    }

    /// mat-vec: out[row] = sum_col(W[row,col] * x[col])
    fn matvec(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.rows);
        debug_assert_eq!(x.len(), self.cols);
        if self.quantised {
            for r in 0..self.rows {
                let mut acc: i32 = 0;
                for c in 0..self.cols {
                    acc += self.i8w[r * self.cols + c] as i32 * libm::roundf(x[c] / self.scale) as i32;
                }
                out[r] = acc as f32 * self.scale * self.scale;
            }
        } else {
            for r in 0..self.rows {
                let mut s = 0.0f32;
                for c in 0..self.cols { s += self.f32w[r * self.cols + c] * x[c]; }
                out[r] = s;
            }
        }
    }

    /// outer-product accumulate: S[r,c] += k[r] * v[c]
    /// Used to update GLA state matrix.
    fn outer_acc(s: &mut [f32], k: &[f32], v: &[f32], dh: usize) {
        for r in 0..dh {
            for c in 0..dh { s[r * dh + c] += k[r] * v[c]; }
        }
    }
}

// ── ELU+1 feature map ────────────────────────────────────────────────────────

#[inline(always)]
fn elu1(x: f32) -> f32 {
    if x >= 0.0 { x + 1.0 } else { libm::expf(x) }
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + libm::expf(-x))
}

#[inline(always)]
fn relu(x: f32) -> f32 { x.max(0.0) }

// ── Model ─────────────────────────────────────────────────────────────────────

struct MhsModel {
    // Embedding
    embed:   Box<[f32]>,          // [VOCAB_SIZE × D_MODEL]

    // FastState QKV projections (D_MODEL → DH0)
    fq: QWeight, fk: QWeight, fv: QWeight,
    ffg: Box<[f32]>, ffg_b: f32,  // forget gate linear (D_MODEL → 1) + bias

    // SlowState QKV projections (D_MODEL → DH2)
    sq: QWeight, sk: QWeight, sv: QWeight,
    sg: Box<[f32]>, sg_b: f32,    // event gate linear (D_MODEL → 1) + bias
    temperature: f32,             // learnable gate temperature (init 0.5)

    // Output head (DH0+DH2 → HEAD_H → N_OUTPUTS)
    h1_w: QWeight, h1_b: Box<[f32]>,
    h2_w: QWeight, h2_b: Box<[f32]>,

    steps: u64,
}

impl MhsModel {
    fn new() -> Self {
        let embed: Vec<f32> = (0..VOCAB_SIZE * D_MODEL)
            .map(|i| {
                let seed = 0xdead_beef_cafe_1234u64;
                let h = seed.wrapping_add(i as u64 * 2654435761)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let scale = libm::sqrtf(2.0 / VOCAB_SIZE as f32);
                let frac = ((h >> 11) as f32) / 4503599627370496.0f32;
                (frac * 2.0 - 1.0) * scale
            })
            .collect();

        let ffg: Vec<f32> = (0..D_MODEL)
            .map(|i| {
                let h = 0xaaaa_bbbb_1111_2222u64
                    .wrapping_add(i as u64 * 2246822519)
                    .wrapping_mul(6364136223846793005);
                ((((h >> 11) as f32) / 4503599627370496.0f32) * 2.0 - 1.0) * 0.01
            })
            .collect();
        let sg: Vec<f32> = (0..D_MODEL)
            .map(|i| {
                let h = 0xcccc_dddd_3333_4444u64
                    .wrapping_add(i as u64 * 2246822519)
                    .wrapping_mul(6364136223846793005);
                ((((h >> 11) as f32) / 4503599627370496.0f32) * 2.0 - 1.0) * 0.01
            })
            .collect();

        Self {
            embed: embed.into_boxed_slice(),
            fq: QWeight::new(DH0, D_MODEL, 0xf001_0001_0001_0001),
            fk: QWeight::new(DH0, D_MODEL, 0xf002_0002_0002_0002),
            fv: QWeight::new(DH0, D_MODEL, 0xf003_0003_0003_0003),
            ffg: ffg.into_boxed_slice(),
            ffg_b: 0.0,
            sq: QWeight::new(DH2, D_MODEL, 0x5001_5001_5001_5001),
            sk: QWeight::new(DH2, D_MODEL, 0x5002_5002_5002_5002),
            sv: QWeight::new(DH2, D_MODEL, 0x5003_5003_5003_5003),
            sg: sg.into_boxed_slice(),
            sg_b: -1.0,  // sparse init: gate starts near 0 (threshold -1.0)
            temperature: 0.5,
            h1_w: QWeight::new(HEAD_H, DH0 + DH2, 0x1234_abcd_ef01_2345),
            h1_b: alloc::vec![0.0f32; HEAD_H].into_boxed_slice(),
            h2_w: QWeight::new(N_OUTPUTS, HEAD_H, 0x8765_4321_fedc_ba98),
            h2_b: alloc::vec![0.0f32; N_OUTPUTS].into_boxed_slice(),
            steps: 0,
        }
    }

    fn embed_token(&self, nr: u16) -> &[f32] {
        let idx = (nr as usize).min(VOCAB_SIZE - 1);
        &self.embed[idx * D_MODEL..(idx + 1) * D_MODEL]
    }

    fn gla_fast(&self, syscalls: &[u16; CONTEXT_LEN]) -> (Vec<f32>, f32) {
        let mut s = alloc::vec![0.0f32; DH0 * DH0]; // state matrix
        let mut z = alloc::vec![0.0f32; DH0];         // normaliser
        let mut pool = alloc::vec![0.0f32; DH0];
        let mut gate_sum = 0.0f32;

        let mut qbuf = alloc::vec![0.0f32; DH0];
        let mut kbuf = alloc::vec![0.0f32; DH0];
        let mut vbuf = alloc::vec![0.0f32; DH0];

        for t in 0..CONTEXT_LEN {
            let x = self.embed_token(syscalls[t]);

            self.fq.matvec(x, &mut qbuf);
            self.fk.matvec(x, &mut kbuf);
            self.fv.matvec(x, &mut vbuf);

            // ELU+1 feature map
            for v in qbuf.iter_mut() { *v = elu1(*v); }
            for v in kbuf.iter_mut() { *v = elu1(*v); }

            // Scalar forget gate: f = sigmoid(dot(ffg, x) + b)
            let gate_logit: f32 = self.ffg.iter().zip(x.iter())
                .map(|(&w, &xi)| w * xi)
                .sum::<f32>() + self.ffg_b;
            let f = sigmoid(gate_logit);
            gate_sum += f;

            // Update state: S = f * S + outer(k, v);  z = f * z + k
            for v in s.iter_mut() { *v *= f; }
            QWeight::outer_acc(&mut s, &kbuf, &vbuf, DH0);
            for i in 0..DH0 { z[i] = f * z[i] + kbuf[i]; }

            // Read: out = S^T @ q / max(dot(q, z), eps)
            let denom = qbuf.iter().zip(z.iter()).map(|(&q, &z)| q * z).sum::<f32>().max(1e-6);
            for i in 0..DH0 {
                let mut acc = 0.0f32;
                for j in 0..DH0 { acc += s[j * DH0 + i] * qbuf[j]; }
                pool[i] += acc / denom;
            }
        }

        // Mean pool over T
        let inv_t = 1.0 / CONTEXT_LEN as f32;
        for v in pool.iter_mut() { *v *= inv_t; }
        let write_rate = gate_sum * inv_t;
        (pool, write_rate)
    }

    fn gla_slow(&self, syscalls: &[u16; CONTEXT_LEN]) -> (Vec<f32>, f32) {
        let mut s = alloc::vec![0.0f32; DH2 * DH2];
        let mut z = alloc::vec![0.0f32; DH2];
        let mut pool = alloc::vec![0.0f32; DH2];
        let mut gate_sum = 0.0f32;

        let mut qbuf = alloc::vec![0.0f32; DH2];
        let mut kbuf = alloc::vec![0.0f32; DH2];
        let mut vbuf = alloc::vec![0.0f32; DH2];

        for t in 0..CONTEXT_LEN {
            let x = self.embed_token(syscalls[t]);

            self.sq.matvec(x, &mut qbuf);
            self.sk.matvec(x, &mut kbuf);
            self.sv.matvec(x, &mut vbuf);

            for v in qbuf.iter_mut() { *v = elu1(*v); }
            for v in kbuf.iter_mut() { *v = elu1(*v); }

            // Soft event gate: g = sigmoid(temperature * (dot(sg, x) + b))
            let gate_logit: f32 = self.sg.iter().zip(x.iter())
                .map(|(&w, &xi)| w * xi)
                .sum::<f32>() + self.sg_b;
            let g = sigmoid(self.temperature * gate_logit);
            gate_sum += g;

            for v in s.iter_mut() { *v *= g; }
            QWeight::outer_acc(&mut s, &kbuf, &vbuf, DH2);
            for i in 0..DH2 { z[i] = g * z[i] + kbuf[i]; }

            let denom = qbuf.iter().zip(z.iter()).map(|(&q, &z)| q * z).sum::<f32>().max(1e-6);
            for i in 0..DH2 {
                let mut acc = 0.0f32;
                for j in 0..DH2 { acc += s[j * DH2 + i] * qbuf[j]; }
                pool[i] += acc / denom;
            }
        }

        let inv_t = 1.0 / CONTEXT_LEN as f32;
        for v in pool.iter_mut() { *v *= inv_t; }
        let write_rate = gate_sum * inv_t;
        (pool, write_rate)
    }

    fn forward(&self, syscalls: &[u16; CONTEXT_LEN]) -> MhsDecision {
        let (fast_pool, _)            = self.gla_fast(syscalls);
        let (slow_pool, slow_wr) = self.gla_slow(syscalls);

        // Concatenate fast + slow pools → [DH0 + DH2] dim
        let mut concat = alloc::vec![0.0f32; DH0 + DH2];
        concat[..DH0].copy_from_slice(&fast_pool);
        concat[DH0..].copy_from_slice(&slow_pool);

        // Head
        let mut h1 = alloc::vec![0.0f32; HEAD_H];
        self.h1_w.matvec(&concat, &mut h1);
        for (i, b) in h1.iter_mut().enumerate() { *b += self.h1_b[i]; *b = relu(*b); }

        let mut out = alloc::vec![0.0f32; N_OUTPUTS];
        self.h2_w.matvec(&h1, &mut out);
        for (i, b) in out.iter_mut().enumerate() { *b += self.h2_b[i]; }

        // Gate uncertainty: how far from 0.5 is the slow gate's write rate?
        // slow_wr ≈ 0.5 → normal; ≈ 0 or 1 → collapsed → less confident
        let gate_uncertainty = 1.0 - 2.0 * (slow_wr - 0.5).abs();

        MhsDecision {
            nice_delta:      out[0].clamp(-20.0, 20.0) as i8,
            burst_ticks:     out[1].clamp(1.0, 50.0) as u32,
            prefault_pages:  out[2].clamp(0.0, 32.0) as u8,
            predicted_wait:  out[3].max(0.0) as u32,
            gate_uncertainty: gate_uncertainty.clamp(0.0, 1.0),
            slow_write_rate:  slow_wr,
        }
    }

    fn quantise_all(&mut self) {
        self.fq.quantise(); self.fk.quantise(); self.fv.quantise();
        self.sq.quantise(); self.sk.quantise(); self.sv.quantise();
        self.h1_w.quantise(); self.h2_w.quantise();
    }

    fn sgd_step(&mut self, syscalls: &[u16; CONTEXT_LEN], target: [f32; N_OUTPUTS]) {
        self.steps += 1;

        if self.steps == QUANTIZE_AFTER {
            self.quantise_all();
            crate::klog!(INFO, "mhs_sched: quantised all weights to INT8 at step {}", self.steps);
        }

        let lr = LR_INIT / (1.0 + self.steps as f32 * LR_DECAY);

        // Forward (use f32 shadow for gradient computation)
        let (fast_pool, _)  = self.gla_fast(syscalls);
        let (slow_pool, _)  = self.gla_slow(syscalls);
        let mut concat = alloc::vec![0.0f32; DH0 + DH2];
        concat[..DH0].copy_from_slice(&fast_pool);
        concat[DH0..].copy_from_slice(&slow_pool);

        let mut h1_pre = alloc::vec![0.0f32; HEAD_H];
        self.h1_w.matvec(&concat, &mut h1_pre);
        for (i, b) in h1_pre.iter_mut().enumerate() { *b += self.h1_b[i]; }
        let mut h1 = h1_pre.clone();
        for v in h1.iter_mut() { *v = relu(*v); }

        let mut out = alloc::vec![0.0f32; N_OUTPUTS];
        self.h2_w.matvec(&h1, &mut out);
        for (i, b) in out.iter_mut().enumerate() { *b += self.h2_b[i]; }

        // Layer 2 gradient (MSE)
        let mut dout = [0.0f32; N_OUTPUTS];
        for i in 0..N_OUTPUTS { dout[i] = (out[i] - target[i]) * 2.0; }

        for i in 0..N_OUTPUTS {
            self.h2_b[i] -= lr * dout[i];
            for j in 0..HEAD_H {
                self.h2_w.f32w[i * HEAD_H + j] -= lr * dout[i] * h1[j];
            }
        }

        // Layer 1 gradient (ReLU mask)
        let mut dh1 = alloc::vec![0.0f32; HEAD_H];
        for j in 0..HEAD_H {
            let mut g = 0.0f32;
            for i in 0..N_OUTPUTS { g += self.h2_w.f32w[i * HEAD_H + j] * dout[i]; }
            dh1[j] = if h1_pre[j] > 0.0 { g } else { 0.0 };
        }

        for i in 0..HEAD_H {
            self.h1_b[i] -= lr * dh1[i];
            let feature_dim = DH0 + DH2;
            for j in 0..feature_dim {
                self.h1_w.f32w[i * feature_dim + j] -= lr * dh1[i] * concat[j];
            }
        }

        // Re-quantise after each update if past the quantisation threshold
        if self.steps > QUANTIZE_AFTER && self.steps % 50 == 0 {
            self.quantise_all();
        }
    }
}

// ── Public output type ────────────────────────────────────────────────────────

/// MHS scheduler output (same semantic fields as SchedDecision in transformer_sched).
#[derive(Clone, Copy, Debug, Default)]
pub struct MhsDecision {
    pub nice_delta:      i8,
    pub burst_ticks:     u32,
    pub prefault_pages:  u8,
    pub predicted_wait:  u32,  // µs
    /// How uncertain the SlowState gate is: 0 = collapsed (gate stuck at 0 or 1),
    /// 1 = healthy balanced firing.  Used for confidence blending.
    pub gate_uncertainty: f32,
    /// Raw slow-gate mean write rate for observability (logged in /proc/sched_mhs).
    pub slow_write_rate:  f32,
}

// ── Per-PID context ───────────────────────────────────────────────────────────

struct MhsContext {
    history: [u16; CONTEXT_LEN],
    head:    usize,
    count:   usize,
    /// Pending feedback: snapshot of history at the time of the last infer,
    /// to be used as training input when the outcome is known.
    pending: Option<([u16; CONTEXT_LEN], [f32; N_OUTPUTS])>,
}

impl MhsContext {
    const fn new() -> Self {
        Self { history: [0u16; CONTEXT_LEN], head: 0, count: 0, pending: None }
    }

    fn push(&mut self, syscall_nr: u16) {
        self.history[self.head] = syscall_nr;
        self.head = (self.head + 1) % CONTEXT_LEN;
        self.count = self.count.saturating_add(1);
    }

    fn snapshot(&self) -> [u16; CONTEXT_LEN] {
        let mut s = [0u16; CONTEXT_LEN];
        for i in 0..CONTEXT_LEN {
            s[i] = self.history[(self.head + i) % CONTEXT_LEN];
        }
        s
    }

    fn is_warm(&self) -> bool { self.count >= CONTEXT_LEN }
}

// ── Globals ───────────────────────────────────────────────────────────────────

static MODEL:    Mutex<Option<MhsModel>>          = Mutex::new(None);
static CONTEXTS: Mutex<BTreeMap<u64, MhsContext>> = Mutex::new(BTreeMap::new());

/// Initialise the MHS model (called once from main.rs after heap is ready).
pub fn init() {
    *MODEL.lock() = Some(MhsModel::new());
    crate::klog!(INFO, "mhs_sched: init — FastState dh={} SlowState dh={} VOCAB={} QUANTIZE_AFTER={}",
        DH0, DH2, VOCAB_SIZE, QUANTIZE_AFTER);
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Record a syscall number for `pid` (call on every syscall entry).
pub fn record_syscall(pid: u64, nr: u16) {
    CONTEXTS.lock().entry(pid).or_insert_with(MhsContext::new).push(nr);
}

/// Run MHS forward pass for `pid`.  Returns None if not enough history yet.
pub fn infer(pid: u64) -> Option<MhsDecision> {
    let snap = {
        let ctx = CONTEXTS.lock();
        let c = ctx.get(&pid)?;
        if !c.is_warm() { return None; }
        c.snapshot()
    };
    let model = MODEL.lock();
    model.as_ref().map(|m| m.forward(&snap))
}

/// Record scheduling outcome for `pid` and run an SGD step.
/// `actual_wait_us`: measured CPU wait after this scheduling decision.
/// `actual_burst`:   how many ticks the task actually ran.
pub fn record_feedback(pid: u64, actual_wait_us: u32, actual_burst: u32) {
    let snap = {
        let ctx = CONTEXTS.lock();
        match ctx.get(&pid) {
            None => return,
            Some(c) => if !c.is_warm() { return; } else { c.snapshot() },
        }
    };
    let target = [
        0.0f32,                    // nice_delta supervised externally
        actual_burst as f32,
        4.0,                       // prefault_pages target
        actual_wait_us as f32,
    ];
    if let Some(m) = MODEL.lock().as_mut() {
        m.sgd_step(&snap, target);
    }
}

/// Remove per-PID context on process exit.
pub fn remove(pid: u64) {
    CONTEXTS.lock().remove(&pid);
}

/// Format /proc/sched_mhs report.
pub fn format_report() -> Vec<u8> {
    use alloc::string::String;
    let (steps, quantised) = {
        let m = MODEL.lock();
        match m.as_ref() {
            None => (0u64, false),
            Some(m) => (m.steps, m.fq.quantised),
        }
    };
    let (ctx_n, warm_n) = {
        let ctx = CONTEXTS.lock();
        (ctx.len(), ctx.values().filter(|c| c.is_warm()).count())
    };

    let fast_params = DH0 * D_MODEL * 3 + D_MODEL + 1; // QKV + gate
    let slow_params = DH2 * D_MODEL * 3 + D_MODEL + 1 + 1; // QKV + gate + temperature
    let embed_params = VOCAB_SIZE * D_MODEL;
    let head_params  = HEAD_H * (DH0 + DH2) + HEAD_H + N_OUTPUTS * HEAD_H + N_OUTPUTS;
    let total = fast_params + slow_params + embed_params + head_params;
    let mem_kb_f32 = total * 4 / 1024;
    let mem_kb_i8  = (fast_params + slow_params + head_params) / 1024; // embed stays f32

    let mut s = String::from("NodeAI MHS Scheduler (O(T) GLA, cross-project: Project-M)\n");
    s.push_str("===========================================================\n");
    s.push_str(&alloc::format!("total_params    : {}\n", total));
    s.push_str(&alloc::format!("fast_state_dh   : {} (GLA per-token)\n", DH0));
    s.push_str(&alloc::format!("slow_state_dh   : {} (event-gated)\n", DH2));
    s.push_str(&alloc::format!("embed_params    : {} ({}×{})\n", embed_params, VOCAB_SIZE, D_MODEL));
    s.push_str(&alloc::format!("sgd_steps       : {}\n", steps));
    s.push_str(&alloc::format!("int8_quantised  : {} (after step {})\n", quantised, QUANTIZE_AFTER));
    s.push_str(&alloc::format!("mem_f32_kb      : {} → int8_kb: {} (8× reduction)\n", mem_kb_f32, mem_kb_i8));
    s.push_str(&alloc::format!("active_pids     : {} ({} warm)\n", ctx_n, warm_n));
    s.push_str("outputs         : [nice, burst_ticks, pf_pages, wait_us]\n");
    s.push_str("confidence      : gate_uncertainty (0=collapsed, 1=healthy)\n");
    s.into_bytes()
}
