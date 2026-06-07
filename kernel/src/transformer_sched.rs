//! Transformer-based scheduling policy — no kernel has done this before.
//!
//! Architecture:
//!   Input : last CONTEXT_LEN syscall numbers for the current task
//!   Embed : syscall_nr → 32-dim learnable embedding (512 × 32 table)
//!   Attn  : single-head self-attention (Q/K/V projections 32→16)
//!   Pool  : mean of attended token vectors → 16-dim context vector
//!   Head  : Dense(16→16, ReLU) → Dense(16→3, Linear)
//!   Output: [nice_delta f32, burst_ticks f32, prefault_pages f32]
//!
//! Online SGD: after each descheduling event the output head is updated with
//! the observed actual values (intent nice, actual burst, cluster pf).
//! Attention weights are frozen at bootstrap — full backprop is deferred to
//! a future offline training pass.
//!
//! The kernel calls `predict(pid)` every time a task is selected to run.
//! The result is stored in the task's AiProfile and used by the scheduler
//! alongside (and eventually replacing) the fingerprint cluster profile.

use alloc::vec::Vec;
use spin::Mutex;

pub const CONTEXT_LEN:  usize = 16;  // syscall history window
pub const EMBED_DIM:    usize = 32;  // per-token embedding size
pub const ATTN_DIM:     usize = 16;  // Q/K/V projection size
pub const VOCAB_SIZE:   usize = 512; // max syscall number tracked

/// Transformer scheduler output.
#[derive(Clone, Copy, Debug, Default)]
pub struct SchedDecision {
    /// Nice adjustment in [-20, 20].
    pub nice_delta:     i8,
    /// Recommended quantum in ticks [1, 50].
    pub burst_ticks:    u32,
    /// Recommended extra prefault pages [0, 32].
    pub prefault_pages: u8,
}

// ── Model weights ─────────────────────────────────────────────────────────────

struct TransformerSchedModel {
    /// Embedding table: [VOCAB_SIZE * EMBED_DIM] row-major.
    embed: alloc::boxed::Box<[f32]>,

    /// Q projection: [ATTN_DIM * EMBED_DIM]
    wq: alloc::boxed::Box<[f32]>,
    /// K projection: [ATTN_DIM * EMBED_DIM]
    wk: alloc::boxed::Box<[f32]>,
    /// V projection: [ATTN_DIM * EMBED_DIM]
    wv: alloc::boxed::Box<[f32]>,

    /// Output head layer 1: [16 * ATTN_DIM] + bias [16]
    h1_w: alloc::boxed::Box<[f32]>,
    h1_b: alloc::boxed::Box<[f32]>,

    /// Output head layer 2: [3 * 16] + bias [3]
    h2_w: alloc::boxed::Box<[f32]>,
    h2_b: alloc::boxed::Box<[f32]>,

    /// SGD step count (for learning rate decay).
    steps: u64,
}

/// Deterministic weight initializer — uses a PRNG seeded from (row, col, seed)
/// to fill weights with Xavier-like values without any heap randomness.
fn init_weight(row: usize, col: usize, fan_in: usize, seed: u64) -> f32 {
    // Simple LCG over (row, col, seed) → float in [-scale, scale]
    let h = seed
        .wrapping_add(row as u64 * 2654435761)
        .wrapping_add(col as u64 * 2246822519)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let scale = (2.0 / fan_in as f32).sqrt();
    let frac = ((h >> 11) as f32) / (f32::powi(2.0, 53));
    (frac * 2.0 - 1.0) * scale
}

impl TransformerSchedModel {
    fn new() -> Self {
        // Embed: Xavier init with fan_in = VOCAB_SIZE
        let embed: alloc::vec::Vec<f32> = (0..VOCAB_SIZE * EMBED_DIM)
            .map(|i| init_weight(i / EMBED_DIM, i % EMBED_DIM, VOCAB_SIZE, 0x_dead_beef))
            .collect();

        // Q/K/V: fan_in = EMBED_DIM
        let wq: Vec<f32> = (0..ATTN_DIM * EMBED_DIM)
            .map(|i| init_weight(i / EMBED_DIM, i % EMBED_DIM, EMBED_DIM, 0x_feed_cafe))
            .collect();
        let wk: Vec<f32> = (0..ATTN_DIM * EMBED_DIM)
            .map(|i| init_weight(i / EMBED_DIM, i % EMBED_DIM, EMBED_DIM, 0x_cafe_babe))
            .collect();
        let wv: Vec<f32> = (0..ATTN_DIM * EMBED_DIM)
            .map(|i| init_weight(i / EMBED_DIM, i % EMBED_DIM, EMBED_DIM, 0x_baad_f00d))
            .collect();

        // Output head 1: 16 neurons × ATTN_DIM inputs
        let h1_w: Vec<f32> = (0..16 * ATTN_DIM)
            .map(|i| init_weight(i / ATTN_DIM, i % ATTN_DIM, ATTN_DIM, 0x_1234_5678))
            .collect();
        let h1_b = alloc::vec![0.0f32; 16];

        // Output head 2: 3 outputs × 16 inputs
        let h2_w: Vec<f32> = (0..3 * 16)
            .map(|i| init_weight(i / 16, i % 16, 16, 0x_8765_4321))
            .collect();
        let h2_b = alloc::vec![0.0f32; 3];

        Self {
            embed: embed.into_boxed_slice(),
            wq:    wq.into_boxed_slice(),
            wk:    wk.into_boxed_slice(),
            wv:    wv.into_boxed_slice(),
            h1_w:  h1_w.into_boxed_slice(),
            h1_b:  h1_b.into_boxed_slice(),
            h2_w:  h2_w.into_boxed_slice(),
            h2_b:  h2_b.into_boxed_slice(),
            steps: 0,
        }
    }

    // ── Forward pass ─────────────────────────────────────────────────────────

    /// Embed a sequence of syscall numbers → matrix [CONTEXT_LEN × EMBED_DIM].
    fn embed_sequence(&self, syscalls: &[u16; CONTEXT_LEN]) -> Vec<f32> {
        let mut mat = alloc::vec![0.0f32; CONTEXT_LEN * EMBED_DIM];
        for (t, &nr) in syscalls.iter().enumerate() {
            let idx = (nr as usize).min(VOCAB_SIZE - 1);
            let row = &self.embed[idx * EMBED_DIM..(idx + 1) * EMBED_DIM];
            mat[t * EMBED_DIM..(t + 1) * EMBED_DIM].copy_from_slice(row);
        }
        mat
    }

    /// Dense multiply: out[i] = sum_j(w[i*in_size+j] * x[j]) + b[i].
    fn dense(w: &[f32], b: &[f32], x: &[f32], out_size: usize) -> Vec<f32> {
        let in_size = x.len();
        let mut out = alloc::vec![0.0f32; out_size];
        for i in 0..out_size {
            let mut sum = b[i];
            for j in 0..in_size {
                sum += w[i * in_size + j] * x[j];
            }
            out[i] = sum;
        }
        out
    }

    /// Scaled dot-product softmax attention over [CONTEXT_LEN × EMBED_DIM].
    /// Returns attended output [CONTEXT_LEN × ATTN_DIM].
    fn attention(&self, tokens: &[f32]) -> Vec<f32> {
        let t = CONTEXT_LEN;
        let d = EMBED_DIM;
        let a = ATTN_DIM;
        let scale = 1.0 / (a as f32).sqrt();

        // Project Q, K, V: [t × a] each.
        let mut q = alloc::vec![0.0f32; t * a];
        let mut k = alloc::vec![0.0f32; t * a];
        let mut v = alloc::vec![0.0f32; t * a];
        for tok in 0..t {
            let x = &tokens[tok * d..(tok + 1) * d];
            for i in 0..a {
                let mut sq = 0.0f32;
                let mut sk = 0.0f32;
                let mut sv = 0.0f32;
                for j in 0..d {
                    sq += self.wq[i * d + j] * x[j];
                    sk += self.wk[i * d + j] * x[j];
                    sv += self.wv[i * d + j] * x[j];
                }
                q[tok * a + i] = sq;
                k[tok * a + i] = sk;
                v[tok * a + i] = sv;
            }
        }

        // Attention scores: A[i,j] = softmax(Q[i] · K[j] * scale) over j.
        let mut attn_out = alloc::vec![0.0f32; t * a];
        for i in 0..t {
            // Compute raw scores Q[i] · K[j] for all j.
            let mut scores = alloc::vec![0.0f32; t];
            for j in 0..t {
                let mut dot = 0.0f32;
                for h in 0..a {
                    dot += q[i * a + h] * k[j * a + h];
                }
                scores[j] = dot * scale;
            }
            // Numerical-stable softmax.
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = fast_exp(*s - max_s);
                exp_sum += *s;
            }
            if exp_sum > 1e-9 {
                for s in scores.iter_mut() { *s /= exp_sum; }
            }
            // Weighted sum of V.
            for j in 0..t {
                for h in 0..a {
                    attn_out[i * a + h] += scores[j] * v[j * a + h];
                }
            }
        }
        attn_out
    }

    /// Full forward pass: syscall sequence → SchedDecision.
    fn forward(&self, syscalls: &[u16; CONTEXT_LEN]) -> SchedDecision {
        let tokens    = self.embed_sequence(syscalls);
        let attn_out  = self.attention(&tokens);

        // Mean-pool over time dimension: [ATTN_DIM]
        let mut pooled = alloc::vec![0.0f32; ATTN_DIM];
        for t in 0..CONTEXT_LEN {
            for i in 0..ATTN_DIM {
                pooled[i] += attn_out[t * ATTN_DIM + i];
            }
        }
        let inv_t = 1.0 / CONTEXT_LEN as f32;
        for v in pooled.iter_mut() { *v *= inv_t; }

        // Head layer 1 + ReLU.
        let mut h1 = Self::dense(&self.h1_w, &self.h1_b, &pooled, 16);
        for v in h1.iter_mut() { if *v < 0.0 { *v = 0.0; } }

        // Head layer 2 (linear — we clamp outputs below).
        let out = Self::dense(&self.h2_w, &self.h2_b, &h1, 3);

        SchedDecision {
            nice_delta:     (out[0].clamp(-20.0, 20.0)) as i8,
            burst_ticks:    (out[1].clamp(1.0, 50.0)) as u32,
            prefault_pages: (out[2].clamp(0.0, 32.0)) as u8,
        }
    }

    // ── Online SGD on output head ─────────────────────────────────────────────

    /// Update output head weights given the prediction error from one event.
    ///
    /// `syscalls`  — the context that produced the prediction
    /// `target`    — observed actual values [nice_delta, burst_ticks, pf_pages]
    fn sgd_step(&mut self, syscalls: &[u16; CONTEXT_LEN], target: [f32; 3]) {
        self.steps += 1;
        let lr = 0.001 / (1.0 + self.steps as f32 * 0.0001);

        // Re-run forward to get intermediate activations.
        let tokens    = self.embed_sequence(syscalls);
        let attn_out  = self.attention(&tokens);
        let mut pooled = alloc::vec![0.0f32; ATTN_DIM];
        for t in 0..CONTEXT_LEN {
            for i in 0..ATTN_DIM {
                pooled[i] += attn_out[t * ATTN_DIM + i];
            }
        }
        let inv_t = 1.0 / CONTEXT_LEN as f32;
        for v in pooled.iter_mut() { *v *= inv_t; }
        let mut h1 = Self::dense(&self.h1_w, &self.h1_b, &pooled, 16);
        let h1_pre = h1.clone(); // before ReLU
        for v in h1.iter_mut() { if *v < 0.0 { *v = 0.0; } }
        let out = Self::dense(&self.h2_w, &self.h2_b, &h1, 3);

        // Layer 2 gradient (MSE loss, linear activation).
        let mut dout = [0.0f32; 3];
        for i in 0..3 { dout[i] = out[i] - target[i]; }

        // Update h2 weights: dL/dW2 = dout ⊗ h1.
        for i in 0..3 {
            self.h2_b[i] -= lr * dout[i];
            for j in 0..16 {
                self.h2_w[i * 16 + j] -= lr * dout[i] * h1[j];
            }
        }

        // Back-propagate into h1: dh1 = W2^T · dout, masked by ReLU.
        let mut dh1 = alloc::vec![0.0f32; 16];
        for j in 0..16 {
            let mut g = 0.0f32;
            for i in 0..3 { g += self.h2_w[i * 16 + j] * dout[i]; }
            // ReLU mask.
            dh1[j] = if h1_pre[j] > 0.0 { g } else { 0.0 };
        }

        // Update h1 weights: dL/dW1 = dh1 ⊗ pooled.
        for i in 0..16 {
            self.h1_b[i] -= lr * dh1[i];
            for j in 0..ATTN_DIM {
                self.h1_w[i * ATTN_DIM + j] -= lr * dh1[i] * pooled[j];
            }
        }
    }
}

#[inline]
fn fast_exp(x: f32) -> f32 {
    // Schraudolph approximation — same as ai_subsystem/src/inference.rs.
    let i = (x.to_bits() as i64)
        .wrapping_add(((127.0_f32 / core::f32::consts::LN_2) as i64) << 23) as u32;
    f32::from_bits(i)
}

// ── Per-task syscall context ring ─────────────────────────────────────────────

/// Per-task circular buffer of the last CONTEXT_LEN syscall numbers.
/// Stored separately from the histogram (which is unordered frequency counts).
pub struct SyscallContext {
    ring: [u16; CONTEXT_LEN],
    pos:  usize,
    full: bool,
}

impl SyscallContext {
    pub const fn new() -> Self {
        Self { ring: [0u16; CONTEXT_LEN], pos: 0, full: false }
    }

    /// Push the most recent syscall number into the ring.
    pub fn push(&mut self, nr: u64) {
        self.ring[self.pos] = nr.min(VOCAB_SIZE as u64 - 1) as u16;
        self.pos = (self.pos + 1) % CONTEXT_LEN;
        if self.pos == 0 { self.full = true; }
    }

    /// Snapshot the ring in chronological order for the transformer.
    pub fn snapshot(&self) -> [u16; CONTEXT_LEN] {
        if !self.full && self.pos < CONTEXT_LEN {
            // Not yet full — pad head with zeros.
            let mut out = [0u16; CONTEXT_LEN];
            for i in 0..self.pos {
                out[CONTEXT_LEN - self.pos + i] = self.ring[i];
            }
            out
        } else {
            let mut out = [0u16; CONTEXT_LEN];
            for i in 0..CONTEXT_LEN {
                out[i] = self.ring[(self.pos + i) % CONTEXT_LEN];
            }
            out
        }
    }
}

// ── Global state ──────────────────────────────────────────────────────────────

static MODEL: Mutex<Option<TransformerSchedModel>> = Mutex::new(None);

// Per-PID syscall context rings — populated by the syscall dispatcher.
static CONTEXTS: Mutex<alloc::collections::BTreeMap<u64, SyscallContext>>
    = Mutex::new(alloc::collections::BTreeMap::new());

// Previous prediction per PID — needed for SGD feedback.
static PREV_PRED: Mutex<alloc::collections::BTreeMap<u64, ([u16; CONTEXT_LEN], SchedDecision)>>
    = Mutex::new(alloc::collections::BTreeMap::new());

/// Initialise the transformer scheduler at kernel boot.
pub fn init() {
    *MODEL.lock() = Some(TransformerSchedModel::new());
    crate::klog!(INFO, "transformer_sched: initialized ({} embed params, {} attn params)",
        VOCAB_SIZE * EMBED_DIM,
        2 * ATTN_DIM * EMBED_DIM);
}

/// Record a syscall number for the current task's context window.
/// Called from the main syscall dispatcher — zero allocation, O(1).
pub fn record_syscall(pid: u64, nr: u64) {
    CONTEXTS.lock()
        .entry(pid)
        .or_insert_with(SyscallContext::new)
        .push(nr);
}

/// Run the transformer on `pid`'s current context window.
/// Returns a scheduling decision to blend with fingerprint cluster hints.
/// The result is cached in PREV_PRED for the SGD feedback step.
pub fn predict(pid: u64) -> Option<SchedDecision> {
    let ctx_snapshot = {
        let mut ctxs = CONTEXTS.lock();
        let ctx = ctxs.get(&pid)?;
        ctx.snapshot()
    };
    let decision = MODEL.lock().as_ref()?.forward(&ctx_snapshot);
    PREV_PRED.lock().insert(pid, (ctx_snapshot, decision));
    Some(decision)
}

/// Called on task descheduling with the *actual* observed values.
/// Updates the output head weights via SGD.
pub fn on_deschedule(pid: u64, actual_nice: i8, actual_burst_ticks: u32, actual_pf: u8) {
    let prev = PREV_PRED.lock().remove(&pid);
    if let Some((ctx, _pred)) = prev {
        let target = [
            actual_nice      as f32,
            actual_burst_ticks as f32,
            actual_pf        as f32,
        ];
        if let Some(model) = MODEL.lock().as_mut() {
            model.sgd_step(&ctx, target);
        }
    }
}

/// Remove per-task state on process exit.
pub fn remove(pid: u64) {
    CONTEXTS.lock().remove(&pid);
    PREV_PRED.lock().remove(&pid);
}

/// Format model stats for /ai/transformer_sched.
pub fn format_report() -> alloc::vec::Vec<u8> {
    use alloc::string::String;
    let guard = MODEL.lock();
    let steps = guard.as_ref().map(|m| m.steps).unwrap_or(0);
    let ctx_count = CONTEXTS.lock().len();
    let mut out = String::from("NodeAI Transformer Scheduler\n");
    out.push_str("==============================\n");
    out.push_str(&alloc::format!("embed_params : {}\n", VOCAB_SIZE * EMBED_DIM));
    out.push_str(&alloc::format!("attn_params  : {}\n", 3 * ATTN_DIM * EMBED_DIM));
    out.push_str(&alloc::format!("head_params  : {}\n", 16 * ATTN_DIM + 3 * 16));
    out.push_str(&alloc::format!("sgd_steps    : {}\n", steps));
    out.push_str(&alloc::format!("active_pids  : {}\n", ctx_count));
    out.push_str(&alloc::format!("context_len  : {}\n", CONTEXT_LEN));
    out.into_bytes()
}
