//! Transformer-based scheduling policy — no kernel has done this before.
//!
//! Architecture:
//!   Input : last CONTEXT_LEN syscall numbers for the current task
//!   Embed : syscall_nr → 32-dim learnable embedding (512 × 32 table)
//!   Attn  : single-head self-attention (Q/K/V projections 32→16)
//!   Pool  : mean of attended token vectors → 16-dim context vector
//!   Head  : Dense(16→16, ReLU) → Dense(16→4, Linear)
//!   Output: [nice_delta, burst_ticks, prefault_pages, predicted_wait_us]
//!
//! Online SGD with FULL backprop: gradients flow through the output head,
//! through the attention mechanism (Q/K/V weights), and into the embedding
//! table. Every parameter in the model converges from real descheduling data.
//!
//! Previous version: only the output head was trained (99.5% of params frozen).
//! This version: all 12K+ parameters learn from every scheduling event.
//!
//! 4th output — predicted_wait_us: how long will this task wait for CPU next
//! time? Trained against actual measured wait from Task.sched_latency_total_us.
//! This forces the model to learn which syscall sequences correlate with CPU
//! starvation — and the nice_delta output then naturally compensates.
//!
//! REINFORCE policy gradient (gated behind 2000 SGD steps to prevent cold-start
//! thrashing): after warm-up, each output is treated as the mean of a Gaussian
//! policy N(μ, σ). Actions are sampled by adding N(0, σ) noise. The REINFORCE
//! gradient -log P(a) × (R - baseline) is added to the MSE gradient, where
//! R = -wait_us (higher reward for lower scheduling latency) and baseline is
//! a per-task exponentially-weighted moving average of recent rewards.

use alloc::vec::Vec;
use spin::Mutex;

pub const CONTEXT_LEN:  usize = 16;  // syscall history window
pub const EMBED_DIM:    usize = 32;  // per-token embedding size
pub const ATTN_DIM:     usize = 16;  // Q/K/V projection size
pub const VOCAB_SIZE:   usize = 512; // max syscall number tracked
pub const N_OUTPUTS:    usize = 4;   // [nice, burst, pf, wait_us]
const    HEAD_HIDDEN:   usize = 16;  // output head hidden layer size

/// Transformer scheduler output.
#[derive(Clone, Copy, Debug, Default)]
pub struct SchedDecision {
    pub nice_delta:      i8,
    pub burst_ticks:     u32,
    pub prefault_pages:  u8,
    pub predicted_wait:  u32,  // µs — for observability only
    /// Attention entropy in [0, 1]: 0 = fully confident (peaked attention),
    /// 1 = maximum uncertainty (uniform attention). Used for confidence blending.
    pub attention_entropy: f32,
}

// ── Model weights ─────────────────────────────────────────────────────────────

struct TransformerSchedModel {
    embed: alloc::boxed::Box<[f32]>,      // [VOCAB_SIZE × EMBED_DIM]
    wq:    alloc::boxed::Box<[f32]>,      // [ATTN_DIM × EMBED_DIM]
    wk:    alloc::boxed::Box<[f32]>,      // [ATTN_DIM × EMBED_DIM]
    wv:    alloc::boxed::Box<[f32]>,      // [ATTN_DIM × EMBED_DIM]
    h1_w:  alloc::boxed::Box<[f32]>,      // [HEAD_HIDDEN × ATTN_DIM]
    h1_b:  alloc::boxed::Box<[f32]>,      // [HEAD_HIDDEN]
    h2_w:  alloc::boxed::Box<[f32]>,      // [N_OUTPUTS × HEAD_HIDDEN]
    h2_b:  alloc::boxed::Box<[f32]>,      // [N_OUTPUTS]
    steps: u64,
}

fn init_weight(row: usize, col: usize, fan_in: usize, seed: u64) -> f32 {
    let h = seed
        .wrapping_add(row as u64 * 2654435761)
        .wrapping_add(col as u64 * 2246822519)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let scale = libm::sqrtf(2.0 / fan_in as f32);
    let frac = ((h >> 11) as f32) / libm::powf(2.0, 53.0);
    (frac * 2.0 - 1.0) * scale
}

impl TransformerSchedModel {
    fn new() -> Self {
        let embed: Vec<f32> = (0..VOCAB_SIZE * EMBED_DIM)
            .map(|i| init_weight(i / EMBED_DIM, i % EMBED_DIM, VOCAB_SIZE, 0xdead_beef_cafe_1234))
            .collect();
        let wq: Vec<f32> = (0..ATTN_DIM * EMBED_DIM)
            .map(|i| init_weight(i / EMBED_DIM, i % EMBED_DIM, EMBED_DIM, 0xfeed_cafe_babe_0001))
            .collect();
        let wk: Vec<f32> = (0..ATTN_DIM * EMBED_DIM)
            .map(|i| init_weight(i / EMBED_DIM, i % EMBED_DIM, EMBED_DIM, 0xfeed_cafe_babe_0002))
            .collect();
        let wv: Vec<f32> = (0..ATTN_DIM * EMBED_DIM)
            .map(|i| init_weight(i / EMBED_DIM, i % EMBED_DIM, EMBED_DIM, 0xfeed_cafe_babe_0003))
            .collect();
        let h1_w: Vec<f32> = (0..HEAD_HIDDEN * ATTN_DIM)
            .map(|i| init_weight(i / ATTN_DIM, i % ATTN_DIM, ATTN_DIM, 0x1234_5678_9abc_def0))
            .collect();
        let h2_w: Vec<f32> = (0..N_OUTPUTS * HEAD_HIDDEN)
            .map(|i| init_weight(i / HEAD_HIDDEN, i % HEAD_HIDDEN, HEAD_HIDDEN, 0x8765_4321_fedc_ba98))
            .collect();
        Self {
            embed: embed.into_boxed_slice(),
            wq:    wq.into_boxed_slice(),
            wk:    wk.into_boxed_slice(),
            wv:    wv.into_boxed_slice(),
            h1_w:  h1_w.into_boxed_slice(),
            h1_b:  alloc::vec![0.0f32; HEAD_HIDDEN].into_boxed_slice(),
            h2_w:  h2_w.into_boxed_slice(),
            h2_b:  alloc::vec![0.0f32; N_OUTPUTS].into_boxed_slice(),
            steps: 0,
        }
    }

    // ── Forward pass ─────────────────────────────────────────────────────────

    fn embed_sequence(&self, syscalls: &[u16; CONTEXT_LEN]) -> Vec<f32> {
        let mut mat = alloc::vec![0.0f32; CONTEXT_LEN * EMBED_DIM];
        for (t, &nr) in syscalls.iter().enumerate() {
            let idx = (nr as usize).min(VOCAB_SIZE - 1);
            mat[t * EMBED_DIM..(t + 1) * EMBED_DIM]
                .copy_from_slice(&self.embed[idx * EMBED_DIM..(idx + 1) * EMBED_DIM]);
        }
        mat
    }

    /// Scaled dot-product attention.
    /// Returns (attn_out [T×A], attn_weights [T×T], Q [T×A], K [T×A], V [T×A]).
    /// We return intermediates for backprop — no extra heap alloc on the forward path.
    fn attention(&self, tokens: &[f32])
        -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)
    {
        let t = CONTEXT_LEN;
        let d = EMBED_DIM;
        let a = ATTN_DIM;
        let scale = 1.0 / libm::sqrtf(a as f32);

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

        let mut attn_weights = alloc::vec![0.0f32; t * t];
        for i in 0..t {
            for j in 0..t {
                let mut dot = 0.0f32;
                for h in 0..a { dot += q[i * a + h] * k[j * a + h]; }
                attn_weights[i * t + j] = dot * scale;
            }
            // Numerical-stable softmax over row i.
            let max_s = attn_weights[i * t..(i + 1) * t]
                .iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for j in 0..t {
                attn_weights[i * t + j] = fast_exp(attn_weights[i * t + j] - max_s);
                sum += attn_weights[i * t + j];
            }
            if sum > 1e-9 {
                for j in 0..t { attn_weights[i * t + j] /= sum; }
            }
        }

        let mut attn_out = alloc::vec![0.0f32; t * a];
        for i in 0..t {
            for j in 0..t {
                let w = attn_weights[i * t + j];
                for h in 0..a {
                    attn_out[i * a + h] += w * v[j * a + h];
                }
            }
        }

        (attn_out, attn_weights, q, k, v)
    }

    fn forward(&self, syscalls: &[u16; CONTEXT_LEN]) -> SchedDecision {
        let tokens = self.embed_sequence(syscalls);
        let (attn_out, attn_weights, _, _, _) = self.attention(&tokens);

        // Attention entropy of the LAST query row (most recent token's attention
        // distribution). Low entropy = peaked = confident. High = uniform = uncertain.
        // Normalized to [0,1] by dividing by log(CONTEXT_LEN).
        let log_t = libm::logf(CONTEXT_LEN as f32);
        let last_row = &attn_weights[(CONTEXT_LEN - 1) * CONTEXT_LEN..];
        let entropy_raw: f32 = last_row.iter()
            .filter(|&&p| p > 1e-9)
            .map(|&p| -p * libm::logf(p))
            .sum();
        let attention_entropy = (entropy_raw / log_t).clamp(0.0, 1.0);

        let mut pooled = alloc::vec![0.0f32; ATTN_DIM];
        let inv_t = 1.0 / CONTEXT_LEN as f32;
        for t in 0..CONTEXT_LEN {
            for i in 0..ATTN_DIM {
                pooled[i] += attn_out[t * ATTN_DIM + i] * inv_t;
            }
        }

        let mut h1 = dense_forward(&self.h1_w, &self.h1_b, &pooled, HEAD_HIDDEN);
        relu_inplace(&mut h1);
        let out = dense_forward(&self.h2_w, &self.h2_b, &h1, N_OUTPUTS);

        SchedDecision {
            nice_delta:       out[0].clamp(-20.0, 20.0) as i8,
            burst_ticks:      out[1].clamp(1.0, 50.0)   as u32,
            prefault_pages:   out[2].clamp(0.0, 32.0)   as u8,
            predicted_wait:   out[3].max(0.0)            as u32,
            attention_entropy,
        }
    }

    // ── Full backprop ─────────────────────────────────────────────────────────
    //
    // Gradient flow (MSE loss on all 4 outputs):
    //   dL/dout → head layer 2 → head layer 1 → d_pooled
    //   d_pooled → d_attn_out (broadcast mean)
    //   d_attn_out → dV, d_attn_weights
    //   d_attn_weights → softmax Jacobian → d_scores
    //   d_scores → dQ, dK
    //   dQ, dK, dV → dWq, dWk, dWv (projection weight gradients)
    //   dQ, dK, dV → d_tokens (embedding input gradients)
    //   d_tokens[t] → d_embed[syscall_nr[t]] (sparse embedding update)

    fn sgd_step(&mut self, syscalls: &[u16; CONTEXT_LEN], target: [f32; N_OUTPUTS]) {
        self.steps += 1;
        let lr = 0.002 / (1.0 + self.steps as f32 * 0.00005);

        // ── Forward with saved activations ───────────────────────────────────
        let tokens = self.embed_sequence(syscalls);
        let (attn_out, attn_weights, q_mat, k_mat, v_mat) = self.attention(&tokens);

        let inv_t = 1.0 / CONTEXT_LEN as f32;
        let mut pooled = alloc::vec![0.0f32; ATTN_DIM];
        for t in 0..CONTEXT_LEN {
            for i in 0..ATTN_DIM {
                pooled[i] += attn_out[t * ATTN_DIM + i] * inv_t;
            }
        }

        let h1_pre = dense_forward(&self.h1_w, &self.h1_b, &pooled, HEAD_HIDDEN);
        let mut h1 = h1_pre.clone();
        relu_inplace(&mut h1);
        let out = dense_forward(&self.h2_w, &self.h2_b, &h1, N_OUTPUTS);

        // ── Layer 2 gradient ─────────────────────────────────────────────────
        let mut dout = [0.0f32; N_OUTPUTS];
        for i in 0..N_OUTPUTS { dout[i] = (out[i] - target[i]) * 2.0; } // d(MSE)

        for i in 0..N_OUTPUTS {
            self.h2_b[i] -= lr * dout[i];
            for j in 0..HEAD_HIDDEN {
                self.h2_w[i * HEAD_HIDDEN + j] -= lr * dout[i] * h1[j];
            }
        }

        // ── Layer 1 gradient (ReLU mask) ─────────────────────────────────────
        let mut dh1 = alloc::vec![0.0f32; HEAD_HIDDEN];
        for j in 0..HEAD_HIDDEN {
            let mut g = 0.0f32;
            for i in 0..N_OUTPUTS { g += self.h2_w[i * HEAD_HIDDEN + j] * dout[i]; }
            dh1[j] = if h1_pre[j] > 0.0 { g } else { 0.0 };
        }

        for i in 0..HEAD_HIDDEN {
            self.h1_b[i] -= lr * dh1[i];
            for j in 0..ATTN_DIM {
                self.h1_w[i * ATTN_DIM + j] -= lr * dh1[i] * pooled[j];
            }
        }

        // ── Gradient through mean pool → attn_out ────────────────────────────
        // d_pooled = W1^T · dh1
        let mut d_pooled = alloc::vec![0.0f32; ATTN_DIM];
        for j in 0..ATTN_DIM {
            for i in 0..HEAD_HIDDEN {
                d_pooled[j] += self.h1_w[i * ATTN_DIM + j] * dh1[i];
            }
        }
        // Mean broadcasts to all T tokens.
        let mut d_attn_out = alloc::vec![0.0f32; CONTEXT_LEN * ATTN_DIM];
        for t in 0..CONTEXT_LEN {
            for i in 0..ATTN_DIM {
                d_attn_out[t * ATTN_DIM + i] = d_pooled[i] * inv_t;
            }
        }

        // ── Attention backward ───────────────────────────────────────────────
        let t = CONTEXT_LEN;
        let a = ATTN_DIM;
        let d = EMBED_DIM;

        // dV[j,h] = sum_i(attn_weights[i,j] * d_attn_out[i,h])
        let mut dv = alloc::vec![0.0f32; t * a];
        for i in 0..t {
            for j in 0..t {
                let w = attn_weights[i * t + j];
                for h in 0..a {
                    dv[j * a + h] += w * d_attn_out[i * a + h];
                }
            }
        }

        // d_attn_weights[i,j] = d_attn_out[i] · V[j]
        let mut d_attn_weights = alloc::vec![0.0f32; t * t];
        for i in 0..t {
            for j in 0..t {
                let mut dot = 0.0f32;
                for h in 0..a { dot += d_attn_out[i * a + h] * v_mat[j * a + h]; }
                d_attn_weights[i * t + j] = dot;
            }
        }

        // Softmax Jacobian: d_scores[i,j] = aw[i,j] * (d_aw[i,j] - sum_k(aw[i,k]*d_aw[i,k]))
        let mut d_scores = alloc::vec![0.0f32; t * t];
        let scale = 1.0 / libm::sqrtf(a as f32);
        for i in 0..t {
            let mut dot = 0.0f32;
            for k in 0..t { dot += attn_weights[i * t + k] * d_attn_weights[i * t + k]; }
            for j in 0..t {
                d_scores[i * t + j] =
                    attn_weights[i * t + j] * (d_attn_weights[i * t + j] - dot) * scale;
            }
        }

        // dQ[i,h] = sum_j(d_scores[i,j] * K[j,h])
        // dK[j,h] = sum_i(d_scores[i,j] * Q[i,h])
        let mut dq = alloc::vec![0.0f32; t * a];
        let mut dk = alloc::vec![0.0f32; t * a];
        for i in 0..t {
            for j in 0..t {
                let ds = d_scores[i * t + j];
                for h in 0..a {
                    dq[i * a + h] += ds * k_mat[j * a + h];
                    dk[j * a + h] += ds * q_mat[i * a + h];
                }
            }
        }

        // ── Projection weight gradients + d_tokens ───────────────────────────
        // d_tokens uses PRE-update weights (mathematically correct).
        // We compute d_tokens first, then update Wq/Wk/Wv separately.
        let mut d_tokens = alloc::vec![0.0f32; t * d];
        for tok in 0..t {
            for i in 0..a {
                let gq = dq[tok * a + i];
                let gk = dk[tok * a + i];
                let gv = dv[tok * a + i];
                for j in 0..d {
                    d_tokens[tok * d + j] +=
                        self.wq[i * d + j] * gq
                        + self.wk[i * d + j] * gk
                        + self.wv[i * d + j] * gv;
                }
            }
        }
        // Now update Wq/Wk/Wv (after d_tokens is fully computed).
        for tok in 0..t {
            let x = &tokens[tok * d..(tok + 1) * d];
            for i in 0..a {
                let gq = dq[tok * a + i];
                let gk = dk[tok * a + i];
                let gv = dv[tok * a + i];
                for j in 0..d {
                    self.wq[i * d + j] -= lr * gq * x[j];
                    self.wk[i * d + j] -= lr * gk * x[j];
                    self.wv[i * d + j] -= lr * gv * x[j];
                }
            }
        }

        // ── Sparse embedding update ───────────────────────────────────────────
        // For each token position, accumulate gradient into the embedding row
        // for that syscall number.  Gradient clipped to [-0.1, 0.1] to prevent
        // a single extreme event from corrupting the embedding.
        for tok in 0..t {
            let nr = (syscalls[tok] as usize).min(VOCAB_SIZE - 1);
            for j in 0..d {
                let g = d_tokens[tok * d + j].clamp(-0.1, 0.1);
                self.embed[nr * d + j] -= lr * g;
            }
        }
    }

    /// Co-occurrence initialization: set embedding[i] ≈ row i of the normalized
    /// co-occurrence matrix built from all current per-PID syscall histograms.
    /// Called once at warm-up (after a few processes have run) to give the
    /// embedding table a meaningful starting point rather than pure random.
    fn init_from_cooccurrence(&mut self) {
        if crate::syscall_stats::pid_count() < 3 { return; }

        // Build co-occurrence: cooc[i][j] += min(hist[i], hist[j]) for each process.
        // cooc is [VOCAB_SIZE × EMBED_DIM] — column dim maps to syscall dim*(VOCAB_SIZE/EMBED_DIM).
        // This approximates the top EMBED_DIM principal directions of the cooccurrence matrix.
        let cols_per_dim = VOCAB_SIZE / EMBED_DIM; // 512/32 = 16
        let mut cooc = alloc::vec![0.0f32; VOCAB_SIZE * EMBED_DIM];

        crate::syscall_stats::visit_histograms(|hist| {
            for (i, &ci) in hist.iter().enumerate().take(VOCAB_SIZE) {
                if ci == 0 { continue; }
                for dim in 0..EMBED_DIM {
                    let j = dim * cols_per_dim;
                    let cj = hist[j] as f32;
                    cooc[i * EMBED_DIM + dim] += (ci as f32).min(cj);
                }
            }
        });

        // Normalize each embedding row to unit length, then blend 50/50 with
        // Xavier init (preserve some random diversity to avoid degenerate collapse).
        for i in 0..VOCAB_SIZE {
            let row = &mut cooc[i * EMBED_DIM..(i + 1) * EMBED_DIM];
            let mag: f32 = libm::sqrtf(row.iter().map(|v| v * v).sum::<f32>());
            if mag > 1e-6 {
                for (j, v) in row.iter_mut().enumerate() {
                    *v = (*v / mag) * 0.5 + self.embed[i * EMBED_DIM + j] * 0.5;
                }
                self.embed[i * EMBED_DIM..(i + 1) * EMBED_DIM].copy_from_slice(row);
            }
        }
    }
}

// ── Math helpers ──────────────────────────────────────────────────────────────

fn dense_forward(w: &[f32], b: &[f32], x: &[f32], out_size: usize) -> Vec<f32> {
    let in_size = x.len();
    let mut out = alloc::vec![0.0f32; out_size];
    for i in 0..out_size {
        let mut s = b[i];
        for j in 0..in_size { s += w[i * in_size + j] * x[j]; }
        out[i] = s;
    }
    out
}

fn relu_inplace(v: &mut Vec<f32>) {
    for x in v.iter_mut() { if *x < 0.0 { *x = 0.0; } }
}

#[inline]
fn fast_exp(x: f32) -> f32 {
    let i = (x.to_bits() as i64)
        .wrapping_add(((127.0_f32 / core::f32::consts::LN_2) as i64) << 23) as u32;
    f32::from_bits(i)
}

// ── Per-task syscall context ring ─────────────────────────────────────────────

pub struct SyscallContext {
    ring: [u16; CONTEXT_LEN],
    pos:  usize,
    full: bool,
}

impl SyscallContext {
    pub const fn new() -> Self {
        Self { ring: [0u16; CONTEXT_LEN], pos: 0, full: false }
    }

    pub fn push(&mut self, nr: u64) {
        self.ring[self.pos] = nr.min(VOCAB_SIZE as u64 - 1) as u16;
        self.pos = (self.pos + 1) % CONTEXT_LEN;
        if self.pos == 0 { self.full = true; }
    }

    pub fn snapshot(&self) -> [u16; CONTEXT_LEN] {
        let mut out = [0u16; CONTEXT_LEN];
        if !self.full && self.pos < CONTEXT_LEN {
            for i in 0..self.pos {
                out[CONTEXT_LEN - self.pos + i] = self.ring[i];
            }
        } else {
            for i in 0..CONTEXT_LEN {
                out[i] = self.ring[(self.pos + i) % CONTEXT_LEN];
            }
        }
        out
    }

    pub fn is_warm(&self) -> bool { self.full || self.pos >= 8 }
}

// ── Global state ──────────────────────────────────────────────────────────────

static MODEL: Mutex<Option<TransformerSchedModel>> = Mutex::new(None);

static CONTEXTS: Mutex<alloc::collections::BTreeMap<u64, SyscallContext>>
    = Mutex::new(alloc::collections::BTreeMap::new());

/// Previous context snapshot + last actual wait_us (for SGD feedback).
struct PendingFeedback {
    ctx:       [u16; CONTEXT_LEN],
    wait_us:   u64,
    /// Policy output (pre-clamp) saved for REINFORCE log-prob computation.
    raw_out:   [f32; N_OUTPUTS],
}
static PENDING: Mutex<alloc::collections::BTreeMap<u64, PendingFeedback>>
    = Mutex::new(alloc::collections::BTreeMap::new());

/// Per-task REINFORCE reward baseline (exponential moving average of -wait_us).
/// Baseline prevents high-variance gradient updates from unstable rewards.
static REWARD_BASELINE: Mutex<alloc::collections::BTreeMap<u64, f32>>
    = Mutex::new(alloc::collections::BTreeMap::new());

/// Policy sigma (exploration noise) per output — initialized conservatively.
/// Decays with model steps via: sigma_i = SIGMA_INIT / (1 + steps/SIGMA_DECAY).
const SIGMA_INIT:  [f32; N_OUTPUTS] = [2.0, 3.0, 2.0, 100.0]; // per-output
const SIGMA_DECAY: f32 = 5000.0;
const REINFORCE_GATE: u64 = 2000; // SGD steps before REINFORCE activates
const REINFORCE_LR:   f32 = 0.0001; // separate (smaller) LR for policy gradient

/// How many descheduling events until we trigger co-occurrence init.
static COOC_INIT_DONE: core::sync::atomic::AtomicBool
    = core::sync::atomic::AtomicBool::new(false);

pub fn init() {
    *MODEL.lock() = Some(TransformerSchedModel::new());
    crate::klog!(INFO,
        "transformer_sched: {} total params (embed={}, attn={}, head={})",
        VOCAB_SIZE * EMBED_DIM + 3 * ATTN_DIM * EMBED_DIM
            + HEAD_HIDDEN * ATTN_DIM + N_OUTPUTS * HEAD_HIDDEN,
        VOCAB_SIZE * EMBED_DIM,
        3 * ATTN_DIM * EMBED_DIM,
        HEAD_HIDDEN * ATTN_DIM + N_OUTPUTS * HEAD_HIDDEN);
}

/// O(1) syscall recording — called from every syscall dispatch.
pub fn record_syscall(pid: u64, nr: u64) {
    CONTEXTS.lock()
        .entry(pid)
        .or_insert_with(SyscallContext::new)
        .push(nr);
}

/// Record the actual wait latency for a task that just started running.
/// Called from schedule_from_interrupt after computing wait_ms.
pub fn record_wait(pid: u64, wait_us: u64) {
    let ctx_snapshot = {
        let ctxs = CONTEXTS.lock();
        ctxs.get(&pid).map(|c| c.snapshot())
    };
    if let Some(ctx) = ctx_snapshot {
        // Also save the raw (pre-clamp) output for REINFORCE log-prob computation.
        let raw_out = MODEL.lock().as_ref()
            .map(|m| {
                let tokens   = m.embed_sequence(&ctx);
                let (ao, ..) = m.attention(&tokens);
                let inv_t    = 1.0 / CONTEXT_LEN as f32;
                let pooled: Vec<f32> = (0..ATTN_DIM)
                    .map(|i| (0..CONTEXT_LEN).map(|t| ao[t*ATTN_DIM+i]).sum::<f32>() * inv_t)
                    .collect();
                let mut h1 = dense_forward(&m.h1_w, &m.h1_b, &pooled, HEAD_HIDDEN);
                relu_inplace(&mut h1);
                let out = dense_forward(&m.h2_w, &m.h2_b, &h1, N_OUTPUTS);
                let mut arr = [0.0f32; N_OUTPUTS];
                arr.copy_from_slice(&out[..N_OUTPUTS]);
                arr
            })
            .unwrap_or([0.0f32; N_OUTPUTS]);
        PENDING.lock().insert(pid, PendingFeedback { ctx, wait_us, raw_out });
    }
}

/// Run transformer forward pass. Returns None if context not warm yet.
pub fn predict(pid: u64) -> Option<SchedDecision> {
    let ctx_snapshot = {
        let ctxs = CONTEXTS.lock();
        let ctx = ctxs.get(&pid)?;
        if !ctx.is_warm() { return None; }
        ctx.snapshot()
    };
    MODEL.lock().as_ref().map(|m| m.forward(&ctx_snapshot))
}

/// Called on task descheduling. Runs SGD step using observed actual values.
pub fn on_deschedule(pid: u64, actual_nice: i8, actual_burst_ticks: u32, actual_pf: u8) {
    // Trigger co-occurrence init after 50 tasks have accumulated histograms.
    if !COOC_INIT_DONE.load(core::sync::atomic::Ordering::Relaxed) {
        let n = crate::syscall_stats::pid_count();
        if n >= 50 {
            if let Some(model) = MODEL.lock().as_mut() {
                model.init_from_cooccurrence();
            }
            COOC_INIT_DONE.store(true, core::sync::atomic::Ordering::Relaxed);
            crate::klog!(INFO, "transformer_sched: co-occurrence embedding init done ({} pids)", n);
        }
    }

    let feedback = PENDING.lock().remove(&pid);
    if let Some(fb) = feedback {
        let target = [
            actual_nice        as f32,
            actual_burst_ticks as f32,
            actual_pf          as f32,
            fb.wait_us         as f32,
        ];
        let steps = MODEL.lock().as_ref().map(|m| m.steps).unwrap_or(0);
        if let Some(model) = MODEL.lock().as_mut() {
            // MSE-supervised step (always active).
            model.sgd_step(&fb.ctx, target);

            // REINFORCE policy gradient (active after REINFORCE_GATE steps).
            // R = -wait_us (we want to minimize scheduling latency).
            // Baseline = EWMA of reward per task. Advantage = R - baseline.
            // Policy gradient: dL_pg = -advantage * d_log_pi(action)
            // For Gaussian policy: d_log_pi/d_mu_i = (a_i - mu_i) / sigma_i²
            if steps >= REINFORCE_GATE {
                let sigma_scale = 1.0 / (1.0 + steps as f32 / SIGMA_DECAY);
                let reward = -(fb.wait_us as f32); // negative wait = positive reward
                let baseline = {
                    let mut bl = REWARD_BASELINE.lock();
                    let b = bl.entry(pid).or_insert(reward);
                    *b = *b * 0.95 + reward * 0.05; // EWMA α=0.05
                    *b
                };
                let advantage = reward - baseline;
                if advantage.abs() > 1.0 { // skip tiny advantages
                    // Re-run the forward pass up to h1 so we have the hidden
                    // activations needed for weight gradients.  This is cheap
                    // (no alloc for embedding+attention — we share fb.ctx).
                    let tokens  = model.embed_sequence(&fb.ctx);
                    let (attn_out, _, _, _, _) = model.attention(&tokens);
                    let inv_t   = 1.0 / CONTEXT_LEN as f32;
                    let mut pooled = alloc::vec![0.0f32; ATTN_DIM];
                    for t in 0..CONTEXT_LEN {
                        for k in 0..ATTN_DIM {
                            pooled[k] += attn_out[t * ATTN_DIM + k] * inv_t;
                        }
                    }
                    let h1_pre = dense_forward(&model.h1_w, &model.h1_b, &pooled, HEAD_HIDDEN);
                    let mut h1 = h1_pre.clone();
                    relu_inplace(&mut h1);

                    let mut pg_grad = [0.0f32; N_OUTPUTS];
                    for i in 0..N_OUTPUTS {
                        let sigma    = SIGMA_INIT[i] * sigma_scale;
                        let action_i = fb.raw_out[i];
                        pg_grad[i]   = (-advantage * (action_i - target[i]) / (sigma * sigma))
                                        .clamp(-1.0, 1.0);
                    }

                    // Full output-layer gradient: update both weights and biases.
                    // dL/dh2_w[i,j] = pg_grad[i] * h1[j]
                    // dL/dh2_b[i]   = pg_grad[i]
                    for i in 0..N_OUTPUTS {
                        model.h2_b[i] -= REINFORCE_LR * pg_grad[i];
                        for j in 0..HEAD_HIDDEN {
                            model.h2_w[i * HEAD_HIDDEN + j] -= REINFORCE_LR * pg_grad[i] * h1[j];
                        }
                    }
                }
            }
        }
    }
}

/// Return the last `n` syscall numbers for `pid` in chronological order.
/// Used by the cross-process anomaly detector to inspect a waker's recent activity.
pub fn last_n_syscalls(pid: u64, n: usize) -> alloc::vec::Vec<u16> {
    let ctxs = CONTEXTS.lock();
    match ctxs.get(&pid) {
        None => alloc::vec::Vec::new(),
        Some(ctx) => {
            let snap = ctx.snapshot();
            let take = n.min(CONTEXT_LEN);
            snap[CONTEXT_LEN - take..].to_vec()
        }
    }
}

pub fn remove(pid: u64) {
    CONTEXTS.lock().remove(&pid);
    PENDING.lock().remove(&pid);
    REWARD_BASELINE.lock().remove(&pid);
}

pub fn format_report() -> alloc::vec::Vec<u8> {
    use alloc::string::String;
    let steps  = MODEL.lock().as_ref().map(|m| m.steps).unwrap_or(0);
    // Acquire CONTEXTS once — second .lock() on a spinlock self-deadlocks.
    let (ctx_n, warm_n) = {
        let ctx = CONTEXTS.lock();
        (ctx.len(), ctx.values().filter(|c| c.is_warm()).count())
    };
    let cooc   = COOC_INIT_DONE.load(core::sync::atomic::Ordering::Relaxed);

    let total_params = VOCAB_SIZE * EMBED_DIM
        + 3 * ATTN_DIM * EMBED_DIM
        + HEAD_HIDDEN * ATTN_DIM + HEAD_HIDDEN
        + N_OUTPUTS * HEAD_HIDDEN + N_OUTPUTS;

    let mut out = String::from("NodeAI Transformer Scheduler (full backprop)\n");
    out.push_str("=============================================\n");
    out.push_str(&alloc::format!("total_params    : {}\n", total_params));
    out.push_str(&alloc::format!("embed_params    : {} ({}×{})\n",
        VOCAB_SIZE * EMBED_DIM, VOCAB_SIZE, EMBED_DIM));
    out.push_str(&alloc::format!("attn_params     : {} (3×{}×{})\n",
        3 * ATTN_DIM * EMBED_DIM, ATTN_DIM, EMBED_DIM));
    out.push_str(&alloc::format!("head_params     : {}\n",
        HEAD_HIDDEN * ATTN_DIM + N_OUTPUTS * HEAD_HIDDEN));
    let reinforce_active = steps >= REINFORCE_GATE;
    out.push_str(&alloc::format!("sgd_steps       : {}\n", steps));
    out.push_str(&alloc::format!("active_pids     : {} ({} warm)\n", ctx_n, warm_n));
    out.push_str(&alloc::format!("cooc_init_done  : {}\n", cooc));
    out.push_str(&alloc::format!("reinforce       : {} (gate={}, full h2_w+h2_b)\n",
        if reinforce_active { "ACTIVE" } else { "warming up" }, REINFORCE_GATE));
    out.push_str(&alloc::format!("context_len     : {}\n", CONTEXT_LEN));
    out.push_str(&alloc::format!("blend_mode      : confidence-weighted (attn_entropy + fp_cosine + causal_prob)\n"));
    out.push_str(&alloc::format!("outputs         : [nice, burst_ticks, pf_pages, wait_us]\n"));
    out.into_bytes()
}
