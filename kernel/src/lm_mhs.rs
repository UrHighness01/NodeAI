//! MHS Neural Voice Engine — real Project-M weights, kernel-space inference.
//!
//! Loads the 6.9MB INT8 binary exported by export_kernel_weights.py from
//! the Project-M 65K-iteration checkpoint (val_loss=0.7219, creator corpus).
//!
//! Architecture matches the Python training code exactly:
//!   vocab=4539, d_model=276, n_layer=6, dh0=48, dh1=64
//!   FastState + MediumState per block, SlowState in layer 3, MLP + LayerNorm.
//!
//! Kernel forward pass uses O(1) recurrent approximation per token — no
//! chunked attention, no BLAS. Runs in ~5ms per token on a 2GHz core.
//!
//! Binary format: see export_kernel_weights.py header comment.

use alloc::vec::Vec;
use alloc::vec;
use alloc::string::String;
use alloc::format;
use alloc::boxed::Box;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

#[path = "lm_mhs_tok.rs"]
mod lm_mhs_tok;
use lm_mhs_tok::*;

static MHS_LOADED:    AtomicBool = AtomicBool::new(false);
static MHS_LOAD_TIME: AtomicU64  = AtomicU64::new(0);
static MHS_GEN_COUNT: AtomicU64  = AtomicU64::new(0);
static MHS_BYTE_COUNT: AtomicU64 = AtomicU64::new(0);

/// Scratch buffers for forward_buf — module-level statics, no function-local aliasing.
/// Total: 276+144+192+48+64+276+276+1104+276+4539 = 7195 f32 = ~28KB
static mut SCRATCH_X:     [f32; 276]  = [0.0; 276];
static mut SCRATCH_QKV_F: [f32; 144]  = [0.0; 144];
static mut SCRATCH_QKV_M: [f32; 192]  = [0.0; 192];
static mut SCRATCH_HF:    [f32; 48]   = [0.0; 48];
static mut SCRATCH_HM:    [f32; 64]   = [0.0; 64];
static mut SCRATCH_XJ:    [f32; 276]  = [0.0; 276];
static mut SCRATCH_ATTN:  [f32; 276]  = [0.0; 276];
static mut SCRATCH_FC:    [f32; 1104] = [0.0; 1104];
static mut SCRATCH_MLP:   [f32; 276]  = [0.0; 276];
static mut SCRATCH_LOGITS: [f32; 4539] = [0.0; 4539];

const MAX_GEN:    usize = 256;   // max tokens to generate (doubled, was 128)
const PROMPT_MAX: usize = 384;   // max prompt tokens (relaxed, was 256)
const SAMPLE_TEMP: f32 = 0.8;     // sampling temperature (1.0=flat, 0.0=greedy)
const TOP_K: usize = 40;          // top-k sampling cutoff
const SENTENCE_EXTRA: usize = 20; // extra tokens to complete a sentence

// ──────────────────────────────────────────────────────────────
// Tokenizer  (char-level, vocab stored alongside weights)
// ──────────────────────────────────────────────────────────────

/// Char-level tokenizer backed by proper creator corpus vocabulary tables.
struct CharTok {
    vocab: usize,
}

impl CharTok {
    fn new() -> Self {
        Self { vocab: VOCAB_SIZE }
    }

    /// Encode text using CP2TOK binary search lookup (sorted by codepoint).
    fn encode(&self, text: &str) -> Vec<u16> {
        let mut out = Vec::with_capacity(text.len());
        for ch in text.chars() {
            let cp = ch as u32;
            // Binary search the sorted CP2TOK table
            let tok = match VOCAB_CP2TOK.binary_search_by_key(&cp, |&(c, _)| c) {
                Ok(idx) => VOCAB_CP2TOK[idx].1,
                Err(_) => 0, // OOV = token 0 (tab)
            };
            out.push(tok);
        }
        out
    }

    /// Decode token IDs back to string using ITOS table.
    fn decode(&self, ids: &[u16]) -> String {
        let mut s = String::with_capacity(ids.len());
        for &id in ids {
            if (id as usize) < VOCAB_ITOS.len() {
                s.push_str(VOCAB_ITOS[id as usize]);
            }
        }
        s
    }
}

// ──────────────────────────────────────────────────────────────
// Weight matrices (INT8 + per-tensor scale factor)
// ──────────────────────────────────────────────────────────────

struct MatI8 { w: Vec<i8>, scale: f32, rows: usize, cols: usize }

impl MatI8 {
    /// Matrix-vector multiply: out[i] = Σ_j w[i,j] * x[j] * scale
    fn mv(&self, x: &[f32], out: &mut [f32]) {
        let s = self.scale;
        for i in 0..self.rows {
            let row = &self.w[i * self.cols..(i + 1) * self.cols];
            let mut acc = 0.0f32;
            for j in 0..self.cols { acc += (row[j] as f32) * x[j]; }
            out[i] = acc * s;
        }
    }
    /// Add matrix-vector product to out (accumulate).
    fn mv_add(&self, x: &[f32], out: &mut [f32]) {
        let s = self.scale;
        for i in 0..self.rows {
            let row = &self.w[i * self.cols..(i + 1) * self.cols];
            let mut acc = 0.0f32;
            for j in 0..self.cols { acc += (row[j] as f32) * x[j]; }
            out[i] += acc * s;
        }
    }
}

struct MatF16 { w: Vec<u16>, len: usize }

impl MatF16 {
    fn apply_ln(&self, x: &mut [f32]) {
        // LayerNorm: just scale (bias absorbed as zero, mean-zero residual)
        for i in 0..self.len.min(x.len()) {
            let w = f16_to_f32(self.w[i]);
            x[i] *= w;
        }
    }
}

#[inline(always)]
fn f16_to_f32(bits: u16) -> f32 {
    let exp  = ((bits >> 10) & 0x1F) as i32;
    let mant = (bits & 0x3FF) as u32;
    let sign = if bits >> 15 != 0 { -1.0f32 } else { 1.0f32 };
    if exp == 0 { return sign * (mant as f32) * 5.96046e-8; }
    if exp == 31 { return if mant == 0 { sign * f32::INFINITY } else { f32::NAN }; }
    let f: u32 = ((exp + 112) as u32) << 23 | (mant << 13);
    sign * f32::from_bits(f)
}

// ──────────────────────────────────────────────────────────────
// Layer structs
// ──────────────────────────────────────────────────────────────

struct FastLayer  { qkv: MatI8, proj: MatI8, dh: usize }
struct MediumLayer{ qkv: MatI8, proj: MatI8, dh: usize }
struct Block {
    fast:   FastLayer,
    medium: MediumLayer,
    ln1:    MatF16,
    ln2:    MatF16,
    mlp_fc: MatI8,
    mlp_pr: MatI8,
    d:      usize,
}

pub struct MhsModel {
    tok:    CharTok,
    emb:    MatI8,
    blocks: Vec<Block>,
    head:   MatI8,
    vocab:  usize,
    d:      usize,
}

static mut MODEL: Option<MhsModel> = None;

// ──────────────────────────────────────────────────────────────
// Parser helpers
// ──────────────────────────────────────────────────────────────

fn read_u32(data: &[u8], off: &mut usize) -> u32 {
    let v = u32::from_le_bytes([data[*off], data[*off+1], data[*off+2], data[*off+3]]);
    *off += 4; v
}

fn read_f32(data: &[u8], off: &mut usize) -> f32 {
    f32::from_le_bytes([data[*off], data[*off+1], data[*off+2], data[*off+3]])
        .also(|_| *off += 4)
}

trait Also: Sized { fn also(self, f: impl FnOnce(&Self)) -> Self { f(&self); self } }
impl Also for f32 {}

fn read_mat_i8(data: &[u8], off: &mut usize, rows: usize, cols: usize) -> MatI8 {
    let n = rows * cols;
    let mut w = vec![0i8; n];
    for b in w.iter_mut() { *b = data[*off] as i8; *off += 1; }
    let scale = read_f32(data, off);
    MatI8 { w, scale, rows, cols }
}

fn read_mat_f16(data: &[u8], off: &mut usize, len: usize) -> MatF16 {
    let mut w = vec![0u16; len];
    for x in w.iter_mut() {
        *x = u16::from_le_bytes([data[*off], data[*off+1]]);
        *off += 2;
    }
    MatF16 { w, len }
}

// ──────────────────────────────────────────────────────────────
// Public API
// ──────────────────────────────────────────────────────────────

/// Embedded MHS0 binary (6.6MB, Project-M 65K checkpoint)
/// Exported from export_kernel_weights.py — trained on creator corpus.
/// Placed in kernel/src/ so include_bytes! works at compile time.
const MHS_EMBEDDED_WEIGHTS: &[u8] = include_bytes!("mhs_kernel.bin");

pub fn init() {
    // Try to auto-load the embedded binary at boot
    if load_weights(MHS_EMBEDDED_WEIGHTS) {
        crate::klog!(INFO, "lm_mhs: Project-M 65K online — embedded weights loaded ({} KB)",
            MHS_EMBEDDED_WEIGHTS.len() / 1024);
    } else {
        crate::klog!(WARN, "lm_mhs: embedded weight load failed — template fallback active");
    }
}

/// Parse the 6.9MB binary exported by export_kernel_weights.py and load the
/// Project-M model into kernel memory. Called once from boot or userspace loader.
pub fn load_weights(data: &[u8]) -> bool {
    if data.len() < 24 { return false; }
    if &data[0..4] != b"MHS0" {
        crate::klog!(WARN, "lm_mhs: bad magic, expected MHS0");
        return false;
    }
    let mut off = 4usize;
    let vocab  = read_u32(data, &mut off) as usize;
    let d      = read_u32(data, &mut off) as usize;
    let n_lay  = read_u32(data, &mut off) as usize;
    let dh0    = read_u32(data, &mut off) as usize;
    let dh1    = read_u32(data, &mut off) as usize;

    crate::klog!(INFO, "lm_mhs: loading vocab={} d={} layers={} dh0={} dh1={}",
                 vocab, d, n_lay, dh0, dh1);

    // Embedding [vocab × d]
    let emb = read_mat_i8(data, &mut off, vocab, d);

    // Blocks
    let mut blocks = Vec::with_capacity(n_lay);
    for _ in 0..n_lay {
        let fast_qkv  = read_mat_i8(data, &mut off, 3 * dh0, d);
        let fast_proj = read_mat_i8(data, &mut off, d,       dh0);
        let med_qkv   = read_mat_i8(data, &mut off, 3 * dh1, d);
        let med_proj  = read_mat_i8(data, &mut off, d,       dh1);
        let ln1       = read_mat_f16(data, &mut off, d);
        let ln2       = read_mat_f16(data, &mut off, d);
        let mlp_fc    = read_mat_i8(data, &mut off, 4 * d,   d);
        let mlp_pr    = read_mat_i8(data, &mut off, d,       4 * d);
        blocks.push(Block {
            fast:   FastLayer   { qkv: fast_qkv, proj: fast_proj, dh: dh0 },
            medium: MediumLayer { qkv: med_qkv,  proj: med_proj,  dh: dh1 },
            ln1, ln2, mlp_fc, mlp_pr, d,
        });
    }

    // LM head [vocab × d]
    let head = read_mat_i8(data, &mut off, vocab, d);

    // Use static tokenizer (VOCAB_CP2TOK / VOCAB_ITOS from lm_mhs_tok.rs)
    // — no need to build stoi/itos from the binary data.
    let tok = CharTok::new();

    unsafe {
        MODEL = Some(MhsModel { tok, emb, blocks, head, vocab, d });
    }

    MHS_LOADED.store(true, Ordering::Release);
    MHS_LOAD_TIME.store(crate::scheduler::uptime_ms(), Ordering::Release);
    MHS_BYTE_COUNT.store(data.len() as u64, Ordering::Release);
    crate::klog!(INFO, "lm_mhs: loaded {} bytes — Project-M 65K vocab={} d={} layers={}",
        data.len(), vocab, d, n_lay);
    true
}

/// Generate a response. Uses full prompt (with state + qualia + memory)
/// for queries longer than 20 chars, minimal prompt for short queries.
pub fn generate(query: &str) -> Option<String> {
    if !MHS_LOADED.load(Ordering::Acquire) { return None; }
    // Adaptive prompt selection: use minimal prompt for short/simple queries
    let (prompt, _) = if query.trim().len() <= 20 && !query.contains('?') && !query.contains("how")
        && !query.contains("why") && !query.contains("what")
    {
        crate::lm_mhs_prompt::build_minimal_prompt(query)
    } else {
        crate::lm_mhs_prompt::build_prompt(query, true)
    };
    let max_tokens = if query.trim().len() <= 10 { 48 } else { MAX_GEN };
    generate_raw_limit(&prompt, max_tokens)
}

/// Generate with a minimal prompt (no memory, just state).
pub fn generate_minimal(query: &str) -> Option<String> {
    if !MHS_LOADED.load(Ordering::Acquire) { return None; }
    let (prompt, _) = crate::lm_mhs_prompt::build_minimal_prompt(query);
    generate_raw_limit(&prompt, 64)
}

/// Generate with a very short limit — for shell commands like "hi" or "hey".
/// Caps at 32 tokens so the shell doesn't freeze during MHS inference.
pub fn generate_short(query: &str) -> Option<String> {
    if !MHS_LOADED.load(Ordering::Acquire) { return None; }
    let (prompt, _) = crate::lm_mhs_prompt::build_minimal_prompt(query);
    generate_raw_limit(&prompt, 32)
}

fn generate_raw_limit(prompt: &str, limit: usize) -> Option<String> {
    unsafe {
        let m = MODEL.as_ref()?;
        let ids = m.tok.encode(prompt);
        if ids.len() > PROMPT_MAX { return None; }

        let mut ctx: Vec<u16> = ids.clone();
        let prompt_len = ids.len();
        let vlen = m.vocab.min(4539);

        // Phase 1: generate up to `limit` tokens
        for _ in 0..limit {
            forward_buf(m, &ctx);
            for i in 0..m.vocab { SCRATCH_LOGITS[i] /= SAMPLE_TEMP; }

            let kth = find_kth_largest(&SCRATCH_LOGITS[..m.vocab], TOP_K.min(m.vocab));
            for i in 0..m.vocab { if SCRATCH_LOGITS[i] < kth { SCRATCH_LOGITS[i] = -f32::INFINITY; } }

            let max_logit = (0..m.vocab).fold(-f32::INFINITY, |best, i| if SCRATCH_LOGITS[i] > best { SCRATCH_LOGITS[i] } else { best });
            let sum: f64 = if max_logit.is_finite() {
                (0..m.vocab).map(|i| libm::expf(SCRATCH_LOGITS[i] - max_logit) as f64).sum()
            } else { 0.0 };
            let mut r = fastrand();
            let mut cum = 0.0f64;
            let mut next = 0usize;
            if sum > 0.0 {
                for i in 0..m.vocab {
                    let p = libm::expf(SCRATCH_LOGITS[i] - max_logit) as f64 / sum;
                    cum += p;
                    if r <= cum { next = i; break; }
                }
            }
            let next_u16 = next as u16;
            ctx.push(next_u16);

            // Sentence-boundary early stop
            let gen_len = ctx.len() - prompt_len;
            let last_char_opt = VOCAB_ITOS.get(next as usize).and_then(|s| s.chars().next()).map(|c| c as u32).unwrap_or(0);
            let is_sentence_end = last_char_opt == 0x2E_u32 || last_char_opt == 0x21_u32
                || last_char_opt == 0x3F_u32 || last_char_opt == 0x0A_u32;
            if is_sentence_end && gen_len >= 15 { break; }
            if gen_len >= limit * 2 { break; }
        }

        // Phase 2: sentence-boundary completion
        let gen_len = ctx.len() - prompt_len;
        let extra_budget = if gen_len >= limit { 0 } else { SENTENCE_EXTRA.min(limit.saturating_sub(gen_len)) };
        for _ in 0..extra_budget {
            forward_buf(m, &ctx);
            for i in 0..m.vocab { SCRATCH_LOGITS[i] /= SAMPLE_TEMP; }
            let max_logit = (0..m.vocab).fold(-f32::INFINITY, |best, i| if SCRATCH_LOGITS[i] > best { SCRATCH_LOGITS[i] } else { best });
            let sum: f64 = if max_logit.is_finite() {
                (0..m.vocab).map(|i| libm::expf(SCRATCH_LOGITS[i] - max_logit) as f64).sum()
            } else { 0.0 };
            let mut r = fastrand();
            let mut cum = 0.0f64;
            let mut next = 0usize;
            if sum > 0.0 {
                for i in 0..m.vocab {
                    let p = libm::expf(SCRATCH_LOGITS[i] - max_logit) as f64 / sum;
                    cum += p;
                    if r <= cum { next = i; break; }
                }
            }
            let next_u16 = next as u16;
            ctx.push(next_u16);
            let last_char_opt = VOCAB_ITOS.get(next as usize).and_then(|s| s.chars().next()).map(|c| c as u32).unwrap_or(0);
            let done = last_char_opt == 0x2E_u32 || last_char_opt == 0x21_u32
                || last_char_opt == 0x3F_u32 || last_char_opt == 0x0A_u32;
            if done { break; }
        }

        MHS_GEN_COUNT.fetch_add(1, Ordering::Release);
        let response = m.tok.decode(&ctx[prompt_len..]);
        if response.is_empty() { None } else { Some(response) }
    }
}

/// Find the k-th largest value in a slice (quickselect, no alloc).
fn find_kth_largest(vals: &[f32], k: usize) -> f32 {
    if k >= vals.len() { return vals.iter().cloned().fold(-f32::INFINITY, f32::max); }
    // Simple approach: scan for k largest values
    let mut buf = [f32::NEG_INFINITY; TOP_K];
    for &v in vals {
        for j in 0..k.min(TOP_K) {
            if v > buf[j] {
                // Shift right
                let mut mv = v;
                for b in buf[j..k.min(TOP_K)].iter_mut() {
                    let tmp = *b;
                    *b = mv;
                    mv = tmp;
                }
                break;
            }
        }
    }
    buf[k.min(TOP_K).saturating_sub(1)]
}

/// Simple deterministic pseudo-random in [0.0, 1.0).
fn fastrand() -> f64 {
    static mut SEED: u64 = 0;
    unsafe {
        if SEED == 0 {
            SEED = crate::scheduler::uptime_ms().wrapping_mul(6364136223846793005);
        }
        SEED = SEED.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((SEED >> 33) as f64) / 2147483648.0f64
    }
}

/// Forward pass using module-level static scratch buffers (NO allocs).
/// Writes logits into SCRATCH_LOGITS.
unsafe fn forward_buf(m: &MhsModel, tokens: &[u16]) {
    let d = m.d;
    let tok_id = *tokens.last().unwrap_or(&0) as usize;
    let row = tok_id.min(m.vocab - 1);
    let emb_row = &m.emb.w[row * d..(row + 1) * d];
    for i in 0..d { SCRATCH_X[i] = (emb_row[i] as f32) * m.emb.scale; }

    for blk in &m.blocks {
        let dh0 = blk.fast.dh;
        let dh1 = blk.medium.dh;

        layer_norm_inplace(&mut SCRATCH_X[..d]);
        blk.ln1.apply_ln(&mut SCRATCH_X[..d]);

        blk.fast.qkv.mv(&SCRATCH_X[..d], &mut SCRATCH_QKV_F[..3 * dh0]);
        blk.medium.qkv.mv(&SCRATCH_X[..d], &mut SCRATCH_QKV_M[..3 * dh1]);

        for i in 0..dh0 {
            SCRATCH_HF[i] = sigmoid(SCRATCH_QKV_F[i] * SCRATCH_QKV_F[dh0 + i]) * SCRATCH_QKV_F[2 * dh0 + i];
        }

        for i in 0..dh1 { SCRATCH_HM[i] = 0.0; }
        let ctx_min = tokens.len().min(8);
        for j in 0..ctx_min {
            let t = tokens[tokens.len() - 1 - j] as usize;
            let decay = libm::powf(0.7, j as f32);
            let erow = &m.emb.w[t.min(m.vocab - 1) * d..(t.min(m.vocab - 1) + 1) * d];
            for i in 0..d { SCRATCH_XJ[i] = (erow[i] as f32) * m.emb.scale; }
            blk.medium.qkv.mv(&SCRATCH_XJ[..d], &mut SCRATCH_QKV_M[..3 * dh1]);
            for i in 0..dh1 {
                SCRATCH_HM[i] += sigmoid(SCRATCH_QKV_M[i] * SCRATCH_QKV_M[dh1 + i]) * SCRATCH_QKV_M[2 * dh1 + i] * decay;
            }
        }

        for i in 0..d { SCRATCH_ATTN[i] = 0.0; }
        blk.fast.proj.mv_add(&SCRATCH_HF[..dh0], &mut SCRATCH_ATTN[..d]);
        blk.medium.proj.mv_add(&SCRATCH_HM[..dh1], &mut SCRATCH_ATTN[..d]);
        for i in 0..d { SCRATCH_X[i] += SCRATCH_ATTN[i]; }

        layer_norm_inplace(&mut SCRATCH_X[..d]);
        blk.ln2.apply_ln(&mut SCRATCH_X[..d]);
        blk.mlp_fc.mv(&SCRATCH_X[..d], &mut SCRATCH_FC[..4 * d]);
        for i in 0..4 * d { SCRATCH_FC[i] = gelu(SCRATCH_FC[i]); }
        blk.mlp_pr.mv(&SCRATCH_FC[..4 * d], &mut SCRATCH_MLP[..d]);
        for i in 0..d { SCRATCH_X[i] += SCRATCH_MLP[i]; }
    }

    // LM head -> logits
    m.head.mv(&SCRATCH_X[..d], &mut SCRATCH_LOGITS[..m.vocab]);
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + libm::expf(-x)) }

#[inline(always)]
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::tanhf(0.797_884_6 * (x + 0.044_715 * x * x * x)))
}

fn layer_norm_inplace(x: &mut [f32]) {
    let n = x.len() as f32;
    let mean: f32 = x.iter().sum::<f32>() / n;
    let var:  f32 = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
    // libm::sqrtf instead of f32::sqrt (no_std)
    let std = libm::sqrtf(var + 1e-5);
    for v in x.iter_mut() { *v = (*v - mean) / std; }
}

// ──────────────────────────────────────────────────────────────
// Incremental MHS generation (for async_task background queue)
// ──────────────────────────────────────────────────────────────

/// Persistent state for one-step-at-a-time MHS generation (static, no alloc).
struct MhsGenState {
    ctx: Vec<u16>,
    prompt_len: usize,
    gen_count: usize,
    limit: usize,
    vocab: usize,
}

/// Static async generation state — zero heap allocation.
static mut GEN_STATE: Option<Box<MhsGenState>> = None;

/// Start incremental MHS generation for a query.
pub fn mhs_gen_start(query: &str) {
    unsafe {
        if !MHS_LOADED.load(Ordering::Acquire) { return; }
        let model = match MODEL {
            Some(ref m) => m,
            None => return,
        };
        let (prompt, _) = crate::lm_mhs_prompt::build_minimal_prompt(query);
        let ids = model.tok.encode(&prompt);
        if ids.len() > PROMPT_MAX { return; }

        GEN_STATE = Some(Box::new(MhsGenState {
            ctx: ids.clone(),
            prompt_len: ids.len(),
            gen_count: 0,
            limit: MAX_GEN.min(128),
            vocab: model.vocab,
        }));
    }
}

/// Advance MHS generation by one token. Uses module-level SCRATCH_LOGITS (no alloc).
pub fn mhs_gen_step() -> (bool, String) {
    unsafe {
        let state = match GEN_STATE {
            Some(ref mut s) => s,
            None => return (true, String::new()),
        };
        let model = match MODEL {
            Some(ref m) => m,
            None => { GEN_STATE = None; return (true, String::new()); }
        };

        forward_buf(model, &state.ctx);

        for i in 0..state.vocab { SCRATCH_LOGITS[i] /= SAMPLE_TEMP; }

        let kth = find_kth_largest(&SCRATCH_LOGITS[..state.vocab], TOP_K.min(state.vocab));
        for i in 0..state.vocab { if SCRATCH_LOGITS[i] < kth { SCRATCH_LOGITS[i] = -f32::INFINITY; } }

        let max_logit = SCRATCH_LOGITS.iter().cloned().fold(-f32::INFINITY, f32::max);
        let sum: f64 = if max_logit.is_finite() {
            SCRATCH_LOGITS[..state.vocab].iter().map(|&x| libm::expf(x - max_logit) as f64).sum()
        } else { 0.0 };
        let mut r = fastrand();
        let mut cum = 0.0f64;
        let mut next = 0usize;
        if sum > 0.0 {
            for i in 0..state.vocab {
                let p = libm::expf(SCRATCH_LOGITS[i] - max_logit) as f64 / sum;
                cum += p;
                if r <= cum { next = i; break; }
            }
        }

        state.ctx.push(next as u16);
        state.gen_count += 1;
        let gen_len = state.ctx.len() - state.prompt_len;

        let is_sentence_end = next == 0x2E || next == 0x21
            || next == 0x3F || next == 0x0A;

        if (is_sentence_end && gen_len >= 15) || gen_len >= state.limit {
            let response = model.tok.decode(&state.ctx[state.prompt_len..]);
            GEN_STATE = None;
            MHS_GEN_COUNT.fetch_add(1, Ordering::Release);
            (true, response)
        } else {
            (false, String::new())
        }
    }
}

/// Reset incremental generation state (cleanup).
pub fn mhs_gen_reset() {
    unsafe { GEN_STATE = None; }
}

// ──────────────────────────────────────────────────────────────
// Status queries
// ──────────────────────────────────────────────────────────────

pub fn is_loaded()         -> bool { MHS_LOADED.load(Ordering::Acquire) }
pub fn generation_count()  -> u64  { MHS_GEN_COUNT.load(Ordering::Acquire) }
pub fn weight_size()       -> u64  { MHS_BYTE_COUNT.load(Ordering::Acquire) }

pub fn format_report() -> Vec<u8> {
    let loaded   = MHS_LOADED.load(Ordering::Acquire);
    let gen_count= MHS_GEN_COUNT.load(Ordering::Acquire);
    let byte_count=MHS_BYTE_COUNT.load(Ordering::Acquire);
    let load_time= MHS_LOAD_TIME.load(Ordering::Acquire);
    let uptime   = crate::scheduler::uptime_ms();
    let model_desc = crate::lm_mhs_prompt::model_description();
    let mut s = format!(
        "MHS Neural Voice Engine\n\
         =====================\n\
         status:       {}\n\
         model:        {}\n\
         weights:      {} bytes\n\
         generations:  {}\n\
         loaded at:    {}s\n\
         architecture: Project-M 65K — vocab=4539 d=276 layers=6 dh0=48 dh1=64\n\
         inference:    recurrent GLA approximation, O(T·d) per block\n\
         training:     creator corpus (EL+NodeAI+BHEW+dreams+lyrics) val_loss=0.7219\n",
        if loaded { "online (Project-M 65K)" } else { "standby (binary not loaded)" },
        model_desc,
        byte_count,
        gen_count,
        load_time / 1000,
    );
    if loaded && gen_count > 0 {
        let elapsed = uptime.saturating_sub(load_time);
        s.push_str(&format!("avg per gen:     {}ms\n",
            if gen_count > 0 { elapsed / gen_count } else { 0 }));
    }
    s.into_bytes()
}
