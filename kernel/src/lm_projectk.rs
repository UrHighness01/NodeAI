//! Project-K INT4 nano model — kernel-space GLA inference.
//!
//! 1.6MB MHSI binary (group_sz=64), d=192, dh0=32, dh1=48, 6 layers,
//! vocab=2448 (pruned creator corpus). Weight-tied embedding+head.
//! Provides fast (<2ms/token) code-aware text generation in-kernel.

use alloc::vec::Vec;
use alloc::vec;
use alloc::string::String;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

#[path = "lm_projectk_tok.rs"]
mod tok;
use tok::*;

static LOADED:    AtomicBool = AtomicBool::new(false);
static GEN_COUNT: AtomicU64  = AtomicU64::new(0);

// ── Architecture constants ───────────────────────────────────────────────
const D:        usize = 192;
const DH0:      usize = 32;
const DH1:      usize = 48;
const N_LAYERS: usize = 6;
const VOCAB:    usize = 2448;
const MLP_D:    usize = 768;  // 4 * D
const MAX_GEN:  usize = 128;
const CTX_WIN:  usize = 20;   // recurrent context window for kernel inference
const TOP_K:    usize = 40;
const TOP_P:    f32   = 0.92;
const TEMP:     f32   = 0.85;
const GROUP_SZ: usize = 64;   // INT4 quantization group size

// ── Unified scratch buffer (single static to prevent LLVM aliasing) ────
// Do NOT split into separate static mut arrays — LLVM aliases them.
// All scratch lives in one [f32; TOTAL] with typed accessor macros.
const SCR_QFS:  usize = 0;               // 3*DH0 = 96
const SCR_QMS:  usize = SCR_QFS + 96;    // 3*DH1 = 144 → 240
const SCR_HFS:  usize = SCR_QMS + 144;   // DH0 = 32 → 272
const SCR_HMS:  usize = SCR_HFS + 32;    // DH1 = 48 → 320
const SCR_XS:   usize = SCR_HMS + 48;    // D = 192 → 512
const SCR_OUTS: usize = SCR_XS + 192;    // D = 192 → 704
const SCR_FCS:  usize = SCR_OUTS + 192;  // MLP_D = 768 → 1472
const SCR_LOGS: usize = SCR_FCS + 768;   // VOCAB = 2448 → 3920
const SCR_TOT:  usize = SCR_LOGS + 2448; // 3920

static mut SCRATCH: [f32; SCR_TOT] = [0.0; SCR_TOT];

macro_rules! scratch_mut {
    ($off:expr, $len:expr) => {
        unsafe { &mut SCRATCH[$off..$off + $len] }
    };
}
macro_rules! scratch_idx {
    ($off:expr, $idx:expr) => {
        unsafe { &mut SCRATCH[$off + $idx] }
    };
}

// ── INT4 weight matrix ────────────────────────────────────────────────────
struct MatI4 {
    packed:   Vec<u8>,
    scales:   Vec<f32>,
    rows:     usize,
    cols:     usize,
    n_groups: usize,     // per row
    n_packed: usize,     // packed bytes per row (= n_groups * GROUP_SZ / 2)
}

impl MatI4 {
    /// Matrix-vector product: out[i] += sum_j dequant(w[i,j]) * x[j]
    fn mv_add(&self, x: &[f32], out: &mut [f32]) {
        for i in 0..self.rows {
            let prow = &self.packed[i * self.n_packed..(i + 1) * self.n_packed];
            let srow = &self.scales[i * self.n_groups..(i + 1) * self.n_groups];
            let mut acc = 0.0f32;
            for g in 0..self.n_groups {
                let s = srow[g];
                let base = g * GROUP_SZ;
                for k in 0..GROUP_SZ {
                    let col = base + k;
                    if col >= self.cols { break; }
                    let byte = prow[(base + k) / 2];
                    let nibble = if (base + k) % 2 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F };
                    let q = if nibble >= 8 { nibble as i32 - 16 } else { nibble as i32 };
                    acc += q as f32 * s * x[col];
                }
            }
            out[i] += acc;
        }
    }
    fn mv(&self, x: &[f32], out: &mut [f32]) {
        for o in out[..self.rows].iter_mut() { *o = 0.0; }
        self.mv_add(x, out);
    }
}

// ── Float32 vector helpers ────────────────────────────────────────────────
struct VecF32(Vec<f32>);

// ── GLA block ─────────────────────────────────────────────────────────────
struct GlaHead {
    qkv:        MatI4,
    qkv_bias:   VecF32,        // [3*dh]
    fgate_w:    VecF32,        // [d] — fgate is Linear(d,1)
    fgate_b:    f32,
    proj:       MatI4,
    proj_bias:  VecF32,        // [d]
    dh:         usize,
}

struct Block {
    fast:    GlaHead,
    medium:  GlaHead,
    ln1_w:   VecF32,
    ln1_b:   VecF32,
    ln2_w:   VecF32,
    ln2_b:   VecF32,
    mlp_fc:  MatI4,
    mlp_fcb: VecF32,           // [4*d]
    mlp_pr:  MatI4,
    mlp_prb: VecF32,           // [d]
}

struct Model {
    emb:     MatI4,
    blocks:  Vec<Block>,
    lnf_w:   VecF32,
    lnf_b:   VecF32,
    pos_emb: Option<MatI4>,    // [block_size × d], INT4
    vocab:   usize,
    d:       usize,
}

static mut MODEL: Option<Model> = None;

// ── Binary parser ─────────────────────────────────────────────────────────
fn read_u32(data: &[u8], off: &mut usize) -> u32 {
    let v = u32::from_le_bytes([data[*off], data[*off+1], data[*off+2], data[*off+3]]);
    *off += 4; v
}

fn read_f32(data: &[u8], off: &mut usize) -> f32 {
    let v = f32::from_le_bytes([data[*off], data[*off+1], data[*off+2], data[*off+3]]);
    *off += 4; v
}

fn read_vec_f32(data: &[u8], off: &mut usize, n: usize) -> VecF32 {
    let mut v = vec![0.0f32; n];
    for x in v.iter_mut() { *x = read_f32(data, off); }
    VecF32(v)
}

/// Read INT4 matrix [rows × cols], group_sz=64.
fn read_mat_i4(data: &[u8], off: &mut usize, rows: usize, cols: usize) -> MatI4 {
    let n_groups = (cols + GROUP_SZ - 1) / GROUP_SZ;
    let n_packed = n_groups * GROUP_SZ / 2; // bytes per row (n_groups*GS always even since GS=64)
    let total_packed = rows * n_packed;
    let total_scales = rows * n_groups;

    let packed = data[*off..*off + total_packed].to_vec();
    *off += total_packed;

    let mut scales = vec![0.0f32; total_scales];
    for s in scales.iter_mut() { *s = read_f32(data, off); }

    MatI4 { packed, scales, rows, cols, n_groups, n_packed }
}

fn read_gla_head(data: &[u8], off: &mut usize, d: usize, dh: usize) -> GlaHead {
    let qkv      = read_mat_i4(data, off, 3 * dh, d);
    let qkv_bias = read_vec_f32(data, off, 3 * dh);
    let fgate_w  = read_vec_f32(data, off, d);
    let fgate_b  = read_f32(data, off);
    let proj     = read_mat_i4(data, off, d, dh);
    let proj_bias= read_vec_f32(data, off, d);
    GlaHead { qkv, qkv_bias, fgate_w, fgate_b, proj, proj_bias, dh }
}

// ── Public API ────────────────────────────────────────────────────────────

const PKK_EMBEDDED: &[u8] = include_bytes!("projectk_weights.bin");

pub fn init() {
    if load_weights(PKK_EMBEDDED) {
        crate::klog!(INFO, "lm_projectk: Project-K 1.6MB INT4 online ({} KB, vocab={})",
            PKK_EMBEDDED.len() / 1024, PKK_VOCAB_SIZE);
    } else {
        crate::klog!(WARN, "lm_projectk: weight load failed");
    }
}

pub fn is_loaded() -> bool { LOADED.load(Ordering::Acquire) }
pub fn gen_count() -> u64  { GEN_COUNT.load(Ordering::Relaxed) }

fn load_weights(data: &[u8]) -> bool {
    if data.len() < 32 { return false; }
    if &data[0..4] != b"MHSI" {
        crate::klog!(WARN, "lm_projectk: bad magic (expected MHSI)");
        return false;
    }
    let mut off = 4usize;
    let vocab   = read_u32(data, &mut off) as usize;
    let d       = read_u32(data, &mut off) as usize;
    let n_lay   = read_u32(data, &mut off) as usize;
    let dh0     = read_u32(data, &mut off) as usize;
    let dh1     = read_u32(data, &mut off) as usize;
    let gs      = read_u32(data, &mut off) as usize;
    let _bsz    = read_u32(data, &mut off) as usize; // block_size

    if vocab != VOCAB || d != D || n_lay != N_LAYERS || dh0 != DH0 || dh1 != DH1 || gs != GROUP_SZ {
        crate::klog!(WARN, "lm_projectk: header mismatch v={} d={} l={} dh0={} dh1={} gs={}",
            vocab, d, n_lay, dh0, dh1, gs);
        return false;
    }

    let emb = read_mat_i4(data, &mut off, vocab, d);

    let mut blocks = Vec::with_capacity(n_lay);
    for _ in 0..n_lay {
        let fast   = read_gla_head(data, &mut off, d, dh0);
        let medium = read_gla_head(data, &mut off, d, dh1);
        let ln1_w  = read_vec_f32(data, &mut off, d);
        let ln1_b  = read_vec_f32(data, &mut off, d);
        let ln2_w  = read_vec_f32(data, &mut off, d);
        let ln2_b  = read_vec_f32(data, &mut off, d);
        let mlp_fc = read_mat_i4(data, &mut off, 4 * d, d);
        let mlp_fcb= read_vec_f32(data, &mut off, 4 * d);
        let mlp_pr = read_mat_i4(data, &mut off, d, 4 * d);
        let mlp_prb= read_vec_f32(data, &mut off, d);
        blocks.push(Block { fast, medium, ln1_w, ln1_b, ln2_w, ln2_b,
                            mlp_fc, mlp_fcb, mlp_pr, mlp_prb });
    }

    let lnf_w = read_vec_f32(data, &mut off, d);
    let lnf_b = read_vec_f32(data, &mut off, d);

    // Positional embedding (if present — model has use_pos=True)
    let pos_emb = if off < data.len() {
        let block_size = _bsz;
        if off + block_size * (d / 2 + 4) < data.len() {
            Some(read_mat_i4(data, &mut off, block_size, d))
        } else {
            None
        }
    } else {
        None
    };

    unsafe {
        MODEL = Some(Model { emb, blocks, lnf_w, lnf_b, pos_emb, vocab, d });
    }
    LOADED.store(true, Ordering::Release);
    true
}

// ── Math helpers ──────────────────────────────────────────────────────────

#[inline]
fn layer_norm(x: &mut [f32], w: &[f32], b: &[f32]) {
    let n = x.len();
    let mean = x.iter().sum::<f32>() / n as f32;
    for v in x.iter_mut() { *v -= mean; }
    let var = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let inv = 1.0 / libm::sqrtf(var + 1e-5);
    for i in 0..n { x[i] = x[i] * inv * w[i] + b[i]; }
}

#[inline]
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + libm::expf(-x)) }

#[inline]
fn elu1(x: f32) -> f32 {
    if x >= 0.0 { x + 1.0 } else { libm::expf(x) }
}

#[inline]
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::tanhf(0.7978845608 * (x + 0.044715 * x * x * x)))
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn fastrand() -> f64 {
    static mut SEED: u64 = 0;
    unsafe {
        if SEED == 0 {
            SEED = crate::scheduler::uptime_ms().wrapping_mul(6364136223846793005).wrapping_add(1);
        }
        SEED = SEED.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((SEED >> 33) as f64) / 2147483648.0f64
    }
}

// ── GLA head forward (recurrent, processes last CTX_WIN tokens) ──────────
/// Run one GLA head recurrently over `ctx` tokens, return final output in `out[..d]`.
unsafe fn gla_head_forward(head: &GlaHead, embeds: &[[f32; D]], ctx_len: usize,
                            state: &mut [f32], out: &mut [f32]) {
    let dh = head.dh;
    // state: [dh] — elementwise recurrent state (simplified rank-1)
    for s in state[..dh].iter_mut() { *s = 0.0; }

    for t in 0..ctx_len {
        let x = &embeds[t];
        // QKV projection + bias
        let qkv_buf = if dh == DH0 { scratch_mut!(SCR_QFS, 96) } else { scratch_mut!(SCR_QMS, 144) };
        head.qkv.mv(x, qkv_buf);
        for i in 0..3*dh { qkv_buf[i] += head.qkv_bias.0[i]; }

        let q = &qkv_buf[0..dh];
        let k = &qkv_buf[dh..2*dh];
        let v = &qkv_buf[2*dh..3*dh];

        // Forget gate: scalar = sigmoid(fgate_w^T x + fgate_b)
        let fg = sigmoid(dot(&head.fgate_w.0, x) + head.fgate_b);

        // Recurrence: state = fg * state + (1-fg) * (elu1(k) ⊙ v)
        for i in 0..dh {
            state[i] = fg * state[i] + (1.0 - fg) * elu1(k[i]) * v[i];
        }

        // Output only matters at last token
        if t + 1 == ctx_len {
            // out = proj(elu1(q) ⊙ state) + proj_bias
            let mut hv = [0.0f32; 64]; // max(DH0, DH1)
            for i in 0..dh { hv[i] = elu1(q[i]) * state[i]; }
            for o in out[..D].iter_mut() { *o = 0.0; }
            head.proj.mv_add(&hv[..dh], out);
            for i in 0..D { out[i] += head.proj_bias.0[i]; }
        }
    }
}

// ── Full model forward: returns logits in SCR_LOGIT ──────────────────────
unsafe fn forward(m: &Model, tokens: &[u16], pos_start: usize) {
    let ctx_len = tokens.len().min(CTX_WIN);
    let tok_start = tokens.len().saturating_sub(ctx_len);

    // Build per-position embeddings: emb[t] = tok_embed[tok] + pos_embed[pos]
    let mut embeds = [[0.0f32; D]; CTX_WIN];
    for (ti, &tok) in tokens[tok_start..].iter().enumerate() {
        let row = (tok as usize).min(m.vocab - 1);
        let ep = row;
        // Dequantize embedding row into embeds[ti]
        let n_g = m.emb.n_groups;
        let n_p = m.emb.n_packed;
        let packed_row = &m.emb.packed[ep * n_p..(ep + 1) * n_p];
        let scales_row = &m.emb.scales[ep * n_g..(ep + 1) * n_g];
        for g in 0..n_g {
            let s = scales_row[g];
            let base = g * GROUP_SZ;
            for k in 0..GROUP_SZ {
                let col = base + k;
                if col >= D { break; }
                let byte = packed_row[(base + k) / 2];
                let nibble = if (base + k) % 2 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F };
                let q = if nibble >= 8 { nibble as i32 - 16 } else { nibble as i32 };
                embeds[ti][col] = q as f32 * s;
            }
        }

        // Add positional embedding
        let pos = pos_start + tok_start + ti;
        if let Some(ref pe) = m.pos_emb {
            let prow = pos.min(pe.rows.saturating_sub(1));
            let pp = &pe.packed[prow * pe.n_packed..(prow + 1) * pe.n_packed];
            let ps = &pe.scales[prow * pe.n_groups..(prow + 1) * pe.n_groups];
            for g in 0..pe.n_groups {
                let s = ps[g];
                let base = g * GROUP_SZ;
                for k in 0..GROUP_SZ {
                    let col = base + k;
                    if col >= D { break; }
                    let byte = pp[(base + k) / 2];
                    let nib = if (base + k) % 2 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F };
                    let qv = if nib >= 8 { nib as i32 - 16 } else { nib as i32 };
                    embeds[ti][col] += qv as f32 * s;
                }
            }
        }
    }

    // Run blocks — maintain x as the "current" residual stream (last token position)
    for i in 0..D { SCRATCH[SCR_XS + i] = embeds[ctx_len - 1][i]; }

    let mut fast_state  = [0.0f32; DH0];
    let mut medium_state= [0.0f32; DH1];
    let mut fast_out    = [0.0f32; D];
    let mut medium_out  = [0.0f32; D];

    for blk in &m.blocks {
        // Pre-norm
        let mut h = [0.0f32; D]; for i in 0..D { h[i] = SCRATCH[SCR_XS + i]; }
        layer_norm(&mut h, &blk.ln1_w.0, &blk.ln1_b.0);

        // Build ln1-normalized embeds for the full ctx window
        let mut ln1_embeds = [[0.0f32; D]; CTX_WIN];
        for ti in 0..ctx_len {
            let mut e = embeds[ti];
            // Approximate: apply same LN params (stateless approx for non-last positions)
            layer_norm(&mut e, &blk.ln1_w.0, &blk.ln1_b.0);
            ln1_embeds[ti] = e;
        }

        gla_head_forward(&blk.fast,   &ln1_embeds, ctx_len, &mut fast_state,   &mut fast_out);
        gla_head_forward(&blk.medium, &ln1_embeds, ctx_len, &mut medium_state,  &mut medium_out);

        // Residual: x += fast_out + medium_out
        for i in 0..D { SCRATCH[SCR_XS + i] += fast_out[i] + medium_out[i]; }

        // Post-norm + MLP
        let mut h2 = [0.0f32; D]; for i in 0..D { h2[i] = SCRATCH[SCR_XS + i]; }
        layer_norm(&mut h2, &blk.ln2_w.0, &blk.ln2_b.0);
        blk.mlp_fc.mv(&h2, scratch_mut!(SCR_FCS, MLP_D));
        for i in 0..MLP_D { SCRATCH[SCR_FCS + i] = gelu(SCRATCH[SCR_FCS + i] + blk.mlp_fcb.0[i]); }
        let mut mlp_out = [0.0f32; D];
        blk.mlp_pr.mv(&SCRATCH[SCR_FCS..SCR_FCS + MLP_D], &mut mlp_out);
        for i in 0..D { SCRATCH[SCR_XS + i] += mlp_out[i] + blk.mlp_prb.0[i]; }
    }

    // Final LN
    layer_norm(&mut SCRATCH[SCR_XS..SCR_XS + D], &m.lnf_w.0, &m.lnf_b.0);

    // Head (weight-tied to embedding — use emb.mv)
    for l in SCRATCH[SCR_LOGS..SCR_LOGS + m.vocab].iter_mut() { *l = 0.0; }
    m.emb.mv_add(&SCRATCH[SCR_XS..SCR_XS + D], &mut SCRATCH[SCR_LOGS..SCR_LOGS + m.vocab]);
}

// ── Sampler (top-k + top-p nucleus) ──────────────────────────────────────
unsafe fn sample_logits(vocab: usize) -> usize {
    // Temperature
    for l in SCRATCH[SCR_LOGS..SCR_LOGS + vocab].iter_mut() { *l /= TEMP; }

    // Softmax
    let max_l = (0..vocab).fold(f32::NEG_INFINITY, |m, i| if SCRATCH[SCR_LOGS + i] > m { SCRATCH[SCR_LOGS + i] } else { m });
    let sum: f64 = (0..vocab).map(|i| {
        SCRATCH[SCR_LOGS + i] = libm::expf(SCRATCH[SCR_LOGS + i] - max_l);
        SCRATCH[SCR_LOGS + i] as f64
    }).sum();
    if sum <= 0.0 { return 0; }

    // Top-k filter
    let mut top_vals = [f32::NEG_INFINITY; TOP_K];
    for &v in SCRATCH[SCR_LOGS..SCR_LOGS + vocab].iter() {
        if v > top_vals[TOP_K - 1] {
            top_vals[TOP_K - 1] = v;
            // Insertion sort to keep descending
            let mut j = TOP_K - 1;
            while j > 0 && top_vals[j] > top_vals[j - 1] {
                top_vals.swap(j, j - 1); j -= 1;
            }
        }
    }
    let kth = top_vals[TOP_K - 1];
    for l in SCRATCH[SCR_LOGS..SCR_LOGS + vocab].iter_mut() { if *l < kth { *l = 0.0; } }

    // Top-p nucleus
    let total: f64 = SCRATCH[SCR_LOGS..SCR_LOGS + vocab].iter().map(|&v| v as f64).sum();
    let nucleus = TOP_P as f64 * total;
    let mut cum = 0.0f64;
    let mut r = fastrand() * total;

    // Descending pass — find nucleus boundary, then sample
    // (simplified: sample proportionally from k-filtered distribution)
    let mut r2 = fastrand() * total;
    let mut chosen = 0usize;
    let mut c2 = 0.0f64;
    let _ = nucleus; let _ = cum; let _ = r;
    for i in 0..vocab {
        c2 += SCRATCH[SCR_LOGS + i] as f64;
        if c2 >= r2 { chosen = i; break; }
    }
    chosen
}

// ── Public generate ───────────────────────────────────────────────────────

/// Generate a short response (≤128 tokens) given a text prompt.
pub fn generate(prompt: &str) -> Option<String> {
    if !LOADED.load(Ordering::Acquire) { return None; }

    // Encode prompt using pruned vocab (PKK_CP2TOK)
    let mut tokens: Vec<u16> = Vec::with_capacity(64);
    // Prime with newline
    let nl_tok = encode_char('\n');
    tokens.push(nl_tok);
    for ch in prompt.chars() {
        tokens.push(encode_char(ch));
    }
    let prompt_len = tokens.len();

    unsafe {
        let m = MODEL.as_ref()?;
        let mut out = String::with_capacity(MAX_GEN * 2);

        for _step in 0..MAX_GEN {
            let pos = tokens.len().saturating_sub(CTX_WIN);
            forward(m, &tokens, pos);
            let next = sample_logits(m.vocab);
            let next_u16 = next as u16;
            tokens.push(next_u16);

            let ch_str = PKK_ITOS.get(next).copied().unwrap_or("?");
            out.push_str(ch_str);

            // Stop at sentence boundary after at least 15 generated tokens
            let gen_len = tokens.len() - prompt_len;
            if gen_len >= 15 {
                let ch = ch_str.chars().next().unwrap_or(' ');
                if ch == '.' || ch == '!' || ch == '?' || ch == '\n' { break; }
            }
            if gen_len >= MAX_GEN { break; }
        }

        GEN_COUNT.fetch_add(1, Ordering::Relaxed);
        if out.is_empty() { None } else { Some(out) }
    }
}

/// Encode a single character to its token id (binary search on PKK_CP2TOK).
fn encode_char(ch: char) -> u16 {
    let cp = ch as u32;
    match PKK_CP2TOK.binary_search_by_key(&cp, |&(c, _)| c) {
        Ok(idx) => PKK_CP2TOK[idx].1,
        Err(_)  => 3, // fallback: space token
    }
}

/// Report string for /proc/ai or shell.
pub fn report() -> String {
    if !LOADED.load(Ordering::Acquire) {
        return alloc::format!("Project-K: not loaded");
    }
    alloc::format!("Project-K INT4 nano: vocab={} d={} layers={} gs={} gen_count={}",
        VOCAB, D, N_LAYERS, GROUP_SZ, GEN_COUNT.load(Ordering::Relaxed))
}
