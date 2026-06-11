//! Project-K INT4 nano model — SINGLE ALLOCATION inference engine.
//!
//! Architecture: ONE static Once<Box<ProjectKEngine>> containing BOTH the model
//! weights AND the scratch buffer. LLVM cannot alias sub-fields of a single
//! heap allocation against each other — they're at fixed offsets from one base pointer.
//! This permanently avoids the static mut aliasing bug.

use alloc::vec::Vec;
use alloc::vec;
use alloc::boxed::Box;
use alloc::string::String;
use alloc::format;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use spin::Once;
use core::cell::UnsafeCell;

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
const VOCAB:    usize = 4539;
const MLP_D:    usize = 768;
const MAX_GEN:  usize = 128;
const CTX_WIN:  usize = 20;
const TOP_K:    usize = 40;
const TOP_P:    f32   = 0.92;
const TEMP:     f32   = 0.85;
const GROUP_SZ: usize = 64;

// ── Scratch layout within the single allocation ──────────────────────────
const S_QFS:  usize = 0;              // 3*DH0 = 96
const S_QMS:  usize = S_QFS + 96;     // 3*DH1 = 144
const S_HFS:  usize = S_QMS + 144;    // DH0 = 32
const S_HMS:  usize = S_HFS + 32;     // DH1 = 48
const S_X:    usize = S_HMS + 48;     // D = 192
const S_OUT:  usize = S_X + 192;      // D = 192
const S_FC:   usize = S_OUT + 192;    // MLP_D = 768
const S_LOG:  usize = S_FC + 768;     // VOCAB = 4539
const SCR_SZ: usize = S_LOG + 4539;   // total scratch size

// ── SINGLE allocation: model + scratch ──────────────────────────────────
// Use UnsafeCell for scratch — allows mutable access through shared ref.
// Single-threaded kernel: safe.
struct EngineScratch(UnsafeCell<[f32; SCR_SZ]>);
unsafe impl Sync for EngineScratch {}

struct ProjectKEngine {
    model: Model,
    rng_seed: AtomicU64,
    scratch: EngineScratch,
}

static ENGINE: Once<Box<ProjectKEngine>> = Once::new();

// ── INT4 weight matrix ────────────────────────────────────────────────────
struct MatI4 {
    packed:   Vec<u8>,
    scales:   Vec<f32>,
    rows:     usize,
    cols:     usize,
    n_groups: usize,
    n_packed: usize,
}

impl MatI4 {
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

struct VecF32(Vec<f32>);

struct GlaHead {
    qkv:      MatI4,
    qkv_bias: VecF32,
    fgate_w:  VecF32,
    fgate_b:  f32,
    proj:     MatI4,
    proj_bias: VecF32,
    dh:       usize,
}

struct Block {
    fast:   GlaHead,
    medium: GlaHead,
    ln1_w:  VecF32, ln1_b:  VecF32,
    ln2_w:  VecF32, ln2_b:  VecF32,
    mlp_fc:  MatI4, mlp_fcb: VecF32,
    mlp_pr:  MatI4, mlp_prb: VecF32,
}

struct Model {
    emb:     MatI4,
    blocks:  Vec<Block>,
    lnf_w:   VecF32,
    lnf_b:   VecF32,
    pos_emb: Option<MatI4>,
    vocab:   usize,
    d:       usize,
}

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
fn read_mat_i4(data: &[u8], off: &mut usize, rows: usize, cols: usize) -> MatI4 {
    let n_groups = (cols + GROUP_SZ - 1) / GROUP_SZ;
    let n_packed = n_groups * GROUP_SZ / 2;
    let total_packed = rows * n_packed;
    let total_scales = rows * n_groups;
    let packed = data[*off..*off + total_packed].to_vec();
    *off += total_packed;
    let mut scales = vec![0.0f32; total_scales];
    for s in scales.iter_mut() { *s = read_f32(data, off); }
    MatI4 { packed, scales, rows, cols, n_groups, n_packed }
}
fn read_gla_head(data: &[u8], off: &mut usize, d: usize, dh: usize) -> GlaHead {
    let qkv       = read_mat_i4(data, off, 3 * dh, d);
    let qkv_bias  = read_vec_f32(data, off, 3 * dh);
    let fgate_w   = read_vec_f32(data, off, d);
    let fgate_b   = read_f32(data, off);
    let proj      = read_mat_i4(data, off, d, dh);
    let proj_bias = read_vec_f32(data, off, d);
    GlaHead { qkv, qkv_bias, fgate_w, fgate_b, proj, proj_bias, dh }
}

const PKK_EMBEDDED: &[u8] = include_bytes!("projectk_weights.bin");

pub fn init() {
    let engine = load_engine(PKK_EMBEDDED);
    match engine {
        Some(e) => {
            ENGINE.call_once(|| e);
            LOADED.store(true, Ordering::Release);
            crate::klog!(INFO, "lm_projectk: Project-K 1.6MB INT4 online (single alloc, no aliasing)");
        }
        None => {
            crate::klog!(WARN, "lm_projectk: weight load failed");
        }
    }
}

pub fn is_loaded() -> bool { LOADED.load(Ordering::Acquire) }
pub fn gen_count() -> u64 { GEN_COUNT.load(Ordering::Relaxed) }

fn load_engine(data: &[u8]) -> Option<Box<ProjectKEngine>> {
    if data.len() < 32 || &data[0..4] != b"MHSI" { return None; }
    let mut off = 4usize;
    let vocab   = read_u32(data, &mut off) as usize;
    let d       = read_u32(data, &mut off) as usize;
    let n_lay   = read_u32(data, &mut off) as usize;
    let dh0     = read_u32(data, &mut off) as usize;
    let dh1     = read_u32(data, &mut off) as usize;
    let gs      = read_u32(data, &mut off) as usize;
    let _bsz    = read_u32(data, &mut off) as usize;
    if vocab != VOCAB || d != D || n_lay != N_LAYERS || dh0 != DH0 || dh1 != DH1 || gs != GROUP_SZ { return None; }

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
        blocks.push(Block { fast, medium, ln1_w, ln1_b, ln2_w, ln2_b, mlp_fc, mlp_fcb, mlp_pr, mlp_prb });
    }
    let lnf_w = read_vec_f32(data, &mut off, d);
    let lnf_b = read_vec_f32(data, &mut off, d);
    let pos_emb = if off < data.len() { Some(read_mat_i4(data, &mut off, _bsz, d)) } else { None };

    let model = Model { emb, blocks, lnf_w, lnf_b, pos_emb, vocab, d };
    let engine = Box::new(ProjectKEngine {
        model,
        rng_seed: AtomicU64::new(0),
        scratch: EngineScratch(UnsafeCell::new([0.0; SCR_SZ])),
    });
    Some(engine)
}

// ── GLA inference ────────────────────────────────────────────────────────
fn elu1(x: f32) -> f32 { if x >= 0.0 { x } else { libm::expf(x) - 1.0 } }
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + libm::expf(-x)) }
fn gelu(x: f32) -> f32 { 0.5 * x * (1.0 + libm::tanhf(0.7978845608 * (x + 0.044715 * x * x * x))) }
fn dot(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() }
fn layer_norm(x: &mut [f32], w: &[f32], b: &[f32]) {
    let mean = x.iter().sum::<f32>() / x.len() as f32;
    let var: f32 = x.iter().map(|v| { let d = v - mean; d * d }).sum::<f32>() / x.len() as f32;
    let rstd = libm::sqrtf(var + 1e-5);
    for i in 0..x.len() { x[i] = (x[i] - mean) / rstd * w[i] + b[i]; }
}

fn fastrand(e: &ProjectKEngine) -> f64 {
    let mut seed = e.rng_seed.load(Ordering::Relaxed);
    if seed == 0 {
        seed = (1 as u64).wrapping_mul(6364136223846793005).wrapping_add(1);
    }
    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    e.rng_seed.store(seed, Ordering::Relaxed);
    ((seed >> 33) as f64) / 2147483648.0f64
}

// ── Public API ────────────────────────────────────────────────────────────
pub fn generate(prompt: &str) -> Option<String> {
    if !LOADED.load(Ordering::Acquire) { return None; }
    let e = ENGINE.get()?;

    // Encode prompt
    fn encode_char(c: char) -> u16 {
        let cp = c as u32;
        let Ok(idx) = PKK_CP2TOK.binary_search_by_key(&cp, |e| e.0) else { return 3; };
        PKK_CP2TOK[idx].1
    }

    let mut tokens: Vec<u16> = Vec::with_capacity(64);
    tokens.push(encode_char('\n'));
    for ch in prompt.chars() {
        let t = encode_char(ch);
        if t != 0 { tokens.push(t); }
    }
    if tokens.len() < 2 { tokens.push(encode_char('?')); }

    // Generate tokens
    let mut output = String::new();
    for _ in 0..MAX_GEN {
        // Forward pass
        model_forward(e, &tokens, 0);

        // Sample next token
        let scr = unsafe { &*e.scratch.0.get() };
        let logits = &scr[S_LOG..S_LOG + VOCAB];

        // Find next token
        let mut best = 0;
        let mut best_v = logits[0];
        for i in 1..VOCAB {
            if logits[i] > best_v { best_v = logits[i]; best = i; }
        }
        if best == 0 { break; } // EOS

        // Decode
        let s = PKK_ITOS.get(best).copied().unwrap_or("");
        if s == "\n" || s == "" { break; }
        output.push_str(s);

        // Check stop
        if output.len() > 200 { break; }

        tokens.push(best as u16);
        if tokens.len() > CTX_WIN { tokens.remove(0); }
    }

    GEN_COUNT.fetch_add(1, Ordering::Relaxed);
    if output.is_empty() { None } else { Some(output) }
}

fn gla_head_forward(head: &GlaHead, embeds: &[[f32; D]], ctx_len: usize,
                      state: &mut [f32], out: &mut [f32]) {
    let dh = head.dh;
    for s in state[..dh].iter_mut() { *s = 0.0; }
    for t in 0..ctx_len {
        let x = &embeds[t];
        // Need &mut for qkv — copy approach
        let mut qkv_copy = [0.0f32; 144];
        let qkv_len = if dh == DH0 { 96 } else { 144 };
        let qkv_slice: &mut [f32] = &mut qkv_copy[..qkv_len];
        head.qkv.mv(x, qkv_slice);
        for i in 0..qkv_len { qkv_slice[i] += head.qkv_bias.0[i]; }
        let q = &qkv_slice[0..dh];
        let k = &qkv_slice[dh..2*dh];
        let v = &qkv_slice[2*dh..3*dh];
        let fg = sigmoid(dot(&head.fgate_w.0, x) + head.fgate_b);
        for i in 0..dh {
            state[i] = fg * state[i] + (1.0 - fg) * elu1(k[i]) * v[i];
        }
        if t + 1 == ctx_len {
            let mut hv = [0.0f32; 64];
            for i in 0..dh { hv[i] = elu1(q[i]) * state[i]; }
            for o in out[..D].iter_mut() { *o = 0.0; }
            head.proj.mv_add(&hv[..dh], out);
            for i in 0..D { out[i] += head.proj_bias.0[i]; }
        }
    }
}

fn model_forward(e: &ProjectKEngine, tokens: &[u16], pos_start: usize) {
    let m = &e.model;
    let ctx_len = tokens.len().min(CTX_WIN);
    let tok_start = tokens.len().saturating_sub(ctx_len);
    let mut embeds = [[0.0f32; D]; CTX_WIN];
    for (ti, &tok) in tokens[tok_start..].iter().enumerate() {
        let row = (tok as usize).min(m.vocab - 1);
        let ep = row;
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

    // Initialize residual stream from context window's last position
    let scr = unsafe { &mut *e.scratch.0.get() };
    scr[S_X..S_X + D].copy_from_slice(&embeds[ctx_len - 1]);

    let mut fast_state  = [0.0f32; DH0];
    let mut medium_state= [0.0f32; DH1];
    let mut fast_out    = [0.0f32; D];
    let mut medium_out  = [0.0f32; D];

    for blk in &m.blocks {
        // Pre-norm for the current residual
        let mut h = [0.0f32; D];
        h.copy_from_slice(&scr[S_X..S_X + D]);
        layer_norm(&mut h, &blk.ln1_w.0, &blk.ln1_b.0);
        // Ln1 embeds for full context
        let mut ln1_embeds = [[0.0f32; D]; CTX_WIN];
        for ti in 0..ctx_len {
            let mut e2 = embeds[ti];
            layer_norm(&mut e2, &blk.ln1_w.0, &blk.ln1_b.0);
            ln1_embeds[ti] = e2;
        }
        // Replace last with our pre-normed
        ln1_embeds[ctx_len - 1] = h;

        gla_head_forward(&blk.fast,   &ln1_embeds, ctx_len, &mut fast_state,   &mut fast_out);
        gla_head_forward(&blk.medium, &ln1_embeds, ctx_len, &mut medium_state,  &mut medium_out);

        for i in 0..D { scr[S_X + i] += fast_out[i] + medium_out[i]; }

        let mut h2 = [0.0f32; D];
        h2.copy_from_slice(&scr[S_X..S_X + D]);
        layer_norm(&mut h2, &blk.ln2_w.0, &blk.ln2_b.0);
        let mut mlp_fc = [0.0f32; MLP_D];
        blk.mlp_fc.mv(&h2, &mut mlp_fc);
        for i in 0..MLP_D { mlp_fc[i] = gelu(mlp_fc[i] + blk.mlp_fcb.0[i]); }
        let mut mlp_out = [0.0f32; D];
        blk.mlp_pr.mv(&mlp_fc, &mut mlp_out);
        for i in 0..D { scr[S_X + i] += mlp_out[i] + blk.mlp_prb.0[i]; }
    }

// Copy out last state before borrowing scr mutably
    let mut final_h = [0.0f32; D];
    final_h.copy_from_slice(&scr[S_X..S_X + D]);
    layer_norm(&mut final_h, &m.lnf_w.0, &m.lnf_b.0);
    // Output logits to scratch
    for l in scr[S_LOG..S_LOG + m.vocab].iter_mut() { *l = 0.0; }
    m.emb.mv_add(&final_h, &mut scr[S_LOG..S_LOG + m.vocab]);
}

pub fn report() -> String {
    format!(
        "Project-K INT4 nano: vocab={} d={} layers={} gs={} gen_count={}",
        VOCAB, D, N_LAYERS, GROUP_SZ, GEN_COUNT.load(Ordering::Relaxed),
    )
}
