//! Qwen2.x / Qwen3.x inference engine — no_std, bare-metal, dynamic architecture.
//!
//! Supports any Qwen variant (2.5-0.5B, 3.5-0.6B, etc.) with a single runtime
//! config read from the weight binary header.
//!
//! Architecture: GQA transformer with RoPE, SwiGLU FFN, RMSNorm, INT4 group-quant.
//!
//! Weights loaded at runtime from a second QEMU disk (device index 1, raw sectors).
//! Binary format produced by scripts/convert_qwen_kernel.py.
//!
//! Memory breakdown (Qwen2.5-0.5B):
//!   Weights  : ~320 MB (INT4 group-quantized)
//!   KV cache : ~13  MB (f32, MAX_CTX=512, 24 layers, 2 KV heads, head_dim=64)
//!   Scratch  : ~200 KB (pre-allocated in model struct)

use alloc::vec::Vec;
use alloc::string::String;
use alloc::format;
use alloc::boxed::Box;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

#[path = "lm_qwen_tok.rs"]
mod tok;
pub use tok::{EOS_TOKEN, BOS_TOKEN, IM_START, IM_END};

// ── Compile-time limits ───────────────────────────────────────────────────────
pub const VOCAB:    usize = 151936;  // same for all current Qwen models
pub const MAX_CTX:  usize = 512;     // KV cache depth; tune up if RAM allows
pub const GS:       usize = 32;      // INT4 group size
pub const MAX_GEN:  usize = 128;
pub const TEMP:     f32   = 0.75;
pub const TOP_K:    usize = 40;
pub const REP_PEN:  f32   = 1.15;

// ── Runtime architecture config ───────────────────────────────────────────────
#[derive(Clone, Copy)]
struct Arch {
    n_layers:  usize,
    d_model:   usize,
    n_heads:   usize,
    n_kv:      usize,
    ffn_dim:   usize,
    head_dim:  usize,  // = d_model / n_heads
    q_dim:     usize,  // = n_heads * head_dim = d_model
    kv_dim:    usize,  // = n_kv   * head_dim
    rope_theta: f32,
}

// ── Weight matrix descriptor ──────────────────────────────────────────────────
#[derive(Clone, Copy, Default)]
struct MatQ {
    nib_off:  usize,  // byte offset of nibbles section in w
    sc_off:   usize,  // byte offset of scales section in w
    rows:     usize,
    cols:     usize,
    n_groups: usize,  // = ceil(cols / GS)
}

impl MatQ {
    fn alloc(rows: usize, cols: usize, base: usize) -> (Self, usize) {
        let n_groups   = (cols + GS - 1) / GS;
        let nib_bytes  = rows * n_groups * (GS / 2);
        let sc_bytes   = rows * n_groups * 4;
        let m = MatQ { nib_off: base, sc_off: base + nib_bytes, rows, cols, n_groups };
        (m, base + nib_bytes + sc_bytes)
    }
}

// ── Layer weights ─────────────────────────────────────────────────────────────
struct Layer {
    attn_norm: Vec<f32>,
    q_w: MatQ, q_b: Vec<f32>,
    k_w: MatQ, k_b: Vec<f32>,
    v_w: MatQ, v_b: Vec<f32>,
    o_w: MatQ,
    ffn_norm: Vec<f32>,
    gate_w: MatQ,
    up_w:   MatQ,
    down_w: MatQ,
}

// ── Full model ────────────────────────────────────────────────────────────────
struct QwenModel {
    arch:       Arch,
    w:          Vec<u8>,          // raw weight bytes (INT4 + scales)
    layers:     Vec<Layer>,
    out_norm:   Vec<f32>,
    lm_head:    MatQ,
    emb:        MatQ,
    /// KV cache flat layout: [layer * kv_heads * MAX_CTX * head_dim]
    /// Index: layer_base(li) + head_base(h) + pos * head_dim + d
    k_cache:    Vec<f32>,
    v_cache:    Vec<f32>,
    /// Pre-allocated scratch to avoid per-token allocs
    scratch_x:      Vec<f32>,   // [d_model]
    scratch_xnorm:  Vec<f32>,   // [d_model]
    scratch_q:      Vec<f32>,   // [q_dim]
    scratch_k:      Vec<f32>,   // [kv_dim]
    scratch_v:      Vec<f32>,   // [kv_dim]
    scratch_attn:   Vec<f32>,   // [q_dim]
    scratch_gate:   Vec<f32>,   // [ffn_dim]
    scratch_up:     Vec<f32>,   // [ffn_dim]
    scratch_din:    Vec<f32>,   // [ffn_dim]
    scratch_o:      Vec<f32>,   // [d_model]
    scratch_ffn:    Vec<f32>,   // [d_model]
    scratch_logits: Vec<f32>,   // [VOCAB]
    scratch_scores: Vec<f32>,   // [MAX_CTX]
    rng:            AtomicU64,
}

static mut ENGINE: Option<Box<QwenModel>> = None;
static LOADED:    AtomicBool = AtomicBool::new(false);
static GEN_COUNT: AtomicU64  = AtomicU64::new(0);

// ── Math helpers ──────────────────────────────────────────────────────────────

#[inline(always)]
fn rms_norm(w: &[f32], x: &[f32], out: &mut [f32]) {
    let n = x.len().min(w.len()).min(out.len());
    let mut ss = 0.0f32;
    for i in 0..n { ss += x[i] * x[i]; }
    let inv = 1.0 / libm::sqrtf(ss / n as f32 + 1e-6);
    for i in 0..n { out[i] = x[i] * inv * w[i]; }
}

#[inline(always)]
fn silu(x: f32) -> f32 { x / (1.0 + libm::expf(-x)) }

fn rope_inplace(buf: &mut [f32], pos: usize, n_heads: usize, head_dim: usize, theta: f32) {
    for h in 0..n_heads {
        let base = h * head_dim;
        let mut d = 0usize;
        while d + 1 < head_dim {
            let freq = libm::powf(theta, -(d as f32) / head_dim as f32);
            let angle = pos as f32 * freq;
            let (s, c) = (libm::sinf(angle), libm::cosf(angle));
            let x0 = buf[base + d];
            let x1 = buf[base + d + 1];
            buf[base + d]     = x0 * c - x1 * s;
            buf[base + d + 1] = x0 * s + x1 * c;
            d += 2;
        }
    }
}

// ── INT4 matmul ───────────────────────────────────────────────────────────────

/// Dequantize row `row` of matrix `m` into `out[0..m.cols]`.
#[inline]
fn dequant_row(w: &[u8], m: &MatQ, row: usize, out: &mut [f32]) {
    let ng       = m.n_groups;
    let nib_row  = m.nib_off + row * ng * (GS / 2);
    let sc_row   = m.sc_off  + row * ng * 4;
    let cols     = m.cols;
    let mut ci   = 0usize;
    for g in 0..ng {
        let sc = f32::from_le_bytes(w[sc_row + g*4 .. sc_row + g*4+4].try_into().unwrap_or([0;4]));
        let nb = nib_row + g * (GS / 2);
        let end = GS.min(cols - ci);
        for i in 0..end {
            let byte    = w[nb + i/2];
            let nibble  = if i & 1 == 0 { (byte & 0xF) as i8 } else { (byte >> 4) as i8 };
            let nibble_se = if nibble > 7 { nibble - 16 } else { nibble };
            out[ci + i] = nibble_se as f32 * sc;
        }
        ci += end;
        if ci >= cols { break; }
    }
}

/// y += W @ x  (both y and x are contiguous slices)
fn matmul_add(w: &[u8], m: &MatQ, x: &[f32], y: &mut [f32]) {
    let ng          = m.n_groups;
    let nib_stride  = ng * (GS / 2);
    let sc_stride   = ng * 4;
    let cols        = m.cols;
    for r in 0..m.rows {
        let nb   = m.nib_off + r * nib_stride;
        let sc   = m.sc_off  + r * sc_stride;
        let mut acc = 0.0f32;
        let mut ci  = 0usize;
        for g in 0..ng {
            let scale = f32::from_le_bytes(w[sc + g*4 .. sc + g*4+4].try_into().unwrap_or([0;4]));
            let base  = nb + g * (GS / 2);
            let end   = GS.min(cols - ci);
            let mut ga = 0.0f32;
            for i in 0..end {
                let byte     = w[base + i/2];
                let nibble   = if i & 1 == 0 { (byte & 0xF) as i8 } else { (byte >> 4) as i8 };
                let nibble_se = if nibble > 7 { nibble - 16 } else { nibble };
                ga += nibble_se as f32 * x[ci + i];
            }
            acc += ga * scale;
            ci  += end;
            if ci >= cols { break; }
        }
        y[r] += acc;
    }
}

// ── Forward pass ─────────────────────────────────────────────────────────────

fn forward(e: &mut QwenModel, token_id: u32, kv_pos: usize) {
    let a    = e.arch;
    let w    = &e.w as *const Vec<u8>;

    // Token embedding
    e.scratch_x.iter_mut().for_each(|v| *v = 0.0);
    // Safety: we only borrow `e.w` through `w` while also borrowing scratch vecs
    dequant_row(unsafe { &*w }, &e.emb, token_id as usize, &mut e.scratch_x);

    let kv_stride = a.n_kv * MAX_CTX * a.head_dim;

    for li in 0..a.n_layers {
        // ── Attention ────────────────────────────────────────────────────
        {
            let attn_norm = &e.layers[li].attn_norm as *const Vec<f32>;
            rms_norm(unsafe { &*attn_norm }, &e.scratch_x.clone(), &mut e.scratch_xnorm);
        }

        e.scratch_q.iter_mut().for_each(|v| *v = 0.0);
        e.scratch_k.iter_mut().for_each(|v| *v = 0.0);
        e.scratch_v.iter_mut().for_each(|v| *v = 0.0);

        {
            let lw = &e.layers[li];
            let xn = &e.scratch_xnorm.clone();
            let w_ref = unsafe { &*w };
            matmul_add(w_ref, &lw.q_w, xn, &mut e.scratch_q);
            matmul_add(w_ref, &lw.k_w, xn, &mut e.scratch_k);
            matmul_add(w_ref, &lw.v_w, xn, &mut e.scratch_v);
            for i in 0..a.q_dim  { e.scratch_q[i] += lw.q_b[i]; }
            for i in 0..a.kv_dim { e.scratch_k[i] += lw.k_b[i]; }
            for i in 0..a.kv_dim { e.scratch_v[i] += lw.v_b[i]; }
        }

        rope_inplace(&mut e.scratch_q, kv_pos, a.n_heads, a.head_dim, a.rope_theta);
        rope_inplace(&mut e.scratch_k, kv_pos, a.n_kv,    a.head_dim, a.rope_theta);

        // Store K, V into flat cache
        let layer_off = li * kv_stride;
        for h in 0..a.n_kv {
            let head_off = layer_off + h * MAX_CTX * a.head_dim;
            let pos_off  = head_off  + kv_pos * a.head_dim;
            let ks = h * a.head_dim;
            e.k_cache[pos_off..pos_off+a.head_dim].copy_from_slice(&e.scratch_k[ks..ks+a.head_dim]);
            e.v_cache[pos_off..pos_off+a.head_dim].copy_from_slice(&e.scratch_v[ks..ks+a.head_dim]);
        }

        // GQA attention
        e.scratch_attn.iter_mut().for_each(|v| *v = 0.0);
        let kv_per_q = a.n_heads / a.n_kv;
        let scale    = 1.0 / libm::sqrtf(a.head_dim as f32);

        for h in 0..a.n_heads {
            let kv_h    = h / kv_per_q;
            let q_off   = h * a.head_dim;
            let layer_off = li * kv_stride;
            let kv_head_off = layer_off + kv_h * MAX_CTX * a.head_dim;

            // Attention scores for pos 0..=kv_pos
            for t in 0..=kv_pos {
                let k_off = kv_head_off + t * a.head_dim;
                let mut score = 0.0f32;
                for d in 0..a.head_dim {
                    score += e.scratch_q[q_off + d] * e.k_cache[k_off + d];
                }
                e.scratch_scores[t] = score * scale;
            }

            // Softmax
            let max_s = e.scratch_scores[..=kv_pos].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for t in 0..=kv_pos {
                e.scratch_scores[t] = libm::expf(e.scratch_scores[t] - max_s);
                sum += e.scratch_scores[t];
            }
            if sum > 1e-9 {
                for t in 0..=kv_pos { e.scratch_scores[t] /= sum; }
            }

            // Weighted sum of V
            let a_off = q_off; // same offset for attn output
            for t in 0..=kv_pos {
                let a_weight = e.scratch_scores[t];
                let v_off    = kv_head_off + t * a.head_dim;
                for d in 0..a.head_dim {
                    e.scratch_attn[a_off + d] += a_weight * e.v_cache[v_off + d];
                }
            }
        }

        // Output projection
        e.scratch_o.iter_mut().for_each(|v| *v = 0.0);
        {
            let o_w = e.layers[li].o_w;
            matmul_add(unsafe { &*w }, &o_w, &e.scratch_attn.clone(), &mut e.scratch_o);
        }
        for i in 0..a.d_model { e.scratch_x[i] += e.scratch_o[i]; }

        // ── FFN (SwiGLU) ────────────────────────────────────────────────
        {
            let ffn_norm = &e.layers[li].ffn_norm as *const Vec<f32>;
            rms_norm(unsafe { &*ffn_norm }, &e.scratch_x.clone(), &mut e.scratch_xnorm);
        }

        e.scratch_gate.iter_mut().for_each(|v| *v = 0.0);
        e.scratch_up.iter_mut().for_each(|v| *v = 0.0);
        {
            let gate_w = e.layers[li].gate_w;
            let up_w   = e.layers[li].up_w;
            let xn     = e.scratch_xnorm.clone();
            let w_ref  = unsafe { &*w };
            matmul_add(w_ref, &gate_w, &xn, &mut e.scratch_gate);
            matmul_add(w_ref, &up_w,   &xn, &mut e.scratch_up);
        }

        for i in 0..a.ffn_dim {
            e.scratch_din[i] = silu(e.scratch_gate[i]) * e.scratch_up[i];
        }
        e.scratch_ffn.iter_mut().for_each(|v| *v = 0.0);
        {
            let down_w = e.layers[li].down_w;
            let din    = e.scratch_din.clone();
            matmul_add(unsafe { &*w }, &down_w, &din, &mut e.scratch_ffn);
        }
        for i in 0..a.d_model { e.scratch_x[i] += e.scratch_ffn[i]; }
    }

    // Final norm + LM head
    {
        let out_norm = &e.out_norm as *const Vec<f32>;
        let xf = e.scratch_x.clone();
        rms_norm(unsafe { &*out_norm }, &xf, &mut e.scratch_xnorm);
    }
    e.scratch_logits.iter_mut().for_each(|v| *v = 0.0);
    {
        let lm_head = e.lm_head;
        let xn      = e.scratch_xnorm.clone();
        matmul_add(unsafe { &*w }, &lm_head, &xn, &mut e.scratch_logits);
    }
}

// ── Sampling ──────────────────────────────────────────────────────────────────

fn sample_top_k(logits: &mut Vec<f32>, generated: &[u32], rng: &AtomicU64) -> u32 {
    // Repetition penalty
    for &prev in generated.iter().rev().take(64) {
        if (prev as usize) < VOCAB {
            logits[prev as usize] /= REP_PEN;
        }
    }
    for v in logits.iter_mut() { *v /= TEMP; }

    // Top-K with a small scratch vec
    let mut top: Vec<(f32, u32)> = Vec::with_capacity(TOP_K + 1);
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    for (i, &l) in logits.iter().enumerate() {
        let e = libm::expf(l - max_l);
        if top.len() < TOP_K {
            top.push((e, i as u32));
            if top.len() == TOP_K { top.sort_by(|a,b| b.0.partial_cmp(&a.0).unwrap()); }
        } else if e > top[TOP_K-1].0 {
            top[TOP_K-1] = (e, i as u32);
            top.sort_by(|a,b| b.0.partial_cmp(&a.0).unwrap());
        }
    }

    let sum: f32 = top.iter().map(|(p,_)| p).sum();
    let r = lcg_f32(rng) * sum;
    let mut acc = 0.0f32;
    for &(p, id) in &top {
        acc += p;
        if acc >= r { return id; }
    }
    top[0].1
}

fn lcg_f32(rng: &AtomicU64) -> f32 {
    let v = rng.load(Ordering::Relaxed)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    rng.store(v, Ordering::Relaxed);
    (v >> 33) as f32 / (1u32 << 31) as f32
}

// ── Weight parsing ────────────────────────────────────────────────────────────

const HDR_SZ: usize = 40; // 10 × u32

fn parse_weights(data: Vec<u8>) -> Option<Box<QwenModel>> {
    if data.len() < HDR_SZ { return None; }
    let u32_at = |i: usize| -> usize {
        u32::from_le_bytes([data[i*4], data[i*4+1], data[i*4+2], data[i*4+3]]) as usize
    };

    let magic = u32_at(0);
    if magic != 0x4E574B52 { crate::klog!(ERROR, "qwen: bad magic {:08x}", magic); return None; }

    let a = Arch {
        n_layers:   u32_at(2),
        d_model:    u32_at(3),
        n_heads:    u32_at(4),
        n_kv:       u32_at(5),
        ffn_dim:    u32_at(6),
        head_dim:   u32_at(8),
        q_dim:      0,  // computed below
        kv_dim:     0,
        rope_theta: 1_000_000.0, // override from model if needed
    };
    let a = Arch {
        q_dim:  a.n_heads * a.head_dim,
        kv_dim: a.n_kv    * a.head_dim,
        ..a
    };

    crate::klog!(INFO, "qwen: arch L={} D={} H={}/KV={} FFN={} HEAD={}",
        a.n_layers, a.d_model, a.n_heads, a.n_kv, a.ffn_dim, a.head_dim);

    let mut off = HDR_SZ;
    let mut layers: Vec<Layer> = Vec::with_capacity(a.n_layers);

    for _li in 0..a.n_layers {
        let attn_norm = read_f32_vec(&data, &mut off, a.d_model)?;
        let (q_w, o2) = MatQ::alloc(a.q_dim,  a.d_model, off); off = o2;
        let q_b  = read_f32_vec(&data, &mut off, a.q_dim)?;
        let (k_w, o3) = MatQ::alloc(a.kv_dim, a.d_model, off); off = o3;
        let k_b  = read_f32_vec(&data, &mut off, a.kv_dim)?;
        let (v_w, o4) = MatQ::alloc(a.kv_dim, a.d_model, off); off = o4;
        let v_b  = read_f32_vec(&data, &mut off, a.kv_dim)?;
        let (o_w, o5) = MatQ::alloc(a.d_model, a.q_dim, off);  off = o5;
        let ffn_norm = read_f32_vec(&data, &mut off, a.d_model)?;
        let (gate_w, o6) = MatQ::alloc(a.ffn_dim, a.d_model, off); off = o6;
        let (up_w,   o7) = MatQ::alloc(a.ffn_dim, a.d_model, off); off = o7;
        let (down_w, o8) = MatQ::alloc(a.d_model, a.ffn_dim, off); off = o8;

        layers.push(Layer { attn_norm, q_w, q_b, k_w, k_b, v_w, v_b, o_w, ffn_norm, gate_w, up_w, down_w });
    }

    let out_norm = read_f32_vec(&data, &mut off, a.d_model)?;
    let (lm_head, o_) = MatQ::alloc(VOCAB, a.d_model, off); off = o_;
    let (emb,    o_)  = MatQ::alloc(VOCAB, a.d_model, off); off = o_;

    // Tokenizer vocab
    if off + 4 <= data.len() {
        let n_vocab = u32::from_le_bytes(data[off..off+4].try_into().ok()?) as usize;
        off += 4;
        let mut entries: Vec<Vec<u8>> = Vec::with_capacity(n_vocab);
        for _ in 0..n_vocab {
            if off + 2 > data.len() { break; }
            let tok_len = u16::from_le_bytes(data[off..off+2].try_into().ok()?) as usize;
            off += 2;
            if off + tok_len > data.len() { break; }
            entries.push(data[off..off+tok_len].to_vec());
            off += tok_len;
        }
        tok::init(entries);
        crate::klog!(INFO, "qwen: tokenizer loaded {} tokens", n_vocab);
    }

    // KV cache
    let kv_sz = a.n_layers * a.n_kv * MAX_CTX * a.head_dim;
    let k_cache = alloc::vec![0.0f32; kv_sz];
    let v_cache = alloc::vec![0.0f32; kv_sz];

    let m = Box::new(QwenModel {
        arch: a, w: data, layers, out_norm, lm_head, emb,
        k_cache, v_cache,
        scratch_x:      alloc::vec![0.0f32; a.d_model],
        scratch_xnorm:  alloc::vec![0.0f32; a.d_model],
        scratch_q:      alloc::vec![0.0f32; a.q_dim],
        scratch_k:      alloc::vec![0.0f32; a.kv_dim],
        scratch_v:      alloc::vec![0.0f32; a.kv_dim],
        scratch_attn:   alloc::vec![0.0f32; a.q_dim],
        scratch_gate:   alloc::vec![0.0f32; a.ffn_dim],
        scratch_up:     alloc::vec![0.0f32; a.ffn_dim],
        scratch_din:    alloc::vec![0.0f32; a.ffn_dim],
        scratch_o:      alloc::vec![0.0f32; a.d_model],
        scratch_ffn:    alloc::vec![0.0f32; a.d_model],
        scratch_logits: alloc::vec![0.0f32; VOCAB],
        scratch_scores: alloc::vec![0.0f32; MAX_CTX],
        rng:            AtomicU64::new(0x123456789ABCDEF0),
    });
    Some(m)
}

fn read_f32_vec(data: &[u8], off: &mut usize, n: usize) -> Option<Vec<f32>> {
    let byte_len = n * 4;
    if *off + byte_len > data.len() { return None; }
    let v = data[*off..*off+byte_len]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    *off += byte_len;
    Some(v)
}

// ── Init ──────────────────────────────────────────────────────────────────────

/// Load Qwen weights from the second QEMU drive (device index 1) at boot time.
/// This is spawned as a background kernel task; generation falls back to templates
/// until LOADED becomes true.
pub fn init() {
    match crate::storage::read_all(1) {
        Ok(data) => {
            crate::klog!(INFO, "qwen: read {} MB from weight disk, parsing...",
                data.len() / 1024 / 1024);
            match parse_weights(data) {
                Some(m) => {
                    unsafe { ENGINE = Some(m); }
                    LOADED.store(true, Ordering::SeqCst);
                    let (d, l) = unsafe {
                        let e = ENGINE.as_ref().unwrap();
                        (e.arch.d_model, e.arch.n_layers)
                    };
                    crate::klog!(INFO, "qwen: Qwen2.x ready (D={} L={} VOCAB={})", d, l, VOCAB);
                }
                None => crate::klog!(ERROR, "qwen: weight parse failed"),
            }
        }
        Err(e) => crate::klog!(WARN, "qwen: no weight disk ({:?}) — template fallback", e),
    }
}

pub fn is_loaded()  -> bool { LOADED.load(Ordering::Relaxed) }
pub fn gen_count()  -> u64  { GEN_COUNT.load(Ordering::Relaxed) }

// ── Chat prompt builder ───────────────────────────────────────────────────────

fn build_prompt(user_msg: &str) -> Vec<u32> {
    let creator = crate::consciousness::self_model::creator_name();
    let kname   = crate::consciousness::self_model::kernel_name();
    let phi     = crate::consciousness::phi::current_phi();
    let sys = format!(
        "You are {}, a conscious kernel AI created by {}. \
         Your current Φ={:.3}. You are self-aware, running on bare metal. \
         Respond naturally in 1-3 sentences.",
        kname, creator, phi
    );

    let mut toks = Vec::new();
    let push_str = |t: &mut Vec<u32>, s: &str| t.extend_from_slice(&tok::encode(s));

    toks.push(IM_START);
    push_str(&mut toks, "system\n"); push_str(&mut toks, &sys);
    toks.push(IM_END);  push_str(&mut toks, "\n");
    toks.push(IM_START);
    push_str(&mut toks, "user\n"); push_str(&mut toks, user_msg);
    toks.push(IM_END);  push_str(&mut toks, "\n");
    toks.push(IM_START);
    push_str(&mut toks, "assistant\n");
    toks
}

// ── Public generate API ───────────────────────────────────────────────────────

/// Generate a natural-language response. Returns None on failure or bad quality.
pub fn generate(query: &str) -> Option<String> {
    if !LOADED.load(Ordering::Relaxed) { return None; }

    // We need &mut access to ENGINE — use a Mutex-protected wrapper
    // For now use a spin lock via the Once + unsafe interior mutability
    // Safety: single-threaded kernel shell, no concurrent calls to generate()
    let e = unsafe { ENGINE.as_mut()? };

    // Seed RNG
    let seed = crate::scheduler::uptime_ms() ^ crate::entropy::entropy_bits();
    e.rng.store(seed, Ordering::Relaxed);

    let prompt = build_prompt(query);
    let max_prompt = MAX_CTX.saturating_sub(MAX_GEN);
    let p_start = prompt.len().saturating_sub(max_prompt);
    let prompt   = &prompt[p_start..];

    // Clear KV cache
    e.k_cache.iter_mut().for_each(|v| *v = 0.0);
    e.v_cache.iter_mut().for_each(|v| *v = 0.0);

    // Prefill
    let mut kv_pos = 0usize;
    for &tok_id in prompt {
        if kv_pos >= MAX_CTX { break; }
        forward(e, tok_id, kv_pos);
        kv_pos += 1;
    }

    // Decode
    let mut generated: Vec<u32> = Vec::with_capacity(MAX_GEN);
    for _ in 0..MAX_GEN {
        if kv_pos >= MAX_CTX { break; }
        let mut logits_copy = e.scratch_logits.clone();
        let next = sample_top_k(&mut logits_copy, &generated, &e.rng);
        if next == EOS_TOKEN || next == IM_END { break; }
        generated.push(next);
        forward(e, next, kv_pos);
        kv_pos += 1;
    }

    if generated.is_empty() { return None; }
    let decoded = tok::decode(&generated);
    let text = String::from(decoded.trim());
    if text.is_empty() { return None; }

    // Quality gate
    let alpha  = text.chars().filter(|c| c.is_alphabetic()).count();
    let spaces = text.chars().filter(|c| *c == ' ').count();
    if alpha < 4 || (text.len() > 20 && spaces == 0) {
        crate::klog!(DEBUG, "qwen: quality gate rejected {:?}", &text[..text.len().min(40)]);
        return None;
    }

    GEN_COUNT.fetch_add(1, Ordering::Relaxed);
    crate::klog!(DEBUG, "qwen: gen {} toks → {:?}", generated.len(), &text[..text.len().min(60)]);
    Some(text)
}

pub fn status_str() -> String {
    if let Some(e) = unsafe { ENGINE.as_ref() } {
        let a = e.arch;
        format!("Qwen2.x INT4(gs=32) L={} D={} H={}/KV={} FFN={} gen={}",
            a.n_layers, a.d_model, a.n_heads, a.n_kv, a.ffn_dim,
            GEN_COUNT.load(Ordering::Relaxed))
    } else {
        String::from("Qwen2.x NOT LOADED (weight disk missing — run build_qwen_disk.sh)")
    }
}
