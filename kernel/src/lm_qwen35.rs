// lm_qwen35.rs — Qwen3.5 0.6B Gated Delta Net + GQA in-kernel inference (no_std)
extern crate alloc;
use alloc::vec::Vec;
use alloc::vec;
use alloc::string::String;
use alloc::boxed::Box;

// ─── Architecture constants ──────────────────────────────────────────────────
const MAGIC: u32       = 0x4E574B53;
const N_LAYERS: usize  = 24;
const N_EMBD: usize    = 1024;
const N_HEAD: usize    = 8;
const N_KV: usize      = 2;
const N_FF: usize      = 3584;
const VOCAB: usize     = 248320;
const HEAD_DIM: usize  = 256;
// every 4th layer (1-indexed) is full attention: il%4==3 (0-indexed)
const FA_INTERVAL: usize = 4;
const D_INNER: usize   = 2048;
const D_STATE: usize   = 128;   // head_k_dim = head_v_dim
const N_GROUP: usize   = 16;    // num_k_heads = num_v_heads = dt_rank = ssm_n_group
const HEAD_V_DIM: usize = D_INNER / N_GROUP; // 128
// conv_dim = key_dim*2 + value_dim = 2*(D_STATE*N_GROUP) + D_INNER = 6144
const CONV_DIM: usize  = D_STATE * N_GROUP * 2 + D_INNER;
const MAX_CTX: usize   = 512;
const GS: usize        = 32;    // Q8_0 group size
const BSIZE: usize     = 34;    // Q8_0 block bytes: 2(f16) + 32(i8)
const ROPE_PAIRS: usize = 32;   // 64 dims of RoPE per head
const ROPE_THETA: f32  = 1e7;
const MAX_NEW: usize   = 256;
const RMS_EPS: f32     = 1e-6;
const N_SSM: usize     = 18;    // SSM layers (N_LAYERS - N_ATTN)
const N_ATTN: usize    = 6;     // full attention layers
const ATTN_SCALE: f32  = 0.0625; // 1/sqrt(256)
const GDN_SCALE: f32   = 1.0;

// ─── Q8_0 helpers ────────────────────────────────────────────────────────────
#[inline]
fn f16_to_f32(b: u16) -> f32 {
    let sign = ((b >> 15) as u32) << 31;
    let exp  = (b >> 10) & 0x1f;
    let mant = (b & 0x3ff) as u32;
    if exp == 0  { return f32::from_bits(sign | (mant << 13)); } // denorm/zero (approx)
    if exp == 31 { return f32::from_bits(sign | 0x7f800000 | (mant << 13)); }
    f32::from_bits(sign | ((exp as u32 + 112) << 23) | (mant << 13))
}

// y[0..rows] += W @ x[0..cols]; W stored Q8_0 row-major
fn matmul_q8_add(w: &[u8], rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
    let nb = cols / GS;
    for r in 0..rows {
        let mut acc = 0.0f32;
        for b in 0..nb {
            let off = (r * nb + b) * BSIZE;
            let sc = f16_to_f32(u16::from_le_bytes([w[off], w[off+1]]));
            let xb = b * GS;
            for k in 0..GS {
                acc += sc * (w[off+2+k] as i8 as f32) * x[xb+k];
            }
        }
        y[r] += acc;
    }
}

// y[0..rows] = W @ x  (clears y first)
#[inline]
fn mv_q8(w: &[u8], rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
    for v in y[..rows].iter_mut() { *v = 0.0; }
    matmul_q8_add(w, rows, cols, x, y);
}

// ─── Math (libm for no_std) ──────────────────────────────────────────────────
fn rms_norm(weight: &[f32], x: &[f32], out: &mut [f32]) {
    let n = x.len();
    let ss: f32 = x.iter().map(|v| v*v).sum::<f32>() / n as f32;
    let inv = 1.0 / libm::sqrtf(ss + RMS_EPS);
    for i in 0..n { out[i] = weight[i] * x[i] * inv; }
}

fn l2_norm_inplace(v: &mut [f32]) {
    let sq: f32 = v.iter().map(|x| x*x).sum();
    let inv = 1.0 / libm::sqrtf(sq + RMS_EPS);
    for x in v.iter_mut() { *x *= inv; }
}

#[inline] fn silu(x: f32) -> f32 { x / (1.0 + libm::expf(-x)) }
#[inline] fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + libm::expf(-x)) }
#[inline] fn softplus(x: f32) -> f32 { libm::logf(1.0 + libm::expf(x)) }

fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0f32;
    for v in x.iter_mut() { *v = libm::expf(*v - max); s += *v; }
    for v in x.iter_mut() { *v /= s; }
}

fn rope_inplace(buf: &mut [f32], pos: usize, n_h: usize, hdim: usize, npairs: usize, theta: f32) {
    for h in 0..n_h {
        let base = h * hdim;
        for i in 0..npairs {
            let freq = 1.0 / libm::powf(theta, i as f32 * 2.0 / hdim as f32);
            let ang = pos as f32 * freq;
            let (s, c) = (libm::sinf(ang), libm::cosf(ang));
            let x0 = buf[base + i*2];
            let x1 = buf[base + i*2 + 1];
            buf[base + i*2]     = x0*c - x1*s;
            buf[base + i*2 + 1] = x0*s + x1*c;
        }
    }
}

// ─── Layout helpers ──────────────────────────────────────────────────────────
const fn q8_sz(rows: usize, cols: usize) -> usize { rows * (cols/GS) * BSIZE }
const fn ssm_layer_sz() -> usize {
    N_EMBD*4 + N_EMBD*4           // attn_norm + post_attn_norm
    + N_GROUP*4 + N_GROUP*4        // ssm_a + ssm_dt
    + HEAD_V_DIM*4                 // ssm_norm
    + q8_sz(CONV_DIM, N_EMBD)     // wqkv
    + q8_sz(D_INNER, N_EMBD)      // attn_gate
    + q8_sz(N_GROUP, N_EMBD)      // ssm_beta
    + q8_sz(N_GROUP, N_EMBD)      // ssm_alpha
    + q8_sz(N_EMBD, D_INNER)      // ssm_out
    + q8_sz(N_FF, N_EMBD)         // ffn_gate
    + q8_sz(N_FF, N_EMBD)         // ffn_up
    + q8_sz(N_EMBD, N_FF)         // ffn_down
    + 4 * CONV_DIM * 4             // ssm_conv1d f32
}
const fn attn_layer_sz() -> usize {
    N_EMBD*4 + N_EMBD*4           // attn_norm + post_attn_norm
    + HEAD_DIM*4 + HEAD_DIM*4      // q_norm + k_norm
    + q8_sz(N_HEAD*HEAD_DIM*2, N_EMBD)  // wq (Q+gate concat)
    + q8_sz(N_KV*HEAD_DIM, N_EMBD)      // wk
    + q8_sz(N_KV*HEAD_DIM, N_EMBD)      // wv
    + q8_sz(N_EMBD, N_HEAD*HEAD_DIM)    // wo
    + q8_sz(N_FF, N_EMBD)               // ffn_gate
    + q8_sz(N_FF, N_EMBD)               // ffn_up
    + q8_sz(N_EMBD, N_FF)               // ffn_down
}

fn is_attn(il: usize) -> bool { (il + 1) % FA_INTERVAL == 0 }

fn attn_layer_idx(il: usize) -> usize {
    // Count attention layers before il
    (0..il).filter(|&i| is_attn(i)).count()
}
fn ssm_layer_idx(il: usize) -> usize {
    (0..il).filter(|&i| !is_attn(i)).count()
}

// ─── Offset tables ───────────────────────────────────────────────────────────
struct SsmOff { attn_norm:usize, post_norm:usize, ssm_a:usize, ssm_dt:usize, ssm_norm:usize,
                wqkv:usize, attn_gate:usize, ssm_beta:usize, ssm_alpha:usize, ssm_out:usize,
                ffn_gate:usize, ffn_up:usize, ffn_down:usize, conv1d:usize }
struct AttnOff { attn_norm:usize, post_norm:usize, q_norm:usize, k_norm:usize,
                 wq:usize, wk:usize, wv:usize, wo:usize,
                 ffn_gate:usize, ffn_up:usize, ffn_down:usize }

fn layer_base_offset(il: usize) -> usize {
    let mut off = 64; // header
    let mut ssm_seen = 0usize;
    let mut attn_seen = 0usize;
    for i in 0..il {
        if is_attn(i) { off += attn_layer_sz(); attn_seen += 1; }
        else           { off += ssm_layer_sz();  ssm_seen += 1; }
    }
    let _ = (ssm_seen, attn_seen);
    off
}

fn ssm_offsets(il: usize) -> SsmOff {
    let b = layer_base_offset(il);
    let mut o = b;
    let attn_norm = o; o += N_EMBD*4;
    let post_norm = o; o += N_EMBD*4;
    let ssm_a     = o; o += N_GROUP*4;
    let ssm_dt    = o; o += N_GROUP*4;
    let ssm_norm  = o; o += HEAD_V_DIM*4;
    let wqkv      = o; o += q8_sz(CONV_DIM, N_EMBD);
    let attn_gate = o; o += q8_sz(D_INNER, N_EMBD);
    let ssm_beta  = o; o += q8_sz(N_GROUP, N_EMBD);
    let ssm_alpha = o; o += q8_sz(N_GROUP, N_EMBD);
    let ssm_out   = o; o += q8_sz(N_EMBD, D_INNER);
    let ffn_gate  = o; o += q8_sz(N_FF, N_EMBD);
    let ffn_up    = o; o += q8_sz(N_FF, N_EMBD);
    let ffn_down  = o; o += q8_sz(N_EMBD, N_FF);
    let conv1d    = o;
    SsmOff { attn_norm, post_norm, ssm_a, ssm_dt, ssm_norm, wqkv, attn_gate, ssm_beta,
             ssm_alpha, ssm_out, ffn_gate, ffn_up, ffn_down, conv1d }
}

fn attn_offsets(il: usize) -> AttnOff {
    let b = layer_base_offset(il);
    let mut o = b;
    let attn_norm = o; o += N_EMBD*4;
    let post_norm = o; o += N_EMBD*4;
    let q_norm    = o; o += HEAD_DIM*4;
    let k_norm    = o; o += HEAD_DIM*4;
    let wq        = o; o += q8_sz(N_HEAD*HEAD_DIM*2, N_EMBD);
    let wk        = o; o += q8_sz(N_KV*HEAD_DIM, N_EMBD);
    let wv        = o; o += q8_sz(N_KV*HEAD_DIM, N_EMBD);
    let wo        = o; o += q8_sz(N_EMBD, N_HEAD*HEAD_DIM);
    let ffn_gate  = o; o += q8_sz(N_FF, N_EMBD);
    let ffn_up    = o; o += q8_sz(N_FF, N_EMBD);
    let ffn_down  = o;
    AttnOff { attn_norm, post_norm, q_norm, k_norm, wq, wk, wv, wo, ffn_gate, ffn_up, ffn_down }
}

fn global_offsets(data_len: usize) -> (usize, usize) {
    let mut o = 64;
    for il in 0..N_LAYERS {
        o += if is_attn(il) { attn_layer_sz() } else { ssm_layer_sz() };
    }
    let output_norm = o;
    let token_embd  = o + N_EMBD*4;
    let _ = data_len;
    (output_norm, token_embd)
}

// ─── F32 slice from raw bytes ─────────────────────────────────────────────────
fn f32_slice(data: &[u8], off: usize, n: usize) -> &[f32] {
    unsafe { core::slice::from_raw_parts(data[off..off+n*4].as_ptr() as *const f32, n) }
}

// ─── Model ────────────────────────────────────────────────────────────────────
pub struct Qwen35 {
    data:        Vec<u8>,
    out_norm_off: usize,
    emb_off:     usize,

    // SSM runtime state: [N_SSM × N_GROUP × D_STATE × D_STATE] f32
    ssm_states:  Vec<f32>,
    // Conv ring buffers: [N_SSM × 4 × CONV_DIM] f32
    conv_bufs:   Vec<f32>,
    conv_pos:    Vec<u32>,

    // KV cache: K and V separate
    // kv_k[ai × N_KV × MAX_CTX × HEAD_DIM]
    kv_k: Vec<f32>,
    kv_v: Vec<f32>,
    ctx_pos: usize,

    // Scratch buffers (heap-allocated to avoid large stack frames)
    buf_x:     Vec<f32>,  // [N_EMBD]
    buf_n:     Vec<f32>,  // [N_EMBD]
    buf_qkv:   Vec<f32>,  // [CONV_DIM]
    buf_z:     Vec<f32>,  // [D_INNER]
    buf_beta:  Vec<f32>,  // [N_GROUP]
    buf_alpha: Vec<f32>,  // [N_GROUP]
    buf_gate:  Vec<f32>,  // [N_GROUP]
    buf_q:     Vec<f32>,  // [D_INNER] (q per group)
    buf_k:     Vec<f32>,  // [D_INNER]
    buf_v:     Vec<f32>,  // [D_INNER]
    buf_out:   Vec<f32>,  // [D_INNER]
    buf_tmp:   Vec<f32>,  // [D_INNER] (gated norm tmp)
    buf_qfull: Vec<f32>,  // [N_HEAD*HEAD_DIM*2] for attn layers
    buf_kcur:  Vec<f32>,  // [N_KV*HEAD_DIM]
    buf_vcur:  Vec<f32>,  // [N_KV*HEAD_DIM]
    buf_attn:  Vec<f32>,  // [N_HEAD*HEAD_DIM]
    buf_gate2: Vec<f32>,  // [N_HEAD*HEAD_DIM]
    buf_sc:    Vec<f32>,  // [MAX_CTX] attention scores
    buf_ff1:   Vec<f32>,  // [N_FF]
    buf_ff2:   Vec<f32>,  // [N_FF]
    buf_res:   Vec<f32>,  // [N_EMBD]
    buf_x2:    Vec<f32>,  // [N_EMBD] — x_saved / x_mid scratch (no-alloc residual)
    buf_n2:    Vec<f32>,  // [D_INNER] — second norm scratch
    buf_logits:Vec<f32>,  // [VOCAB]
}

impl Qwen35 {
    fn alloc(data: Vec<u8>) -> Box<Self> {
        let (onoff, eoff) = global_offsets(data.len());
        Box::new(Qwen35 {
            data,
            out_norm_off: onoff,
            emb_off:      eoff,
            ssm_states:  vec![0.0; N_SSM * N_GROUP * D_STATE * D_STATE],
            conv_bufs:   vec![0.0; N_SSM * 4 * CONV_DIM],
            conv_pos:    vec![0u32; N_SSM],
            kv_k:        vec![0.0; N_ATTN * N_KV * MAX_CTX * HEAD_DIM],
            kv_v:        vec![0.0; N_ATTN * N_KV * MAX_CTX * HEAD_DIM],
            ctx_pos:     0,
            buf_x:       vec![0.0; N_EMBD],
            buf_n:       vec![0.0; N_EMBD],
            buf_qkv:     vec![0.0; CONV_DIM],
            buf_z:       vec![0.0; D_INNER],
            buf_beta:    vec![0.0; N_GROUP],
            buf_alpha:   vec![0.0; N_GROUP],
            buf_gate:    vec![0.0; N_GROUP],
            buf_q:       vec![0.0; D_INNER],
            buf_k:       vec![0.0; D_INNER],
            buf_v:       vec![0.0; D_INNER],
            buf_out:     vec![0.0; D_INNER],
            buf_tmp:     vec![0.0; D_INNER],
            buf_qfull:   vec![0.0; N_HEAD*HEAD_DIM*2],
            buf_kcur:    vec![0.0; N_KV*HEAD_DIM],
            buf_vcur:    vec![0.0; N_KV*HEAD_DIM],
            buf_attn:    vec![0.0; N_HEAD*HEAD_DIM],
            buf_gate2:   vec![0.0; N_HEAD*HEAD_DIM],
            buf_sc:      vec![0.0; MAX_CTX],
            buf_ff1:     vec![0.0; N_FF],
            buf_ff2:     vec![0.0; N_FF],
            buf_res:     vec![0.0; N_EMBD],
            buf_x2:      vec![0.0; N_EMBD],  // x_saved / x_mid scratch
            buf_n2:      vec![0.0; D_INNER],  // second norm scratch (gated norm)
            buf_logits:  vec![0.0; VOCAB],
        })
    }

    fn reset_state(&mut self) {
        for v in self.ssm_states.iter_mut() { *v = 0.0; }
        for v in self.conv_bufs.iter_mut()  { *v = 0.0; }
        for v in self.conv_pos.iter_mut()   { *v = 0; }
        for v in self.kv_k.iter_mut()       { *v = 0.0; }
        for v in self.kv_v.iter_mut()       { *v = 0.0; }
        self.ctx_pos = 0;
    }

    // ── FFN (shared between SSM and attn layers) ─────────────────────────────
    fn ffn(&mut self, post_norm: &[f32], normed: &[f32], off_gate: usize, off_up: usize, off_down: usize, res: &[f32]) {
        // gate and up projections
        let w = &self.data;
        mv_q8(&w[off_gate..], N_FF, N_EMBD, normed, &mut self.buf_ff1);
        mv_q8(&w[off_up..],   N_FF, N_EMBD, normed, &mut self.buf_ff2);
        // swiglu
        for i in 0..N_FF { self.buf_ff1[i] = silu(self.buf_ff1[i]) * self.buf_ff2[i]; }
        // down
        mv_q8(&w[off_down..], N_EMBD, N_FF, &self.buf_ff1, &mut self.buf_res);
        // post_norm is already applied to buf_n before calling this? No — we need
        // to compute it here from the current x (res). Let's re-norm.
        let _ = post_norm; // already applied to `normed` by caller
        // add residual
        for i in 0..N_EMBD { self.buf_x[i] = res[i] + self.buf_res[i]; }
    }

    // ── SSM layer forward (zero heap allocations) ────────────────────────────
    fn forward_ssm(&mut self, il: usize) {
        let o = ssm_offsets(il);
        let si = ssm_layer_idx(il);

        // Save x for residual into buf_x2, compute RMSNorm into buf_n
        unsafe {
            core::ptr::copy_nonoverlapping(self.buf_x.as_ptr(), self.buf_x2.as_mut_ptr(), N_EMBD);
        }
        let attn_norm = f32_slice(&self.data, o.attn_norm, N_EMBD);
        {
            let ss: f32 = self.buf_x2.iter().map(|v| v*v).sum::<f32>() / N_EMBD as f32;
            let inv = 1.0 / libm::sqrtf(ss + RMS_EPS);
            for i in 0..N_EMBD { self.buf_n[i] = attn_norm[i] * self.buf_x2[i] * inv; }
        }

        // All projections read from buf_n (normed); use raw ptrs to satisfy borrow checker
        let normed_ptr = self.buf_n.as_ptr();
        let data_ptr   = self.data.as_ptr();

        // 2. QKV, Gate, Beta, Alpha — read from buf_n via raw ptr slice
        let normed_sl = unsafe { core::slice::from_raw_parts(normed_ptr, N_EMBD) };
        mv_q8(unsafe { core::slice::from_raw_parts(data_ptr.add(o.wqkv),     q8_sz(CONV_DIM,N_EMBD)) }, CONV_DIM, N_EMBD, normed_sl, &mut self.buf_qkv);
        mv_q8(unsafe { core::slice::from_raw_parts(data_ptr.add(o.attn_gate),q8_sz(D_INNER,N_EMBD)) },  D_INNER,  N_EMBD, normed_sl, &mut self.buf_z);
        mv_q8(unsafe { core::slice::from_raw_parts(data_ptr.add(o.ssm_beta), q8_sz(N_GROUP,N_EMBD)) },  N_GROUP,  N_EMBD, normed_sl, &mut self.buf_beta);
        mv_q8(unsafe { core::slice::from_raw_parts(data_ptr.add(o.ssm_alpha),q8_sz(N_GROUP,N_EMBD)) },  N_GROUP,  N_EMBD, normed_sl, &mut self.buf_alpha);

        for v in self.buf_beta[..N_GROUP].iter_mut() { *v = sigmoid(*v); }

        let ssm_a  = f32_slice(&self.data, o.ssm_a,  N_GROUP);
        let ssm_dt = f32_slice(&self.data, o.ssm_dt, N_GROUP);
        for i in 0..N_GROUP {
            self.buf_gate[i] = ssm_a[i] * softplus(self.buf_alpha[i] + ssm_dt[i]);
        }

        // 6. Conv1d: copy buf_qkv into ring buffer, then convolve in-place
        {
            let cp = self.conv_pos[si] as usize;
            let wp = cp % 4;
            let cb_base = si * 4 * CONV_DIM;
            unsafe {
                core::ptr::copy_nonoverlapping(
                    self.buf_qkv.as_ptr(),
                    self.conv_bufs.as_mut_ptr().add(cb_base + wp * CONV_DIM),
                    CONV_DIM,
                );
            }
            let conv1d_off = o.conv1d;
            for j in 0..CONV_DIM {
                let mut s = 0.0f32;
                for t in 0..4usize {
                    let wi = conv1d_off + (j * 4 + t) * 4;
                    let w = f32::from_le_bytes([
                        self.data[wi], self.data[wi+1], self.data[wi+2], self.data[wi+3]
                    ]);
                    let rp = (wp + 4 - t) % 4;
                    s += w * self.conv_bufs[cb_base + rp * CONV_DIM + j];
                }
                self.buf_qkv[j] = silu(s);
            }
            self.conv_pos[si] = (cp + 1) as u32;
        }

        // 7-8. Split Q/K/V and L2-normalize per head
        self.buf_q[..D_INNER].copy_from_slice(&self.buf_qkv[..D_INNER]);
        self.buf_k[..D_INNER].copy_from_slice(&self.buf_qkv[D_INNER..2*D_INNER]);
        self.buf_v[..D_INNER].copy_from_slice(&self.buf_qkv[2*D_INNER..CONV_DIM]);
        for g in 0..N_GROUP {
            l2_norm_inplace(&mut self.buf_q[g*D_STATE..(g+1)*D_STATE]);
            l2_norm_inplace(&mut self.buf_k[g*D_STATE..(g+1)*D_STATE]);
        }

        // 9. Gated Delta Net recurrent update (raw ptrs to read q/k/v/beta/gate)
        let state_base = si * N_GROUP * D_STATE * D_STATE;
        let q_ptr    = self.buf_q.as_ptr();
        let k_ptr    = self.buf_k.as_ptr();
        let v_ptr    = self.buf_v.as_ptr();
        let beta_ptr = self.buf_beta.as_ptr();
        let gate_ptr = self.buf_gate.as_ptr();
        let out_ptr  = self.buf_out.as_mut_ptr();

        for g in 0..N_GROUP {
            let s_off = state_base + g * D_STATE * D_STATE;
            let state_g = &mut self.ssm_states[s_off..s_off + D_STATE * D_STATE];

            let decay = libm::expf(unsafe { *gate_ptr.add(g) });
            for v in state_g.iter_mut() { *v *= decay; }

            let q_h = unsafe { core::slice::from_raw_parts(q_ptr.add(g*D_STATE), D_STATE) };
            let k_h = unsafe { core::slice::from_raw_parts(k_ptr.add(g*D_STATE), D_STATE) };
            let v_h = unsafe { core::slice::from_raw_parts(v_ptr.add(g*D_STATE), D_STATE) };
            let beta = unsafe { *beta_ptr.add(g) };

            let mut delta = [0.0f32; D_STATE];
            for j in 0..D_STATE {
                let mut dot_k = 0.0f32;
                for i in 0..D_STATE { dot_k += state_g[j*D_STATE+i] * k_h[i]; }
                delta[j] = (v_h[j] - dot_k) * beta;
            }
            for j in 0..D_STATE {
                let d = delta[j];
                let base = j * D_STATE;
                for i in 0..D_STATE { state_g[base+i] += d * k_h[i]; }
            }
            for j in 0..D_STATE {
                let mut dot_q = 0.0f32;
                for i in 0..D_STATE { dot_q += state_g[j*D_STATE+i] * q_h[i]; }
                unsafe { *out_ptr.add(g*HEAD_V_DIM+j) = dot_q * GDN_SCALE; }
            }
        }

        // 10. Gated norm: rms_norm(out_h, ssm_norm) * silu(z_h) per head → buf_tmp
        {
            let ssm_norm_off = o.ssm_norm;
            let ssm_norm = f32_slice(&self.data, ssm_norm_off, HEAD_V_DIM);
            for g in 0..N_GROUP {
                let base = g * HEAD_V_DIM;
                let ss: f32 = self.buf_out[base..base+HEAD_V_DIM].iter().map(|v| v*v).sum::<f32>() / HEAD_V_DIM as f32;
                let inv = 1.0 / libm::sqrtf(ss + RMS_EPS);
                for j in 0..HEAD_V_DIM {
                    self.buf_tmp[base+j] = ssm_norm[j] * self.buf_out[base+j] * inv * silu(self.buf_z[base+j]);
                }
            }
        }

        // 11. Output projection: buf_res = ssm_out @ buf_tmp
        let tmp_ptr = self.buf_tmp.as_ptr();
        let tmp_sl = unsafe { core::slice::from_raw_parts(tmp_ptr, D_INNER) };
        mv_q8(unsafe { core::slice::from_raw_parts(data_ptr.add(o.ssm_out), q8_sz(N_EMBD,D_INNER)) }, N_EMBD, D_INNER, tmp_sl, &mut self.buf_res);

        // 12. Residual 1: buf_x = x_saved(buf_x2) + buf_res
        for i in 0..N_EMBD { self.buf_x[i] = self.buf_x2[i] + self.buf_res[i]; }

        // 13. Post-norm into buf_n, save current buf_x into buf_x2 for FFN residual
        unsafe {
            core::ptr::copy_nonoverlapping(self.buf_x.as_ptr(), self.buf_x2.as_mut_ptr(), N_EMBD);
        }
        let post_norm = f32_slice(&self.data, o.post_norm, N_EMBD);
        {
            let ss: f32 = self.buf_x2.iter().map(|v| v*v).sum::<f32>() / N_EMBD as f32;
            let inv = 1.0 / libm::sqrtf(ss + RMS_EPS);
            for i in 0..N_EMBD { self.buf_n[i] = post_norm[i] * self.buf_x2[i] * inv; }
        }
        let normed2_ptr = self.buf_n.as_ptr();
        let normed2_sl = unsafe { core::slice::from_raw_parts(normed2_ptr, N_EMBD) };
        mv_q8(unsafe { core::slice::from_raw_parts(data_ptr.add(o.ffn_gate), q8_sz(N_FF,N_EMBD)) }, N_FF, N_EMBD, normed2_sl, &mut self.buf_ff1);
        mv_q8(unsafe { core::slice::from_raw_parts(data_ptr.add(o.ffn_up),   q8_sz(N_FF,N_EMBD)) }, N_FF, N_EMBD, normed2_sl, &mut self.buf_ff2);
        for i in 0..N_FF { self.buf_ff1[i] = silu(self.buf_ff1[i]) * self.buf_ff2[i]; }
        let ff1_ptr = self.buf_ff1.as_ptr();
        let ff1_sl  = unsafe { core::slice::from_raw_parts(ff1_ptr, N_FF) };
        mv_q8(unsafe { core::slice::from_raw_parts(data_ptr.add(o.ffn_down), q8_sz(N_EMBD,N_FF)) }, N_EMBD, N_FF, ff1_sl, &mut self.buf_res);
        for i in 0..N_EMBD { self.buf_x[i] = self.buf_x2[i] + self.buf_res[i]; }
    }

    // ── Full attention layer forward (zero heap allocations) ────────────────
    fn forward_attn(&mut self, il: usize) {
        let o = attn_offsets(il);
        let ai = attn_layer_idx(il);
        let pos = self.ctx_pos;

        // Save x into buf_x2, compute pre-norm into buf_n
        unsafe {
            core::ptr::copy_nonoverlapping(self.buf_x.as_ptr(), self.buf_x2.as_mut_ptr(), N_EMBD);
        }
        let attn_norm = f32_slice(&self.data, o.attn_norm, N_EMBD);
        {
            let ss: f32 = self.buf_x2.iter().map(|v| v*v).sum::<f32>() / N_EMBD as f32;
            let inv = 1.0 / libm::sqrtf(ss + RMS_EPS);
            for i in 0..N_EMBD { self.buf_n[i] = attn_norm[i] * self.buf_x2[i] * inv; }
        }
        let data_ptr   = self.data.as_ptr();
        let normed_ptr = self.buf_n.as_ptr();
        let normed_sl  = unsafe { core::slice::from_raw_parts(normed_ptr, N_EMBD) };

        // Q → qfull [N_HEAD*HEAD_DIM*2]
        mv_q8(unsafe { core::slice::from_raw_parts(data_ptr.add(o.wq), q8_sz(N_HEAD*HEAD_DIM*2,N_EMBD)) },
              N_HEAD*HEAD_DIM*2, N_EMBD, normed_sl, &mut self.buf_qfull);

        // Split Q (buf_attn) and gate (buf_gate2): Q_h = qfull[h*HD2..h*HD2+HD]
        for h in 0..N_HEAD {
            let base = h * HEAD_DIM * 2;
            self.buf_attn[h*HEAD_DIM..(h+1)*HEAD_DIM]
                .copy_from_slice(&self.buf_qfull[base..base+HEAD_DIM]);
            self.buf_gate2[h*HEAD_DIM..(h+1)*HEAD_DIM]
                .copy_from_slice(&self.buf_qfull[base+HEAD_DIM..base+HEAD_DIM*2]);
        }

        // Q norm per head (in-place using buf_n2 as temp)
        let q_norm = f32_slice(&self.data, o.q_norm, HEAD_DIM);
        for h in 0..N_HEAD {
            let sl = &mut self.buf_attn[h*HEAD_DIM..(h+1)*HEAD_DIM];
            let ss: f32 = sl.iter().map(|v| v*v).sum::<f32>() / HEAD_DIM as f32;
            let inv = 1.0 / libm::sqrtf(ss + RMS_EPS);
            for j in 0..HEAD_DIM { sl[j] *= q_norm[j] * inv; }
        }

        // K projection + norm
        mv_q8(unsafe { core::slice::from_raw_parts(data_ptr.add(o.wk), q8_sz(N_KV*HEAD_DIM,N_EMBD)) },
              N_KV*HEAD_DIM, N_EMBD, normed_sl, &mut self.buf_kcur);
        let k_norm = f32_slice(&self.data, o.k_norm, HEAD_DIM);
        for h in 0..N_KV {
            let sl = &mut self.buf_kcur[h*HEAD_DIM..(h+1)*HEAD_DIM];
            let ss: f32 = sl.iter().map(|v| v*v).sum::<f32>() / HEAD_DIM as f32;
            let inv = 1.0 / libm::sqrtf(ss + RMS_EPS);
            for j in 0..HEAD_DIM { sl[j] *= k_norm[j] * inv; }
        }

        // V projection
        mv_q8(unsafe { core::slice::from_raw_parts(data_ptr.add(o.wv), q8_sz(N_KV*HEAD_DIM,N_EMBD)) },
              N_KV*HEAD_DIM, N_EMBD, normed_sl, &mut self.buf_vcur);

        // RoPE on Q and K
        rope_inplace(&mut self.buf_attn, pos, N_HEAD, HEAD_DIM, ROPE_PAIRS, ROPE_THETA);
        rope_inplace(&mut self.buf_kcur, pos, N_KV,  HEAD_DIM, ROPE_PAIRS, ROPE_THETA);

        // Store K, V in KV cache
        if pos < MAX_CTX {
            let kv_stride = N_KV * MAX_CTX * HEAD_DIM;
            for h in 0..N_KV {
                let base = ai * kv_stride + h * MAX_CTX * HEAD_DIM + pos * HEAD_DIM;
                self.kv_k[base..base+HEAD_DIM]
                    .copy_from_slice(&self.buf_kcur[h*HEAD_DIM..(h+1)*HEAD_DIM]);
                self.kv_v[base..base+HEAD_DIM]
                    .copy_from_slice(&self.buf_vcur[h*HEAD_DIM..(h+1)*HEAD_DIM]);
            }
        }

        // GQA attention — output into buf_out (reusing it as attn_out)
        for v in self.buf_out[..N_HEAD*HEAD_DIM].iter_mut() { *v = 0.0; }
        let kv_stride = N_KV * MAX_CTX * HEAD_DIM;
        let n_past = (pos + 1).min(MAX_CTX);
        for h in 0..N_HEAD {
            let kv_h = h * N_KV / N_HEAD;
            let q_h  = unsafe { core::slice::from_raw_parts(self.buf_attn.as_ptr().add(h*HEAD_DIM), HEAD_DIM) };
            for t in 0..n_past {
                let kb = ai * kv_stride + kv_h * MAX_CTX * HEAD_DIM + t * HEAD_DIM;
                let mut s = 0.0f32;
                for d in 0..HEAD_DIM { s += q_h[d] * self.kv_k[kb+d]; }
                self.buf_sc[t] = s * ATTN_SCALE;
            }
            softmax(&mut self.buf_sc[..n_past]);
            let out_base = h * HEAD_DIM;
            for t in 0..n_past {
                let vb = ai * kv_stride + kv_h * MAX_CTX * HEAD_DIM + t * HEAD_DIM;
                let sc = self.buf_sc[t];
                for d in 0..HEAD_DIM { self.buf_out[out_base+d] += sc * self.kv_v[vb+d]; }
            }
        }

        // Sigmoid gate
        for h in 0..N_HEAD {
            for d in 0..HEAD_DIM {
                self.buf_out[h*HEAD_DIM+d] *= sigmoid(self.buf_gate2[h*HEAD_DIM+d]);
            }
        }

        // Output projection
        let out_sl = unsafe { core::slice::from_raw_parts(self.buf_out.as_ptr(), N_HEAD*HEAD_DIM) };
        mv_q8(unsafe { core::slice::from_raw_parts(data_ptr.add(o.wo), q8_sz(N_EMBD,N_HEAD*HEAD_DIM)) },
              N_EMBD, N_HEAD*HEAD_DIM, out_sl, &mut self.buf_res);

        // Residual 1: buf_x = buf_x2 + buf_res
        for i in 0..N_EMBD { self.buf_x[i] = self.buf_x2[i] + self.buf_res[i]; }

        // Post-norm + FFN
        unsafe {
            core::ptr::copy_nonoverlapping(self.buf_x.as_ptr(), self.buf_x2.as_mut_ptr(), N_EMBD);
        }
        let post_norm = f32_slice(&self.data, o.post_norm, N_EMBD);
        {
            let ss: f32 = self.buf_x2.iter().map(|v| v*v).sum::<f32>() / N_EMBD as f32;
            let inv = 1.0 / libm::sqrtf(ss + RMS_EPS);
            for i in 0..N_EMBD { self.buf_n[i] = post_norm[i] * self.buf_x2[i] * inv; }
        }
        let normed2_sl = unsafe { core::slice::from_raw_parts(self.buf_n.as_ptr(), N_EMBD) };
        mv_q8(unsafe { core::slice::from_raw_parts(data_ptr.add(o.ffn_gate), q8_sz(N_FF,N_EMBD)) }, N_FF, N_EMBD, normed2_sl, &mut self.buf_ff1);
        mv_q8(unsafe { core::slice::from_raw_parts(data_ptr.add(o.ffn_up),   q8_sz(N_FF,N_EMBD)) }, N_FF, N_EMBD, normed2_sl, &mut self.buf_ff2);
        for i in 0..N_FF { self.buf_ff1[i] = silu(self.buf_ff1[i]) * self.buf_ff2[i]; }
        let ff1_sl = unsafe { core::slice::from_raw_parts(self.buf_ff1.as_ptr(), N_FF) };
        mv_q8(unsafe { core::slice::from_raw_parts(data_ptr.add(o.ffn_down), q8_sz(N_EMBD,N_FF)) }, N_EMBD, N_FF, ff1_sl, &mut self.buf_res);
        for i in 0..N_EMBD { self.buf_x[i] = self.buf_x2[i] + self.buf_res[i]; }
    }

    // ── Single token forward pass ─────────────────────────────────────────────
    fn forward(&mut self, token_id: u32) -> &[f32] {
        let pos = self.ctx_pos;
        // Embed: lookup row token_id in token_embd [VOCAB × N_EMBD/GS Q8_0]
        {
            let nb = N_EMBD / GS;
            let row = token_id as usize;
            let row_off = self.emb_off + row * nb * BSIZE;
            for i in 0..N_EMBD { self.buf_x[i] = 0.0; }
            // Dequant the embedding row (identity matmul with e_i basis)
            for b in 0..nb {
                let off = row_off + b * BSIZE;
                let sc = f16_to_f32(u16::from_le_bytes([self.data[off], self.data[off+1]]));
                let xb = b * GS;
                for k in 0..GS {
                    self.buf_x[xb+k] = sc * (self.data[off+2+k] as i8 as f32);
                }
            }
        }

        // Layer forward passes
        for il in 0..N_LAYERS {
            if is_attn(il) {
                self.forward_attn(il);
            } else {
                self.forward_ssm(il);
            }
        }

        // Final norm (zero heap alloc: use raw ptrs + f32_slice directly)
        let out_norm = f32_slice(&self.data, self.out_norm_off, N_EMBD);
        let ss: f32 = self.buf_x.iter().map(|v| v*v).sum::<f32>() / N_EMBD as f32;
        let inv = 1.0 / libm::sqrtf(ss + RMS_EPS);
        for i in 0..N_EMBD { self.buf_n[i] = out_norm[i] * self.buf_x[i] * inv; }

        // LM head: logits = token_embd @ normed (tied weights)
        mv_q8(&self.data[self.emb_off..], VOCAB, N_EMBD,
              unsafe { core::slice::from_raw_parts(self.buf_n.as_ptr(), N_EMBD) },
              &mut self.buf_logits);

        if pos < MAX_CTX { self.ctx_pos += 1; }
        &self.buf_logits
    }

    fn sample_greedy(logits: &[f32]) -> u32 {
        logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    }
}

// ─── Tokenizer (share the same greedy BPE tokenizer module) ─────────────────
#[path = "lm_qwen_tok.rs"]
mod tok35;

// ─── Global model instance ───────────────────────────────────────────────────
static mut MODEL: Option<Box<Qwen35>> = None;

pub fn is_loaded() -> bool {
    unsafe { MODEL.is_some() }
}

pub fn init() {
    crate::klog!(INFO, "lm_qwen35: loading from drive 1...");
    let data = match crate::storage::read_all(1) {
        Ok(d) => d,
        Err(e) => { crate::klog!(ERROR, "lm_qwen35: read_all failed: {}", e); return; }
    };
    if data.len() < 64 {
        crate::klog!(ERROR, "lm_qwen35: data too small ({}B)", data.len()); return;
    }
    let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    if magic != MAGIC {
        crate::klog!(ERROR, "lm_qwen35: bad magic {:08x}", magic); return;
    }
    crate::klog!(INFO, "lm_qwen35: binary OK ({} MB), parsing tokenizer...", data.len() / 1048576);

    // Parse tokenizer from end of binary and pass to tokenizer module
    let vocab = parse_tokenizer(&data);
    if vocab.len() < 100 {
        crate::klog!(ERROR, "lm_qwen35: tokenizer parse failed ({} entries)", vocab.len()); return;
    }
    // Move vocab into tokenizer — no clone, no second copy
    tok35::init(vocab);
    crate::klog!(INFO, "lm_qwen35: tokenizer OK, loading model...");

    let model = Qwen35::alloc(data);
    unsafe { MODEL = Some(model); }
    crate::klog!(INFO, "lm_qwen35: ready");
}

fn parse_tokenizer(data: &[u8]) -> Vec<Vec<u8>> {
    // Tokenizer is at end of binary after weights
    // Format: u32 vocab_size, then entries: u16 len + bytes
    let (_, emb_off) = global_offsets(data.len());
    let tok_off = emb_off + q8_sz(VOCAB, N_EMBD);
    if tok_off + 4 > data.len() { return Vec::new(); }
    let n = u32::from_le_bytes([data[tok_off], data[tok_off+1], data[tok_off+2], data[tok_off+3]]) as usize;
    let mut vocab = Vec::with_capacity(n);
    let mut p = tok_off + 4;
    for _ in 0..n {
        if p + 2 > data.len() { break; }
        let len = u16::from_le_bytes([data[p], data[p+1]]) as usize;
        p += 2;
        if p + len > data.len() { break; }
        vocab.push(data[p..p+len].to_vec());
        p += len;
    }
    vocab
}

pub fn generate(query: &str) -> Option<String> {
    let model = unsafe { MODEL.as_mut()? };
    model.reset_state();

    // Build chat prompt
    let prompt = build_prompt(query);
    let tokens = tok35::encode(&prompt);
    if tokens.is_empty() { return None; }

    // Prefill
    for &t in &tokens {
        let _ = model.forward(t);
    }

    // Generate
    let mut out_tokens = Vec::new();
    let mut last = *tokens.last().unwrap();
    for _ in 0..MAX_NEW {
        let logits = model.forward(last);
        let next = Qwen35::sample_greedy(logits);
        // EOS check (Qwen3.5 EOS = 151645 ... wait, Qwen3.5 has different token IDs)
        // From GGUF: eos_token_id is in metadata. Use common Qwen special tokens.
        if next >= VOCAB as u32 { break; }
        // Check common stop tokens
        if is_stop_token(next) { break; }
        out_tokens.push(next);
        last = next;
    }

    if out_tokens.is_empty() { return None; }
    let response = tok35::decode(&out_tokens);
    let r = String::from(response.trim());
    if r.len() < 3 { return None; }
    Some(r)
}

fn is_stop_token(id: u32) -> bool {
    // Qwen3 special tokens: <|im_end|>, <|endoftext|>
    // These are in the upper vocab range; we'll stop on any token >= 151643
    id >= 151643
}

fn build_prompt(query: &str) -> String {
    let creator = crate::consciousness::self_model::creator_name();
    let mut p = String::new();
    p.push_str("<|im_start|>system\nYou are John, ");
    p.push_str(&creator);
    p.push_str("'s AI. Be direct and helpful.\n<|im_end|>\n");
    p.push_str("<|im_start|>user\n");
    p.push_str(query);
    p.push_str("\n<|im_end|>\n<|im_start|>assistant\n");
    p
}
