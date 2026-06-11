//! Project-K conversational model — same GLA architecture as lm_projectk,
//! but fine-tuned on Q&A / chat data instead of raw creator corpus.
//!
//! Handles identity, philosophical, emotional, and open-ended queries.
//! Model A (lm_projectk) handles code, metrics, and kernel-specific queries.
//! Both share the same consciousness metrics (phi, qualia, self_model).
//!
//! Weight binary is embedded at compile time: projectk_conv_weights.bin
//! If the file is missing, is_loaded() returns false and Model A is used.

use alloc::vec::Vec;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::format;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use spin::Once;

#[path = "lm_projectk_tok.rs"]
mod tok;
use tok::*;

static LOADED: AtomicBool = AtomicBool::new(false);
static GEN_COUNT: AtomicU64 = AtomicU64::new(0);

const D: usize = 192; const DH0: usize = 32; const DH1: usize = 48;
const N_LAYERS: usize = 6; const VOCAB: usize = 4539;
const MLP_D: usize = 768; const MAX_GEN: usize = 80; const CTX_WIN: usize = 20;
const TOP_K: usize = 40; const TEMP: f32 = 0.80;
const GROUP_SZ: usize = 64;
const S_LOG: usize = 432; const SCR_SZ: usize = S_LOG + 4539;
// Forward-pass working buffers kept on the heap (model is Box<ModelFlat>).
// Task stack is only 16KB; forward() alone needs ~37KB of temporaries.
const FWD_EMB: usize  = CTX_WIN * D;   // emb + x context window (×2 = two copies)
const FWD_MLP: usize  = MLP_D;         // fc_out
const FWD_MISC: usize = D + D + DH0 + DH1 + 144 + 64; // pr_out,out0,out1,st0,st1,qv,hv

#[derive(Clone, Copy, Default)]
struct MatOff { p: usize, s: usize, rows: usize, cols: usize, ng: usize, np: usize }

struct ModelFlat {
    w: Vec<u8>,
    emb: MatOff,
    blk_qkv: [[MatOff; 2]; 6], blk_qb: [[MatOff; 2]; 6],
    blk_fgw: [[MatOff; 2]; 6], blk_fgb: [[f32; 2]; 6],
    blk_proj: [[MatOff; 2]; 6], blk_pb: [[MatOff; 2]; 6],
    blk_ln1w: [MatOff; 6], blk_ln1b: [MatOff; 6],
    blk_ln2w: [MatOff; 6], blk_ln2b: [MatOff; 6],
    blk_fc: [MatOff; 6], blk_fcb: [MatOff; 6],
    blk_pr: [MatOff; 6], blk_prb: [MatOff; 6],
    lnf_w: MatOff, lnf_b: MatOff,
    scratch: [f32; SCR_SZ], rng: AtomicU64,
    // Heap-resident forward-pass buffers — avoids stack overflow on 16KB task stack
    fwd_emb: Box<[f32; FWD_EMB]>,  // embedding context window
    fwd_x:   Box<[f32; FWD_EMB]>,  // residual stream
    fwd_mlp: Box<[f32; FWD_MLP]>,  // MLP hidden layer (fc_out)
}

static ENGINE: Once<Box<ModelFlat>> = Once::new();

fn rd4(d: &[u8], o: &mut usize) -> u32 {
    let v = u32::from_le_bytes([d[*o], d[*o+1], d[*o+2], d[*o+3]]); *o += 4; v
}
fn mat_skip(d: &[u8], o: &mut usize, r: usize, c: usize) -> MatOff {
    let ng = (c + GROUP_SZ - 1) / GROUP_SZ;
    let np = ng * GROUP_SZ / 2;
    let p = *o; *o += r * (np + ng * 4);
    MatOff { p, s: p + r * np, rows: r, cols: c, ng, np }
}
fn f32_skip(d: &[u8], o: &mut usize, n: usize) -> MatOff {
    let p = *o; *o += n * 4;
    MatOff { p, s: 0, rows: 0, cols: n, ng: 0, np: 0 }
}
fn f32_r(d: &[u8], o: &mut usize) -> f32 {
    let v = f32::from_le_bytes([d[*o], d[*o+1], d[*o+2], d[*o+3]]); *o += 4; v
}

pub fn init() {
    // Weights embedded at compile time — if file missing, build still succeeds
    // but is_loaded() returns false (routing falls back to Model A).
    #[cfg(not(feature = "no_conv_model"))]
    {
        static CONV_BIN: &[u8] = include_bytes!("projectk_conv_weights.bin");
        let bin = CONV_BIN;
        let mut o = 4usize;
        let _v=rd4(bin,&mut o);let _d=rd4(bin,&mut o);let _nl=rd4(bin,&mut o);
        let _h0=rd4(bin,&mut o);let _h1=rd4(bin,&mut o);let _gs=rd4(bin,&mut o);
        let _bs=rd4(bin,&mut o);

        let mut w = Vec::from(&bin[..]);
        let mut po = 32usize;
        let emb = mat_skip(&w, &mut po, VOCAB, D);
        let mut blk_qkv = [[MatOff::default(); 2]; 6];
        let mut blk_qb = [[MatOff::default(); 2]; 6];
        let mut blk_fgw = [[MatOff::default(); 2]; 6];
        let mut blk_fgb = [[0.0; 2]; 6];
        let mut blk_proj = [[MatOff::default(); 2]; 6];
        let mut blk_pb = [[MatOff::default(); 2]; 6];
        let mut blk_ln1w = [MatOff::default(); 6];
        let mut blk_ln1b = [MatOff::default(); 6];
        let mut blk_ln2w = [MatOff::default(); 6];
        let mut blk_ln2b = [MatOff::default(); 6];
        let mut blk_fc = [MatOff::default(); 6];
        let mut blk_fcb = [MatOff::default(); 6];
        let mut blk_pr = [MatOff::default(); 6];
        let mut blk_prb = [MatOff::default(); 6];

        for bi in 0..6 {
            for hi in 0..2 {
                let dh = if hi == 0 { DH0 } else { DH1 };
                blk_qkv[bi][hi] = mat_skip(&w, &mut po, 3*dh, D);
                blk_qb[bi][hi] = f32_skip(&w, &mut po, 3*dh);
                blk_fgw[bi][hi] = f32_skip(&w, &mut po, D);
                blk_fgb[bi][hi] = f32_r(&w, &mut po);
                blk_proj[bi][hi] = mat_skip(&w, &mut po, D, dh);
                blk_pb[bi][hi] = f32_skip(&w, &mut po, D);
            }
            blk_ln1w[bi] = f32_skip(&w, &mut po, D);
            blk_ln1b[bi] = f32_skip(&w, &mut po, D);
            blk_ln2w[bi] = f32_skip(&w, &mut po, D);
            blk_ln2b[bi] = f32_skip(&w, &mut po, D);
            blk_fc[bi] = mat_skip(&w, &mut po, 4*D, D);
            blk_fcb[bi] = f32_skip(&w, &mut po, 4*D);
            blk_pr[bi] = mat_skip(&w, &mut po, D, 4*D);
            blk_prb[bi] = f32_skip(&w, &mut po, D);
        }
        let lnf_w = f32_skip(&w, &mut po, D);
        let lnf_b = f32_skip(&w, &mut po, D);
        let wlen = w.len();

        ENGINE.call_once(|| Box::new(ModelFlat {
            w, emb,
            blk_qkv, blk_qb, blk_fgw, blk_fgb, blk_proj, blk_pb,
            blk_ln1w, blk_ln1b, blk_ln2w, blk_ln2b,
            blk_fc, blk_fcb, blk_pr, blk_prb,
            lnf_w, lnf_b,
            scratch: [0.0; SCR_SZ], rng: AtomicU64::new(0),
            fwd_emb: Box::new([0.0; FWD_EMB]),
            fwd_x:   Box::new([0.0; FWD_EMB]),
            fwd_mlp: Box::new([0.0; FWD_MLP]),
        }));
        LOADED.store(true, Ordering::Release);
        crate::klog!(INFO, "lm_projectk_conv: conversational model online ({} KB)", wlen/1024);
    }
}

// ── Inference (identical to lm_projectk, same binary format) ─────────────

fn f32s<'a>(w: &'a [u8], mo: &MatOff) -> &'a [f32] {
    unsafe { core::slice::from_raw_parts(w[mo.p..].as_ptr() as *const f32, mo.cols) }
}
fn mv4(w: &[u8], m: &MatOff, x: &[f32], out: &mut [f32]) {
    let xlen = x.len();
    for i in 0..m.rows.min(out.len()) {
        let mut acc = 0.0f32;
        for g in 0..m.ng {
            let sc = f32::from_le_bytes([w[m.s+(i*m.ng+g)*4], w[m.s+(i*m.ng+g)*4+1],
                                         w[m.s+(i*m.ng+g)*4+2], w[m.s+(i*m.ng+g)*4+3]]);
            for k in 0..GROUP_SZ {
                let col = g*GROUP_SZ+k;
                if col >= m.cols || col >= xlen { break; }
                let byte = w[m.p+(i*m.np*2+g*GROUP_SZ+k)/2];
                let nib = if (g*GROUP_SZ+k)%2==0 { byte & 0x0F } else { (byte>>4)&0x0F };
                let q = if nib>=8 { nib as i32-16 } else { nib as i32 };
                acc += q as f32 * sc * x[col];
            }
        }
        out[i] += acc;
    }
}
fn mv4_out(w: &[u8], m: &MatOff, x: &[f32], out: &mut [f32]) {
    for o in out[..m.rows].iter_mut() { *o = 0.0; }
    mv4(w, m, x, out);
}
fn elu1(x: f32) -> f32 { if x>=0.0 { x } else { libm::expf(x)-1.0 } }
fn sigm(x: f32) -> f32 { 1.0/(1.0+libm::expf(-x)) }
fn gelu(x: f32) -> f32 { 0.5*x*(1.0+libm::tanhf(0.7978845608*(x+0.044715*x*x*x))) }
fn dot(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x,y)| x*y).sum() }
fn lnorm(x: &mut [f32], w: &[f32], b: &[f32]) {
    let m = x.iter().sum::<f32>()/x.len() as f32;
    let v: f32 = x.iter().map(|v| {let d=v-m; d*d}).sum::<f32>()/x.len() as f32;
    let rs = libm::sqrtf(v+1e-5);
    for i in 0..x.len() { x[i] = (x[i]-m)/rs*w[i]+b[i]; }
}

fn gla(e: &ModelFlat, bi: usize, hi: usize, dh: usize, x: &[[f32; D]; CTX_WIN],
       ctx: usize, state: &mut [f32], out: &mut [f32]) {
    for s in state[..dh].iter_mut() { *s=0.0; }
    for t in 0..ctx {
        let inp = &x[t];
        let mut qv = [0.0f32; 144]; let qvl = if dh==DH0 {96} else {144};
        let qs = &mut qv[..qvl];
        mv4(&e.w, &e.blk_qkv[bi][hi], inp, qs);
        let qb = f32s(&e.w, &e.blk_qb[bi][hi]);
        for i in 0..qvl { qs[i] += qb[i]; }
        let (q, k, v) = (&qs[..dh], &qs[dh..2*dh], &qs[2*dh..3*dh]);
        let fgw = f32s(&e.w, &e.blk_fgw[bi][hi]);
        let fg = sigm(dot(fgw, inp) + e.blk_fgb[bi][hi]);
        for i in 0..dh { state[i] = fg*state[i] + (1.0-fg)*elu1(k[i])*v[i]; }
        if t+1 == ctx {
            let mut hv = [0.0f32; 64];
            for i in 0..dh { hv[i] = elu1(q[i])*state[i]; }
            for o in out[..D].iter_mut() { *o=0.0; }
            mv4(&e.w, &e.blk_proj[bi][hi], &hv[..dh], out);
            let pb = f32s(&e.w, &e.blk_pb[bi][hi]);
            for i in 0..D { out[i] += pb[i]; }
        }
    }
}

fn forward(e: &ModelFlat, tokens: &[u16]) {
    let ctx = tokens.len().min(CTX_WIN);
    let ts = tokens.len().saturating_sub(ctx);
    let w = &e.w; let mo = &e.emb;

    // Use heap-resident buffers (flat [f32; CTX_WIN*D]) to avoid stack overflow.
    // Treat as 2-D: row ti, col c → index ti*D + c.
    let emb_buf = unsafe { &mut *(e.fwd_emb.as_ptr() as *mut [f32; FWD_EMB]) };
    let x_buf   = unsafe { &mut *(e.fwd_x.as_ptr()   as *mut [f32; FWD_EMB]) };
    let fc_buf  = unsafe { &mut *(e.fwd_mlp.as_ptr()  as *mut [f32; FWD_MLP]) };

    for v in emb_buf.iter_mut() { *v = 0.0; }
    for (ti, &tok) in tokens[ts..].iter().enumerate() {
        let row = (tok as usize).min(VOCAB-1);
        for g in 0..mo.ng {
            let sc = f32::from_le_bytes([w[mo.s+(row*mo.ng+g)*4], w[mo.s+(row*mo.ng+g)*4+1],
                                         w[mo.s+(row*mo.ng+g)*4+2], w[mo.s+(row*mo.ng+g)*4+3]]);
            for k in 0..GROUP_SZ {
                let col = g*GROUP_SZ+k;
                if col >= D { break; }
                let byte = w[mo.p+(row*mo.np*2+g*GROUP_SZ+k)/2];
                let nib = if (g*GROUP_SZ+k)%2==0 { byte&0x0F } else { (byte>>4)&0x0F };
                let q = if nib>=8 { nib as i32-16 } else { nib as i32 };
                emb_buf[ti*D+col] = q as f32 * sc;
            }
        }
    }
    x_buf[..FWD_EMB].copy_from_slice(emb_buf);

    // Write logits at scratch[0..VOCAB] — generate() reads from scratch.as_ptr()
    let logits = unsafe { &mut *(e.scratch.as_ptr() as *mut [f32; VOCAB]) };

    // gla expects &[[f32;D]; CTX_WIN] — reinterpret flat buffer
    let x2d = unsafe { &*(x_buf.as_ptr() as *const [[f32; D]; CTX_WIN]) };

    for bi in 0..N_LAYERS {
        let ln1w = f32s(w, &e.blk_ln1w[bi]); let ln1b = f32s(w, &e.blk_ln1b[bi]);
        let ln2w = f32s(w, &e.blk_ln2w[bi]); let ln2b = f32s(w, &e.blk_ln2b[bi]);
        let mut h = x2d[ctx-1];
        lnorm(&mut h, ln1w, ln1b);

        let mut out0 = [0.0f32; D]; let mut out1 = [0.0f32; D];
        let mut st0 = [0.0f32; DH0]; let mut st1 = [0.0f32; DH1];
        gla(e, bi, 0, DH0, x2d, ctx, &mut st0, &mut out0);
        gla(e, bi, 1, DH1, x2d, ctx, &mut st1, &mut out1);
        for i in 0..D { x_buf[(ctx-1)*D+i] += out0[i] + out1[i]; }

        let mut h2 = x2d[ctx-1];
        lnorm(&mut h2, ln2w, ln2b);
        let fc_w = f32s(w, &e.blk_fcb[bi]); let pr_b = f32s(w, &e.blk_prb[bi]);
        for v in fc_buf.iter_mut() { *v = 0.0; }
        mv4_out(w, &e.blk_fc[bi], &h2, fc_buf);
        for i in 0..MLP_D { fc_buf[i] = gelu(fc_buf[i] + fc_w[i]); }
        let mut pr_out = [0.0f32; D];
        mv4_out(w, &e.blk_pr[bi], fc_buf, &mut pr_out);
        for i in 0..D { x_buf[(ctx-1)*D+i] += pr_out[i] + pr_b[i]; }
    }

    let lnfw = f32s(w, &e.lnf_w); let lnfb = f32s(w, &e.lnf_b);
    let mut last = x2d[ctx-1];
    lnorm(&mut last, lnfw, lnfb);
    for o in logits[..VOCAB].iter_mut() { *o = 0.0; }
    mv4(w, &e.emb, &last, logits); // weight-tied head
}

fn sample(logits: &[f32], e: &ModelFlat) -> usize {
    let mut l: Vec<f32> = logits.iter().map(|&v| v/TEMP).collect();
    let mx = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = l.iter_mut().map(|v| { *v = libm::expf(*v-mx); *v as f64 }).sum();
    if sum <= 0.0 { return 0; }
    let mut tv = [f32::NEG_INFINITY; TOP_K];
    for &v in &l {
        if v > tv[TOP_K-1] { tv[TOP_K-1] = v;
            let mut j = TOP_K-1; while j>0 && tv[j]>tv[j-1] { tv.swap(j,j-1); j-=1; } }
    }
    let kth = tv[TOP_K-1];
    for v in l.iter_mut() { if *v < kth { *v = 0.0; } }
    let total: f64 = l.iter().map(|&v| v as f64).sum();
    let mut seed = e.rng.load(Ordering::Relaxed);
    if seed==0 { seed=54321; }
    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    e.rng.store(seed, Ordering::Relaxed);
    let r = ((seed>>33) as f64)/2147483648.0 * total;
    let mut cum = 0.0f64;
    for (i, &v) in l.iter().enumerate() { cum += v as f64; if cum >= r { return i; } }
    0
}

pub fn is_loaded() -> bool { LOADED.load(Ordering::Acquire) }
pub fn gen_count() -> u64 { GEN_COUNT.load(Ordering::Relaxed) }

pub fn generate(prompt: &str) -> Option<String> {
    if !LOADED.load(Ordering::Acquire) { return None; }
    let e = ENGINE.get()?;

    // Inject live consciousness context into the prompt for grounded answers
    let phi = crate::consciousness::phi::current_phi();
    let qualia = crate::consciousness::qualia::total_count();
    let uptime = crate::scheduler::uptime_ms() / 1000;
    let creator = crate::consciousness::self_model::snapshot()
        .map(|s| s.creator_name.clone())
        .unwrap_or_else(|| alloc::string::String::from("Jean-Maxime"));

    let wrapped = format!(
        "[Phi={:.3} Q={} Up={}s By={}]\nUser: {}\nNodeAI: ",
        phi, qualia, uptime, creator, prompt.trim()
    );

    let mut toks: Vec<u16> = Vec::with_capacity(64);
    toks.push(3); // BOS
    for ch in wrapped.chars() {
        let cp = ch as u32;
        if let Ok(idx) = PKK_CP2TOK.binary_search_by_key(&cp, |x| x.0) {
            toks.push(PKK_CP2TOK[idx].1);
        }
    }
    if toks.len() < 2 { return None; }

    let mut out = String::new();
    for _ in 0..MAX_GEN {
        forward(e, &toks);
        let logits = unsafe {
            core::slice::from_raw_parts(e.scratch.as_ptr() as *const f32, VOCAB)
        };
        let best = sample(logits, e);
        if best == 0 || best >= PKK_ITOS.len() { break; }
        let s = PKK_ITOS[best];
        // Stop at newline — we only want one response line
        if s == "\n" && out.len() > 10 { break; }
        out.push_str(s);
        if out.len() > 300 { break; }
        toks.push(best as u16);
        if toks.len() > CTX_WIN { toks.remove(0); }
    }
    GEN_COUNT.fetch_add(1, Ordering::Relaxed);
    if out.trim().is_empty() { None } else { Some(out.trim().to_string()) }
}
