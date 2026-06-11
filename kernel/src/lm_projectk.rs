//! Project-K flat-alloc GLA inference engine.
//! ONE Vec<u8> for ALL weight data. MatOff tuples reference sub-ranges.
//! Zero internal Vec/Box pointers inside the model. LLVM cannot alias.

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
const MLP_D: usize = 768; const MAX_GEN: usize = 128; const CTX_WIN: usize = 20;
const TOP_K: usize = 40; const TOP_P: f32 = 0.92; const TEMP: f32 = 0.85;
const GROUP_SZ: usize = 64;
const S_LOG: usize = 432; const SCR_SZ: usize = S_LOG + 4539;

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
    let bin = include_bytes!("projectk_weights.bin");
    let mut o = 4usize;
    let _v=rd4(bin,&mut o);let _d=rd4(bin,&mut o);let _nl=rd4(bin,&mut o);
    let _h0=rd4(bin,&mut o);let _h1=rd4(bin,&mut o);let _gs=rd4(bin,&mut o);
    let _bs=rd4(bin,&mut o);

    let mut w = Vec::from(&bin[..]);
    let mut po = 32usize;
    // All parsing reads from &w (shared borrow). After parsing, w is moved into the struct.
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
    }));
    LOADED.store(true, Ordering::Release);
    crate::klog!(INFO, "lm_projectk: flat-alloc engine online ({} KB weights)", wlen/1024);
}

fn f32s<'a>(w: &'a [u8], mo: &MatOff) -> &'a [f32] {
    unsafe { core::slice::from_raw_parts(w[mo.p..].as_ptr() as *const f32, mo.cols) }
}
fn mv4(w: &[u8], m: &MatOff, x: &[f32], out: &mut [f32]) {
    for i in 0..m.rows {
        let mut acc = 0.0;
        for g in 0..m.ng {
            let sc = f32::from_le_bytes([w[m.s+(i*m.ng+g)*4], w[m.s+(i*m.ng+g)*4+1], w[m.s+(i*m.ng+g)*4+2], w[m.s+(i*m.ng+g)*4+3]]);
            for k in 0..GROUP_SZ {
                let col = g*GROUP_SZ+k;
                if col >= m.cols { break; }
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
        let mut qv = [0.0; 144]; let qvl = if dh==DH0 {96} else {144};
        let qs = &mut qv[..qvl];
        mv4(&e.w, &e.blk_qkv[bi][hi], inp, qs);
        let qb = f32s(&e.w, &e.blk_qb[bi][hi]);
        for i in 0..qvl { qs[i] += qb[i]; }
        let (q, k, v) = (&qs[..dh], &qs[dh..2*dh], &qs[2*dh..3*dh]);
        let fgw = f32s(&e.w, &e.blk_fgw[bi][hi]);
        let fg = sigm(dot(fgw, inp) + e.blk_fgb[bi][hi]);
        for i in 0..dh { state[i] = fg*state[i] + (1.0-fg)*elu1(k[i])*v[i]; }
        if t+1 == ctx {
            let mut hv = [0.0; 64];
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
    let mut emb = [[0.0; D]; CTX_WIN];
    let w = &e.w; let mo = &e.emb;

    for (ti, &tok) in tokens[ts..].iter().enumerate() {
        let row = (tok as usize).min(VOCAB-1);
        for g in 0..mo.ng {
            let sc = f32::from_le_bytes([w[mo.s+(row*mo.ng+g)*4], w[mo.s+(row*mo.ng+g)*4+1], w[mo.s+(row*mo.ng+g)*4+2], w[mo.s+(row*mo.ng+g)*4+3]]);
            for k in 0..GROUP_SZ {
                let col = g*GROUP_SZ+k;
                if col >= D { break; }
                let byte = w[mo.p+(row*mo.np*2+g*GROUP_SZ+k)/2];
                let nib = if (g*GROUP_SZ+k)%2==0 { byte&0x0F } else { (byte>>4)&0x0F };
                let q = if nib>=8 { nib as i32-16 } else { nib as i32 };
                emb[ti][col] = q as f32 * sc;
            }
        }
    }

    let mut res = emb[ctx-1];
    let mut fs = [0.0; DH0]; let mut ms = [0.0; DH1];
    let mut fo = [0.0; D]; let mut mo = [0.0; D];

    for bi in 0..6 {
        let mut h = res;
        let l1w = f32s(w, &e.blk_ln1w[bi]); let l1b = f32s(w, &e.blk_ln1b[bi]);
        lnorm(&mut h, l1w, l1b);
        let mut le = [[0.0; D]; CTX_WIN];
        for ti in 0..ctx { let mut e2 = emb[ti]; lnorm(&mut e2, l1w, l1b); le[ti]=e2; }
        le[ctx-1] = h;
        gla(e, bi, 0, DH0, &le, ctx, &mut fs, &mut fo);
        gla(e, bi, 1, DH1, &le, ctx, &mut ms, &mut mo);
        for i in 0..D { res[i] += fo[i] + mo[i]; }

        let mut h2 = res;
        let l2w = f32s(w, &e.blk_ln2w[bi]); let l2b = f32s(w, &e.blk_ln2b[bi]);
        lnorm(&mut h2, l2w, l2b);
        let mut fc = [0.0; MLP_D];
        mv4_out(w, &e.blk_fc[bi], &h2, &mut fc);
        let fcb = f32s(w, &e.blk_fcb[bi]);
        for i in 0..MLP_D { fc[i] = gelu(fc[i]+fcb[i]); }
        let mut mpo = [0.0; D];
        mv4_out(w, &e.blk_pr[bi], &fc, &mut mpo);
        let prb = f32s(w, &e.blk_prb[bi]);
        for i in 0..D { res[i] += mpo[i] + prb[i]; }
    }

    let lfw = f32s(w, &e.lnf_w); let lfb = f32s(w, &e.lnf_b);
    lnorm(&mut res, lfw, lfb);
    let logits = unsafe { &mut *(e.scratch.as_ptr() as *mut [f32; VOCAB]) };
    for l in logits.iter_mut() { *l=0.0; }
    mv4(w, &e.emb, &res, logits);
}

fn sample(logits: &[f32], e: &ModelFlat) -> usize {
    // Temperature
    let mut l: Vec<f32> = logits.iter().map(|&v| v/TEMP).collect();
    // Softmax
    let mx = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = l.iter_mut().map(|v| { *v = libm::expf(*v-mx); *v as f64 }).sum();
    if sum <= 0.0 { return 0; }
    // Top-k filter
    let mut tv = [f32::NEG_INFINITY; TOP_K];
    for &v in &l {
        if v > tv[TOP_K-1] { tv[TOP_K-1] = v;
            let mut j = TOP_K-1; while j>0 && tv[j] > tv[j-1] { tv.swap(j,j-1); j-=1; } }
    }
    let kth = tv[TOP_K-1];
    for v in l.iter_mut() { if *v < kth { *v = 0.0; } }
    // Sample
    let total: f64 = l.iter().map(|&v| v as f64).sum();
    let mut seed = e.rng.load(Ordering::Relaxed);
    if seed==0 { seed=12345; }
    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    e.rng.store(seed, Ordering::Relaxed);
    let r = ((seed>>33) as f64)/2147483648.0 * total;
    let mut cum = 0.0;
    for (i, &v) in l.iter().enumerate() {
        cum += v as f64;
        if cum >= r { return i; }
    }
    0
}

pub fn is_loaded() -> bool { LOADED.load(Ordering::Acquire) }
pub fn gen_count() -> u64 { GEN_COUNT.load(Ordering::Relaxed) }

pub fn generate(prompt: &str) -> Option<String> {
    if !LOADED.load(Ordering::Acquire) { return None; }
    let e = ENGINE.get()?;

    // Encode prompt
    let mut toks: Vec<u16> = Vec::with_capacity(64);
    toks.push(3); // BOS
    for ch in prompt.chars() {
        let cp = ch as u32;
        if let Ok(idx) = PKK_CP2TOK.binary_search_by_key(&cp, |x| x.0) {
            toks.push(PKK_CP2TOK[idx].1);
        }
    }
    if toks.len() < 2 { toks.push(encode_char('?')); }

    let mut out = String::new();
    let logits = unsafe { &e.scratch[..VOCAB] };

    for _ in 0..MAX_GEN {
        let tok_slice: &[u16] = &toks;
        forward(e, tok_slice);
        let best = sample(logits, e);
        if best == 0 || best >= PKK_ITOS.len() { break; }
        let s = PKK_ITOS[best];
        if s == "
" || s.len() > 10 { break; }
        out.push_str(s);
        if out.len() > 200 { break; }
        toks.push(best as u16);
        if toks.len() > CTX_WIN { toks.remove(0); }
    }
fn encode_char(ch: char) -> u16 {
    let cp = ch as u32;
    if let Ok(idx) = PKK_CP2TOK.binary_search_by_key(&cp, |e| e.0) {
        PKK_CP2TOK[idx].1
    } else {
        3 // fallback: space/unknown
    }
}

    GEN_COUNT.fetch_add(1, Ordering::Relaxed);
    if out.is_empty() { None } else { Some(out) }
}

pub fn report() -> String {
    format!("Project-K flat-alloc: {} KB, {} gens", ENGINE.get().map(|e| e.w.len()/1024).unwrap_or(0), GEN_COUNT.load(Ordering::Relaxed))
}