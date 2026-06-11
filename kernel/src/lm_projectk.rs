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
const TEMP:     f32 = 0.85;
const GROUP_SZ: usize = 64;
const SCR_SZ:   usize = 96 + 144 + 32 + 48 + 192 + 192 + 768 + 4539; // = 6007
const S_QF: usize = 0; const S_QM: usize = 96; const S_X: usize = 240;
const S_FC: usize = 432; const S_LOG: usize = 1200;

// ── Offset-based weight refs (NO Vec pointers inside model) ─────────────
#[derive(Clone, Copy)]
struct MatOff { p: usize, s: usize, rows: usize, cols: usize, ng: usize, np: usize }
#[derive(Clone, Copy)]
struct F32Off { o: usize, n: usize }

// ── Flat engine: single Vec<u8> for ALL weight data ───────────────────
struct ModelFlat { w: Vec<u8>, scratch: [f32; SCR_SZ], rng: AtomicU64 }
static ENGINE: Once<Box<ModelFlat>> = Once::new();

fn rd32(d: &[u8], o: &mut usize) -> u32 {
    let v = u32::from_le_bytes([d[*o], d[*o+1], d[*o+2], d[*o+3]]); *o += 4; v
}
fn rdi4(d: &[u8], o: &mut usize, rows: usize, cols: usize) -> (usize, usize) {
    let ng = (cols + GROUP_SZ - 1) / GROUP_SZ;
    let np = ng * GROUP_SZ / 2;
    let p = *o; *o += rows * (np + ng * 4);
    (p, rows * np) // offset, packed_bytes
}

pub fn init() {
    let bin = include_bytes!("projectk_weights.bin");
    let mut o = 4usize;
    let _v=rd32(bin,&mut o);let _d=rd32(bin,&mut o);let _nl=rd32(bin,&mut o);
    let _h0=rd32(bin,&mut o);let _h1=rd32(bin,&mut o);let _gs=rd32(bin,&mut o);
    let _bs=rd32(bin,&mut o);

    // Load ALL weight data into one Vec<u8>. The parser records offsets
    // which will be used during inference to read directly from this buffer.
    let w = Vec::from(&bin[..]); // ONE allocation
    
    ENGINE.call_once(|| Box::new(ModelFlat { w, scratch: [0.0; SCR_SZ], rng: AtomicU64::new(0) }));
    LOADED.store(true, Ordering::Release);
    crate::klog!(INFO, "lm_projectk: flat-alloc engine (1 Vec<u8> for all weights)");
}

/// Access a contiguous scale/weight region as &[f32]
fn f32s(w: &[u8], off: usize, n: usize) -> &[f32] {
    unsafe { core::slice::from_raw_parts(w[off..].as_ptr() as *const f32, n) }
}

/// INT4 mat-vec: w_slice at `off`, first `packed_bytes` are nibbles, then f32 scales
fn mv4(w: &[u8], off: usize, packed: usize, rows: usize, cols: usize,
       ng: usize, np: usize, x: &[f32], out: &mut [f32]) {
    for i in 0..rows {
        let mut acc = 0.0f32;
        let base = off + i * np * 2;
        let sbase = off + packed * 2 + i * ng * 4; // scales start after all packed data
        for g in 0..ng {
            let sc = f32::from_le_bytes([w[sbase+g*4], w[sbase+g*4+1], w[sbase+g*4+2], w[sbase+g*4+3]]);
            for k in 0..GROUP_SZ {
                let col = g * GROUP_SZ + k;
                if col >= cols { break; }
                let byte = w[base + (g * GROUP_SZ + k) / 2];
                let nib = if (g * GROUP_SZ + k) % 2 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F };
                let q = if nib >= 8 { nib as i32 - 16 } else { nib as i32 };
                acc += q as f32 * sc * x[col];
            }
        }
        out[i] += acc;
    }
}

pub fn is_loaded() -> bool { LOADED.load(Ordering::Acquire) }
pub fn gen_count() -> u64 { GEN_COUNT.load(Ordering::Relaxed) }

/// Generate a response (currently returns None — templated fallback used instead)
pub fn generate(_prompt: &str) -> Option<String> {
    let _e = ENGINE.get()?;
    None // Full GLA forward pass will be wired in following commit
}

pub fn report() -> String {
    format!("Project-K: flat-alloc, {} KB weights", alloc::format!("{}", ENGINE.get().map(|e| e.w.len()/1024).unwrap_or(0)))
}