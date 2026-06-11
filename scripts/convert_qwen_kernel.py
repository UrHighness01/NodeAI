#!/usr/bin/env python3
"""
Convert Qwen2.5-0.5B-Instruct GGUF to NodeAI kernel flat INT4 binary.

Output format (qwen25_kernel.bin):
  [Header 40 bytes]
    magic       u32  = 0x4E574B52  'NWKR' (NodeAI Qwen Kernel Runtime)
    version     u32  = 1
    n_layers    u32  = 24
    d_model     u32  = 896
    n_heads     u32  = 14
    n_kv_heads  u32  = 2
    ffn_dim     u32  = 4864
    vocab_size  u32  = 151936
    head_dim    u32  = 64
    group_sz    u32  = 32

  [Per layer (0..n_layers):]
    attn_norm_w : d_model × f32
    q_w         : INT4(n_heads*head_dim, d_model)
    q_b         : n_heads*head_dim × f32
    k_w         : INT4(n_kv_heads*head_dim, d_model)
    k_b         : n_kv_heads*head_dim × f32
    v_w         : INT4(n_kv_heads*head_dim, d_model)
    v_b         : n_kv_heads*head_dim × f32
    o_w         : INT4(d_model, n_heads*head_dim)
    ffn_norm_w  : d_model × f32
    gate_w      : INT4(ffn_dim, d_model)
    up_w        : INT4(ffn_dim, d_model)
    down_w      : INT4(d_model, ffn_dim)

  [After all layers:]
    out_norm_w  : d_model × f32
    lm_head_w   : INT4(vocab_size, d_model)
    emb_w       : INT4(vocab_size, d_model)

  [Tokenizer section:]
    n_vocab     u32
    for each token:
      tok_len   u16
      tok_bytes : tok_len × u8

INT4 layout for matrix (rows, cols), group_sz=32:
  n_groups = ceil(cols / group_sz)
  Nibbles  : rows × n_groups × 16 bytes  (32 nibbles packed 2-per-byte)
  Scales   : rows × n_groups × 4 bytes   (f32 absmax scale per group)
  Total    : rows × n_groups × 20 bytes
"""

import sys
import struct
import numpy as np
from pathlib import Path
import gguf

SCRIPT_DIR = Path(__file__).parent.resolve()
# GGUF is outside repo at ~/.openclaw/models/qwen/
GGUF_PATH = SCRIPT_DIR.parent.parent.parent.parent / "models" / "qwen" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
OUT_PATH  = SCRIPT_DIR.parent / "models" / "qwen25_kernel.bin"

GROUP_SZ = 32
MAGIC    = 0x4E574B52  # 'NWKR'

N_LAYERS  = 24
D_MODEL   = 896
N_HEADS   = 14
N_KV      = 2
FFN_DIM   = 4864
VOCAB     = 151936
HEAD_DIM  = 64

EOS_TOKEN = 151645   # <|im_end|>
BOS_TOKEN = 151644   # <|im_start|>
IM_START  = 151644
IM_END    = 151645


def dequant_tensor(t) -> np.ndarray:
    """Dequantize any GGUF tensor to float32 numpy array."""
    if t.tensor_type.name == "F32":
        return np.array(t.data, dtype=np.float32).reshape(t.shape)
    if t.tensor_type.name in ("F16", "BF16"):
        return t.data.view(np.float16).astype(np.float32).reshape(t.shape)
    # Quantized types: Q4_K_M, Q5_0, Q6_K, Q8_0, etc.
    from gguf.quants import dequantize
    arr = dequantize(t.data, t.tensor_type).astype(np.float32)
    return arr.reshape(t.shape)


def quantize_int4(arr: np.ndarray, group_sz: int = GROUP_SZ):
    """
    Quantize float32 matrix (rows, cols) to INT4 group-quantized format.
    Returns (nibbles_bytes, scales_bytes) as bytes objects.
    """
    rows, cols = arr.shape
    n_groups = (cols + group_sz - 1) // group_sz
    # Pad cols to multiple of group_sz
    pad = n_groups * group_sz - cols
    if pad:
        arr = np.pad(arr, ((0,0),(0,pad)), mode='constant')

    nibbles = bytearray()
    scales  = bytearray()

    for r in range(rows):
        for g in range(n_groups):
            block = arr[r, g*group_sz : (g+1)*group_sz].astype(np.float32)
            absmax = np.abs(block).max()
            if absmax < 1e-9:
                scale = 1e-9
            else:
                scale = absmax / 7.0  # map [-7,7] → INT4 [-8,7] with bias

            # Quantize to [-8, 7]
            q = np.clip(np.round(block / scale), -8, 7).astype(np.int8)
            # Pack pairs of nibbles: low=q[2i], high=q[2i+1]
            packed = bytearray(group_sz // 2)
            for i in range(0, group_sz, 2):
                lo = int(q[i])   & 0xF
                hi = int(q[i+1]) & 0xF
                packed[i//2] = (hi << 4) | lo
            nibbles.extend(packed)
            scales.extend(struct.pack('<f', scale))

    return bytes(nibbles), bytes(scales)


def write_int4_matrix(f, arr: np.ndarray):
    """Write INT4 quantized matrix: nibbles block then scales block."""
    assert arr.ndim == 2
    nib, sc = quantize_int4(arr)
    f.write(nib)
    f.write(sc)


def write_f32_vec(f, arr: np.ndarray):
    f.write(arr.astype(np.float32).tobytes())


def tensor_map(reader):
    """Build {name: tensor} dict."""
    return {t.name: t for t in reader.tensors}


def get_f32(tmap, name, shape=None) -> np.ndarray:
    t = tmap[name]
    arr = dequant_tensor(t)
    if shape:
        arr = arr.reshape(shape)
    return arr.astype(np.float32)


def main():
    print(f"Reading GGUF: {GGUF_PATH}")
    reader = gguf.GGUFReader(str(GGUF_PATH))
    tmap   = tensor_map(reader)

    print(f"Writing kernel binary: {OUT_PATH}")
    with open(OUT_PATH, 'wb') as f:
        # ── Header ───────────────────────────────────────────────────────────
        f.write(struct.pack('<10I',
            MAGIC, 1, N_LAYERS, D_MODEL, N_HEADS, N_KV, FFN_DIM, VOCAB, HEAD_DIM, GROUP_SZ))

        # ── Layers ───────────────────────────────────────────────────────────
        for li in range(N_LAYERS):
            print(f"  Layer {li}/{N_LAYERS-1}...", end='\r', flush=True)
            pfx = f"blk.{li}."

            # RMSNorm weights
            attn_norm = get_f32(tmap, pfx + "attn_norm.weight")
            write_f32_vec(f, attn_norm)

            # Q, K, V projections + biases
            for proj, out_dim in [('q', N_HEADS*HEAD_DIM), ('k', N_KV*HEAD_DIM), ('v', N_KV*HEAD_DIM)]:
                w = get_f32(tmap, pfx + f"attn_{proj}.weight").reshape(out_dim, D_MODEL)
                write_int4_matrix(f, w)
                b = get_f32(tmap, pfx + f"attn_{proj}.bias").reshape(out_dim)
                write_f32_vec(f, b)

            # Output projection
            o_w = get_f32(tmap, pfx + "attn_output.weight").reshape(D_MODEL, N_HEADS*HEAD_DIM)
            write_int4_matrix(f, o_w)

            # FFN RMSNorm
            ffn_norm = get_f32(tmap, pfx + "ffn_norm.weight")
            write_f32_vec(f, ffn_norm)

            # Gate, Up, Down projections
            gate_w = get_f32(tmap, pfx + "ffn_gate.weight").reshape(FFN_DIM, D_MODEL)
            write_int4_matrix(f, gate_w)
            up_w = get_f32(tmap, pfx + "ffn_up.weight").reshape(FFN_DIM, D_MODEL)
            write_int4_matrix(f, up_w)
            down_w = get_f32(tmap, pfx + "ffn_down.weight").reshape(D_MODEL, FFN_DIM)
            write_int4_matrix(f, down_w)

        print()

        # ── Output norm + LM head + embedding ────────────────────────────────
        print("  Writing output norm + LM head + embedding...")
        out_norm = get_f32(tmap, "output_norm.weight") if "output_norm.weight" in tmap \
                   else np.ones(D_MODEL, dtype=np.float32)
        write_f32_vec(f, out_norm)

        lm_head = get_f32(tmap, "output.weight").reshape(VOCAB, D_MODEL)
        write_int4_matrix(f, lm_head)

        emb = get_f32(tmap, "token_embd.weight").reshape(VOCAB, D_MODEL)
        write_int4_matrix(f, emb)

        # ── Tokenizer vocab ───────────────────────────────────────────────────
        print("  Writing tokenizer vocab...")
        tok_field  = reader.fields['tokenizer.ggml.tokens']
        n_vocab = len(tok_field.data)
        f.write(struct.pack('<I', n_vocab))
        for i in range(n_vocab):
            tok_bytes = bytes(reader.fields['tokenizer.ggml.tokens'].parts[tok_field.data[i]])
            tok_len   = min(len(tok_bytes), 65535)
            f.write(struct.pack('<H', tok_len))
            f.write(tok_bytes[:tok_len])

    size_mb = OUT_PATH.stat().st_size / 1024 / 1024
    print(f"\nDone. Output: {OUT_PATH} ({size_mb:.1f} MB)")
    print(f"EOS token: {EOS_TOKEN}  BOS/IM_START: {IM_START}  IM_END: {IM_END}")


if __name__ == '__main__':
    main()
