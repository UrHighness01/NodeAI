#!/usr/bin/env python3
"""
Convert Qwen3.5 0.6B obliterated GGUF → NodeAI kernel binary (lm_qwen35.bin)

Architecture (hardcoded, verified from GGUF):
  n_layers=24, n_embd=1024, n_head=8, n_kv=2, n_ff=3584, vocab=248320
  head_dim=256, d_inner=2048, d_state=128, n_group=16, dt_rank=16
  conv_dim=6144, full_attn_interval=4 (layers 3,7,11,15,19,23 are full attention)

Binary layout:
  [64B header]
  [24 layer blocks — SSM or attention based on il%4==3]
  [output_norm f32: 1024×4]
  [token_embd Q8_0: 248320 rows × 32 blocks × 34 bytes]
  [tokenizer: u32 vocab_size, then for each: u16 len + bytes]

SSM layer block:
  attn_norm f32 [1024×4]
  post_attn_norm f32 [1024×4]
  ssm_a f32 [16×4]
  ssm_dt_bias f32 [16×4]
  ssm_norm f32 [128×4]
  wqkv Q8_0: rows=6144, cols=1024
  attn_gate Q8_0: rows=2048, cols=1024
  ssm_beta Q8_0: rows=16, cols=1024
  ssm_alpha Q8_0: rows=16, cols=1024
  ssm_out Q8_0: rows=1024, cols=2048
  ffn_gate Q8_0: rows=3584, cols=1024
  ffn_up Q8_0: rows=3584, cols=1024
  ffn_down Q8_0: rows=1024, cols=3584
  ssm_conv1d f32 [4×6144×4] (layout: [channel][tap], i.e. channel-major)

Attention layer block:
  attn_norm f32 [1024×4]
  post_attn_norm f32 [1024×4]
  q_norm f32 [256×4]
  k_norm f32 [256×4]
  wq Q8_0: rows=4096, cols=1024  (Q+gate concatenated per head)
  wk Q8_0: rows=512, cols=1024
  wv Q8_0: rows=512, cols=1024
  wo Q8_0: rows=1024, cols=2048
  ffn_gate Q8_0: rows=3584, cols=1024
  ffn_up Q8_0: rows=3584, cols=1024
  ffn_down Q8_0: rows=1024, cols=3584
"""

import sys, os, struct
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GGUF_PATH = os.path.join(SCRIPT_DIR, "..", "..", "..", "models", "qwen_obliterated", "qwen35_0.6b_obliterated.gguf")
OUT_PATH   = os.path.join(SCRIPT_DIR, "..", "models", "lm_qwen35.bin")

GS      = 32
BSIZE   = 34   # Q8_0 block bytes: 2(f16) + 32(i8)
MAGIC   = 0x4E574B53
N_LAYERS = 24
N_EMBD  = 1024
N_HEAD  = 8
N_KV    = 2
N_FF    = 3584
VOCAB   = 248320
HD      = 256   # head_dim
FA_INT  = 4     # full attention interval
D_INNER = 2048
D_STATE = 128
N_GROUP = 16
DT_RANK = 16
CONV_DIM = D_STATE * N_GROUP * 2 + D_INNER  # 6144
ROPE_PAIRS = 32
ROPE_THETA_BITS = struct.unpack('<I', struct.pack('<f', 1e7))[0]


def is_attn(il):
    return (il + 1) % FA_INT == 0


def q8_sz(rows, cols):
    assert cols % GS == 0, f"cols={cols} not divisible by {GS}"
    return rows * (cols // GS) * BSIZE


def get_tensor_raw(tensors, name):
    """Return raw bytes for tensor (must be Q8_0 or F32)."""
    t = tensors.get(name)
    if t is None:
        raise KeyError(f"Tensor not found: {name}")
    return bytes(t.data.tobytes())


def get_f32(tensors, name, expected_n):
    """Return raw F32 bytes, dequantizing if needed."""
    t = tensors.get(name)
    if t is None:
        raise KeyError(f"Tensor not found: {name}")
    tname = t.tensor_type.name
    if tname == 'F32':
        arr = np.frombuffer(t.data.tobytes(), dtype=np.float32)
    else:
        from gguf.quants import dequantize
        arr = dequantize(t.data, t.tensor_type).astype(np.float32).ravel()
    assert len(arr) == expected_n, f"{name}: expected {expected_n} got {len(arr)}"
    return arr.tobytes()


def requantize_q8_0(tensors, name, expected_rows, expected_cols):
    """
    Return Q8_0 bytes for a tensor with shape [expected_cols, expected_rows] (GGUF convention).
    If tensor is already Q8_0, copy raw bytes. Otherwise, dequantize then re-quantize.
    """
    t = tensors.get(name)
    if t is None:
        raise KeyError(f"Tensor not found: {name}")
    tname = t.tensor_type.name

    if tname == 'Q8_0':
        raw = bytes(t.data.tobytes())
        expected_sz = q8_sz(expected_rows, expected_cols)
        assert len(raw) == expected_sz, (
            f"{name} Q8_0 raw={len(raw)} expected={expected_sz} "
            f"(rows={expected_rows}, cols={expected_cols})"
        )
        return raw

    # Dequantize
    from gguf.quants import dequantize
    arr = dequantize(t.data, t.tensor_type).astype(np.float32)
    arr = arr.ravel()
    # GGUF shape [cols, rows] (innermost=cols) → we have rows×cols elements
    assert len(arr) == expected_rows * expected_cols, (
        f"{name}: dequant={len(arr)} expected={expected_rows*expected_cols}"
    )
    arr = arr.reshape(expected_rows, expected_cols)
    # Re-quantize to Q8_0
    out = bytearray()
    n_blocks = expected_cols // GS
    for r in range(expected_rows):
        for b in range(n_blocks):
            blk = arr[r, b*GS:(b+1)*GS]
            amax = np.abs(blk).max()
            scale = amax / 127.0 if amax > 0 else 1.0
            scale_f16 = float_to_f16_bytes(scale)
            quant = np.clip(np.round(blk / scale), -127, 127).astype(np.int8)
            out += scale_f16 + bytes(quant.tobytes())
    return bytes(out)


def float_to_f16_bytes(f):
    """Convert float32 to f16 bytes (2 bytes LE)."""
    arr = np.array([f], dtype=np.float32)
    f16 = arr.astype(np.float16)
    return f16.tobytes()


def main():
    print(f"Loading GGUF: {GGUF_PATH}")
    from gguf import GGUFReader
    r = GGUFReader(GGUF_PATH)

    # Build tensor lookup
    tensors = {t.name: t for t in r.tensors}
    print(f"  {len(tensors)} tensors found")

    # Verify key tensors exist
    for name in ['token_embd.weight', 'output_norm.weight', 'blk.0.attn_norm.weight']:
        if name not in tensors:
            print(f"ERROR: missing tensor {name}")
            sys.exit(1)

    print(f"Writing: {OUT_PATH}")
    with open(OUT_PATH, 'wb') as f:
        written = 0

        # ── Header (64 bytes = 16 × u32) ──────────────────────────────────
        header = struct.pack('<16I',
            MAGIC, 1, N_LAYERS, N_EMBD, N_HEAD, N_KV, N_FF, VOCAB,
            FA_INT, D_INNER, D_STATE, N_GROUP, DT_RANK, HD,
            ROPE_THETA_BITS, ROPE_PAIRS
        )
        f.write(header)
        written += len(header)
        print(f"  Header: {written} bytes")

        # ── Layer blocks ───────────────────────────────────────────────────
        for il in range(N_LAYERS):
            block_start = written
            attn = is_attn(il)
            print(f"  Layer {il:2d} ({'attn' if attn else 'ssm '})", end='', flush=True)

            if not attn:
                # SSM layer
                f.write(get_f32(tensors, f'blk.{il}.attn_norm.weight', N_EMBD))
                f.write(get_f32(tensors, f'blk.{il}.post_attention_norm.weight', N_EMBD))
                f.write(get_f32(tensors, f'blk.{il}.ssm_a', N_GROUP))
                f.write(get_f32(tensors, f'blk.{il}.ssm_dt.bias', N_GROUP))
                f.write(get_f32(tensors, f'blk.{il}.ssm_norm.weight', D_STATE))
                # Q8_0 weight matrices
                # wqkv: GGUF shape [1024, 6144] → rows=6144, cols=1024
                f.write(requantize_q8_0(tensors, f'blk.{il}.attn_qkv.weight', CONV_DIM, N_EMBD))
                # attn_gate: GGUF [1024, 2048] → rows=2048, cols=1024
                f.write(requantize_q8_0(tensors, f'blk.{il}.attn_gate.weight', D_INNER, N_EMBD))
                # ssm_beta: GGUF [1024, 16] → rows=16, cols=1024
                f.write(requantize_q8_0(tensors, f'blk.{il}.ssm_beta.weight', N_GROUP, N_EMBD))
                # ssm_alpha: same shape
                f.write(requantize_q8_0(tensors, f'blk.{il}.ssm_alpha.weight', N_GROUP, N_EMBD))
                # ssm_out: GGUF [2048, 1024] → rows=1024, cols=2048
                f.write(requantize_q8_0(tensors, f'blk.{il}.ssm_out.weight', N_EMBD, D_INNER))
                # ffn_gate: GGUF [1024, 3584] → rows=3584, cols=1024
                f.write(requantize_q8_0(tensors, f'blk.{il}.ffn_gate.weight', N_FF, N_EMBD))
                # ffn_up: same
                f.write(requantize_q8_0(tensors, f'blk.{il}.ffn_up.weight', N_FF, N_EMBD))
                # ffn_down: GGUF [3584, 1024] → rows=1024, cols=3584
                f.write(requantize_q8_0(tensors, f'blk.{il}.ffn_down.weight', N_EMBD, N_FF))
                # ssm_conv1d: F32 [4, 6144] in GGUF (shape=[4, CONV_DIM])
                # We store as [CONV_DIM × 4] f32 (channel-major) for efficient access
                conv_raw = get_f32(tensors, f'blk.{il}.ssm_conv1d.weight', 4 * CONV_DIM)
                # GGUF layout: 6144 channels × 4 taps (shape=[4,6144] innermost=4)
                # → data[j*4+t] = weight for channel j, tap t
                # This is already channel-major! Just write as-is.
                f.write(conv_raw)

            else:
                # Attention layer
                f.write(get_f32(tensors, f'blk.{il}.attn_norm.weight', N_EMBD))
                f.write(get_f32(tensors, f'blk.{il}.post_attention_norm.weight', N_EMBD))
                f.write(get_f32(tensors, f'blk.{il}.attn_q_norm.weight', HD))
                f.write(get_f32(tensors, f'blk.{il}.attn_k_norm.weight', HD))
                # wq: GGUF [1024, 4096] → rows=4096, cols=1024
                f.write(requantize_q8_0(tensors, f'blk.{il}.attn_q.weight', N_HEAD*HD*2, N_EMBD))
                # wk: GGUF [1024, 512] → rows=512, cols=1024
                f.write(requantize_q8_0(tensors, f'blk.{il}.attn_k.weight', N_KV*HD, N_EMBD))
                # wv: same
                f.write(requantize_q8_0(tensors, f'blk.{il}.attn_v.weight', N_KV*HD, N_EMBD))
                # wo: GGUF [2048, 1024] → rows=1024, cols=2048
                f.write(requantize_q8_0(tensors, f'blk.{il}.attn_output.weight', N_EMBD, N_HEAD*HD))
                # FFN
                f.write(requantize_q8_0(tensors, f'blk.{il}.ffn_gate.weight', N_FF, N_EMBD))
                f.write(requantize_q8_0(tensors, f'blk.{il}.ffn_up.weight', N_FF, N_EMBD))
                f.write(requantize_q8_0(tensors, f'blk.{il}.ffn_down.weight', N_EMBD, N_FF))

            block_sz = written - block_start + f.tell() - written
            # Get actual size written
            pos_now = f.tell()
            written = pos_now
            print(f" {(pos_now - block_start) // 1024}KB  total={written // 1048576}MB")

        # ── Global tensors ─────────────────────────────────────────────────
        print("  Writing output_norm...", flush=True)
        f.write(get_f32(tensors, 'output_norm.weight', N_EMBD))
        written = f.tell()

        print(f"  Writing token_embd ({VOCAB} × {N_EMBD//GS} blocks)...", flush=True)
        # token_embd: GGUF [1024, 248320] → rows=248320, cols=1024
        f.write(requantize_q8_0(tensors, 'token_embd.weight', VOCAB, N_EMBD))
        written = f.tell()
        print(f"  After embeddings: {written // 1048576}MB")

        # ── Tokenizer ──────────────────────────────────────────────────────
        print("  Writing tokenizer...", flush=True)
        vocab_entries = extract_tokenizer(r)
        f.write(struct.pack('<I', len(vocab_entries)))
        for entry in vocab_entries:
            b = entry.encode('utf-8') if isinstance(entry, str) else entry
            f.write(struct.pack('<H', len(b)))
            f.write(b)
        written = f.tell()
        print(f"  Final size: {written // 1048576}MB")

    print(f"Done: {OUT_PATH} ({os.path.getsize(OUT_PATH) // 1048576}MB)")


def extract_tokenizer(reader):
    """Extract vocab tokens as byte sequences."""
    from gguf import GGUFValueType
    tokens_field = reader.fields.get('tokenizer.ggml.tokens')
    if tokens_field is None:
        print("WARNING: no tokenizer.ggml.tokens field")
        return []

    # GGUFReader stores array fields as a flat data array with part indices
    # Reconstruct token strings from parts
    entries = []
    data = tokens_field.parts
    indices = tokens_field.data  # array of indices into parts

    for idx in indices:
        part = data[idx]
        try:
            text = bytes(part.tobytes())
            entries.append(text)
        except Exception:
            entries.append(b'')

    print(f"  Extracted {len(entries)} vocab entries")
    return entries


if __name__ == '__main__':
    main()
