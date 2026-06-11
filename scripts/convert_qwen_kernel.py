#!/usr/bin/env python3
"""Convert Qwen2.5-0.5B-Instruct GGUF to NodeAI kernel flat INT4 binary.
Memory-efficient: tokenizer first, then heaviest tensors one at a time."""
import sys, struct
import numpy as np
from pathlib import Path
import gguf

SCRIPT_DIR = Path(__file__).parent.resolve()
GGUF_PATH = SCRIPT_DIR.parent.parent.parent.parent / "models" / "qwen" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
OUT_PATH  = SCRIPT_DIR.parent / "models" / "qwen25_kernel.bin"

GROUP_SZ=32; MAGIC=0x4E574B52
NL=24; DM=896; NH=14; NK=2; FF=4864; VOC=151936; HD=64

def wf32(f, a): f.write(a.astype(np.float32).tobytes())

def wint4(f, arr):
    rows, cols = arr.shape
    ng = (cols + GROUP_SZ - 1) // GROUP_SZ
    arr = arr.astype(np.float32)
    for r in range(rows):
        for g in range(ng):
            start = g * GROUP_SZ
            blk = arr[r, start:start + GROUP_SZ]
            amax = float(np.abs(blk).max())
            sc = amax / 7.0 if amax > 1e-10 else 1.0
            q = np.clip(np.round(blk / sc), -8, 7).astype(np.int8)
            pk = bytearray(16)
            for k in range(32):
                v = q[k] & 0x0F
                if k % 2 == 0: pk[k//2] = v
                else: pk[k//2] |= (v << 4)
            f.write(bytes(pk))
            f.write(struct.pack('<f', sc))

def tmap(r): return {t.name: t for t in r.tensors}

def gf32(tm, n):
    t = tm[n]
    if t.tensor_type.name == 'F32':
        return np.frombuffer(t.data.tobytes(), dtype=np.float32)
    from gguf.quants import dequantize
    return dequantize(t.data, t.tensor_type).astype(np.float32).ravel()

def main():
    print(f"Output: {OUT_PATH}")
    # Phase 1: extract tokenizer via lightweight fields API
    r = gguf.GGUFReader(str(GGUF_PATH))
    print("  Extracting tokenizer...", end=' ')
    tf = r.fields['tokenizer.ggml.tokens']
    tokens = [bytes(r.fields['tokenizer.ggml.tokens'].parts[tf.data[i]]) for i in range(len(tf.data))]
    print(f"{len(tokens)} tokens")
    del r

    with open(OUT_PATH, 'wb') as f:
        f.write(struct.pack('<10I', MAGIC, 1, NL, DM, NH, NK, FF, VOC, HD, GROUP_SZ))
        f.write(struct.pack('<I', len(tokens)))
        for t in tokens:
            f.write(struct.pack('<H', len(t)))
            f.write(t)
        del tokens
        print(f"  Tokenizer at offset {f.tell()}")

        r = gguf.GGUFReader(str(GGUF_PATH))
        tm = tmap(r)
        for li in range(NL):
            print(f"  Layer {li}/{NL-1}...", end='\r', flush=True)
            pfx = f"blk.{li}."
            wf32(f, gf32(tm, pfx+"attn_norm.weight"))
            for proj, od in [('q', NH*HD), ('k', NK*HD), ('v', NK*HD)]:
                wint4(f, gf32(tm, pfx+f"attn_{proj}.weight").reshape(od, DM))
                wf32(f, gf32(tm, pfx+f"attn_{proj}.bias").reshape(od))
            wint4(f, gf32(tm, pfx+"attn_output.weight").reshape(DM, NH*HD))
            wf32(f, gf32(tm, pfx+"ffn_norm.weight"))
            wint4(f, gf32(tm, pfx+"ffn_gate.weight").reshape(FF, DM))
            wint4(f, gf32(tm, pfx+"ffn_up.weight").reshape(FF, DM))
            wint4(f, gf32(tm, pfx+"ffn_down.weight").reshape(DM, FF))
        print(f"\n  Layers done at offset {f.tell()}")

        wf32(f, gf32(tm,"output_norm.weight") if "output_norm.weight" in tm else np.ones(DM,dtype=np.float32))
        print("  Writing lm_head...", end=' ')
        lh = gf32(tm, "output.weight").reshape(VOC, DM)
        wint4(f, lh)
        del lh, r, tm
        print(f"OK (offset {f.tell()})")

        r2 = gguf.GGUFReader(str(GGUF_PATH))
        tm2 = tmap(r2)
        print("  Writing embedding...", end=' ')
        emb = gf32(tm2, "token_embd.weight").reshape(VOC, DM)
        wint4(f, emb)
        del emb, r2, tm2
        print(f"OK (offset {f.tell()})")

    sz = OUT_PATH.stat().st_size
    print(f"\nDone: {sz} bytes ({sz/1024/1024:.0f}MB)")

if __name__ == '__main__':
    main()
