#!/usr/bin/env python3
"""Generate a tiny synthetic GPT-2 style model in .ecm format for smoke tests.

The model has random weights (no training) but a valid structure so the C
runtime can exercise prefill, generation, KV cache, and benchmarking.

Produces two artifacts:
  tiny-gpt2-int8.ecm   linear weights int8 (per-channel scales)
  tiny-gpt2-fp16.ecm   linear weights fp16 (true fp16 baseline)
"""
import argparse
import struct
from pathlib import Path

import numpy as np

from edgecore.ecm import EcmWriter, quantize_int8_symmetric


def make_vocab(vocab: int) -> bytes:
    """Byte-level vocab: token i maps to a single byte (0..vocab-1)."""
    blob = bytearray(struct.pack("<I", vocab))
    for i in range(vocab):
        b = bytes([i])
        blob += struct.pack("<II", i, len(b)) + b
    return bytes(blob)


def make_merges() -> bytes:
    blob = bytearray(struct.pack("<I", 0))
    return bytes(blob)


def build(precision: str, out: str, args) -> None:
    rng = np.random.default_rng(args.seed)
    L, H, E, C, V = args.n_layer, args.n_head, args.n_embd, args.n_ctx, args.vocab

    wr = EcmWriter()
    wr.set_meta(model_name=f"tiny-gpt2-{precision}", n_layer=L, n_head=H, n_embd=E,
                n_ctx=C, vocab=V, eps=1e-5)

    wr.add("wte", "fp16", rng.standard_normal((V, E)).astype(np.float32) * 0.02)
    wr.add("wpe", "fp16", rng.standard_normal((C, E)).astype(np.float32) * 0.02)
    wr.add("ln_f.g", "fp32", np.ones(E, dtype=np.float32))
    wr.add("ln_f.b", "fp32", np.zeros(E, dtype=np.float32))

    n_params = V * E + C * E + 2 * E
    for i in range(L):
        p = f"blocks.{i}"
        wr.add(f"{p}.ln_1.g", "fp32", np.ones(E, dtype=np.float32))
        wr.add(f"{p}.ln_1.b", "fp32", np.zeros(E, dtype=np.float32))
        wr.add(f"{p}.ln_2.g", "fp32", np.ones(E, dtype=np.float32))
        wr.add(f"{p}.ln_2.b", "fp32", np.zeros(E, dtype=np.float32))

        shapes = {
            "attn": (3 * E, E),
            "proj": (E, E),
            "fc": (4 * E, E),
            "fc2": (E, 4 * E),
        }
        for suffix, (outd, ind) in shapes.items():
            w = (rng.standard_normal((outd, ind)) * (1.0 / np.sqrt(ind))).astype(np.float32)
            if precision == "int8":
                q, s = quantize_int8_symmetric(w)
                wr.add(f"{p}.{suffix}.w", "int8", q)
                wr.add(f"{p}.{suffix}.s", "fp32", s)
            else:
                wr.add(f"{p}.{suffix}.w", "fp16", w)
            b = rng.standard_normal(outd).astype(np.float32) * 0.01
            wr.add(f"{p}.{suffix}.b", "fp32", b)
            n_params += w.size + b.size

    wr.add("tokenizer.vocab", "text", np.frombuffer(make_vocab(V), dtype=np.uint8))
    wr.add("tokenizer.merges", "text", np.frombuffer(make_merges(), dtype=np.uint8))
    wr.set_meta(n_params=n_params)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        wr.write(fh)
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes, {L} layers x{E} embd, vocab {V}, {precision})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a tiny synthetic GPT-2 .ecm model")
    ap.add_argument("--out", default="models/tiny-gpt2-int8.ecm")
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--n-head", type=int, default=2)
    ap.add_argument("--n-embd", type=int, default=32)
    ap.add_argument("--n-ctx", type=int, default=64)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--precision", default="both",
                    choices=["fp16", "int8", "both"],
                    help="which artifact(s) to write (default both)")
    args = ap.parse_args()

    if args.precision in ("both", "int8"):
        build("int8", args.out if args.precision == "int8" else "models/tiny-gpt2-int8.ecm", args)
    if args.precision in ("both", "fp16"):
        build("fp16", "models/tiny-gpt2-fp16.ecm", args)


if __name__ == "__main__":
    main()
