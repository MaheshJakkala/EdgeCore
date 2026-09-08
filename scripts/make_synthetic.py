#!/usr/bin/env python3
"""Generate tiny synthetic models in .ecm format for smoke tests.

The models have random weights (no training) but a valid structure so the C
runtime can exercise prefill, generation, KV cache, and benchmarking.

gpt2  -> tiny-gpt2-int8.ecm / tiny-gpt2-fp16.ecm
qwen2 -> tiny-qwen2-int8.ecm / tiny-qwen2-fp16.ecm  (RMSNorm/RoPE/GQA/SwiGLU)
"""
import argparse
import struct
from pathlib import Path

import numpy as np

from edgecore.ecm import EcmWriter, quantize_int8_symmetric, f32_to_bf16


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
                n_ctx=C, vocab=V, eps=1e-5, arch="gpt2")

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


def build_qwen2(precision: str, out: str, args) -> None:
    rng = np.random.default_rng(args.seed)
    L, H, E, C, V = args.n_layer, args.n_head, args.n_embd, args.n_ctx, args.vocab
    KVH = args.n_kv_head
    inter = args.intermediate
    hd = E // H
    kv_embd = KVH * hd
    rope_theta = 1_000_000.0

    wr = EcmWriter()
    wr.set_meta(model_name=f"tiny-qwen2-{precision}", n_layer=L, n_head=H,
                n_kv_head=KVH, n_embd=E, intermediate=inter, n_ctx=C, vocab=V,
                eps=1e-6, rope_theta=rope_theta, rope_type="default",
                yarn_factor=1.0, yarn_orig_max=C, yarn_attention_factor=1.0,
                sliding_window=0, tie_word_embeddings=0, eos_token_id=V - 1,
                arch="qwen2")

    def emb(arr: np.ndarray) -> np.ndarray:
        if precision == "bf16":
            return f32_to_bf16(arr)
        return arr.astype(np.float32)

    wr.add("wte", "fp16" if precision != "bf16" else "bf16", emb(
        rng.standard_normal((V, E)).astype(np.float32) * 0.02))
    wr.add("lm_head.w", "fp16" if precision != "bf16" else "bf16", emb(
        rng.standard_normal((V, E)).astype(np.float32) * 0.02))
    wr.add("ln_f.g", "fp32", np.ones(E, dtype=np.float32))

    n_params = V * E * 2 + E

    def add_lin(p: str, outd: int, ind: int) -> None:
        w = (rng.standard_normal((outd, ind)) * (1.0 / np.sqrt(ind))).astype(np.float32)
        if precision == "int8":
            q, s = quantize_int8_symmetric(w)
            wr.add(f"{p}.w", "int8", q)
            wr.add(f"{p}.s", "fp32", s)
        elif precision == "bf16":
            wr.add(f"{p}.w", "bf16", f32_to_bf16(w))
        else:
            wr.add(f"{p}.w", "fp16", w)
        nonlocal n_params
        n_params += w.size

    for i in range(L):
        p = f"blocks.{i}"
        wr.add(f"{p}.ln_1.g", "fp32", np.ones(E, dtype=np.float32))
        wr.add(f"{p}.ln_2.g", "fp32", np.ones(E, dtype=np.float32))
        for suffix, (outd, ind) in {
            "q": (E, E), "k": (kv_embd, E), "v": (kv_embd, E), "o": (E, E),
            "gate": (inter, E), "up": (inter, E), "down": (E, inter),
        }.items():
            add_lin(f"{p}.{suffix}", outd, ind)

    wr.add("tokenizer.vocab", "text", np.frombuffer(make_vocab(V), dtype=np.uint8))
    wr.add("tokenizer.merges", "text", np.frombuffer(make_merges(), dtype=np.uint8))
    wr.set_meta(n_params=n_params)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        wr.write(fh)
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes, {L} layers x{E} embd "
          f"H{H} KVH{KVH} inter{inter}, vocab {V}, {precision})")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Generate a tiny synthetic GPT-2 or Qwen2 .ecm model")
    ap.add_argument("--arch", default="both", choices=["gpt2", "qwen2", "both"],
                    help="architecture to generate (default both)")
    ap.add_argument("--out", default="models/tiny-gpt2-int8.ecm")
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--n-head", type=int, default=2)
    ap.add_argument("--n-kv-head", type=int, default=1,
                    help="Qwen2 KV heads per layer (GQA); default 1")
    ap.add_argument("--n-embd", type=int, default=32)
    ap.add_argument("--intermediate", type=int, default=48,
                    help="Qwen2 MLP intermediate size")
    ap.add_argument("--n-ctx", type=int, default=64)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--precision", default="both",
                    choices=["fp16", "bf16", "int8", "both", "all"],
                    help="which artifact(s) to write (default both = fp16,int8; all = fp16,bf16,int8)")
    args = ap.parse_args(argv)

    if args.precision == "all":
        precs = ["fp16", "bf16", "int8"]
    else:
        precs = ["int8", "fp16"] if args.precision == "both" else [args.precision]
    for prec in precs:
        if args.arch in ("gpt2", "both") and prec == "bf16":
            continue  # gpt2 synthetic stays fp16/int8 (fp32 source, no bf16 benefit)
        if args.arch in ("gpt2", "both"):
            build(prec, f"models/tiny-gpt2-{prec}.ecm", args)
        if args.arch in ("qwen2", "both"):
            build_qwen2(prec, f"models/tiny-qwen2-{prec}.ecm", args)


if __name__ == "__main__":
    main()

