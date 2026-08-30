"""EdgeCore - model analyzer.

Step 7: extracts characteristics of an ``.ecm`` model that influence
inference optimization and deployment, including parameter counts,
architecture dimensions, tensor layout, weight footprint by dtype,
KV-cache requirements, estimated memory footprint, operator distribution,
and FLOP estimates. The output feeds the search-space construction in
``bench.py`` and is recorded in the deployment manifest by
``package.py``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from .ecm import EcmReader


def dtype_bytes(dtype: str) -> int:
    return {"fp32": 4, "fp16": 2, "int8": 1}.get(dtype, 0)


def analyze_ecm(path: str | Path, batch: int = 1) -> dict[str, Any]:
    path = Path(path)
    r = EcmReader(path)
    meta = r.meta
    n_layer = int(meta["n_layer"])
    n_head = int(meta["n_head"])
    n_embd = int(meta["n_embd"])
    n_ctx = int(meta["n_ctx"])
    vocab = int(meta["vocab"])
    n_params = int(meta.get("n_params", 0))
    head_dim = n_embd // n_head

    # ---- tensor inventory -------------------------------------------------
    total_tensors = len(r.tensor_meta)
    bytes_by_dtype: dict[str, int] = {}
    text_bytes = 0
    for name, tm in r.tensor_meta.items():
        if tm["dtype"] == "text":
            text_bytes += int(tm["bytes"])
        else:
            bytes_by_dtype[tm["dtype"]] = bytes_by_dtype.get(tm["dtype"], 0) + int(tm["bytes"])

    # ---- weight footprint per component ------------------------------------
    def block_bytes(prefix: str) -> int:
        return sum(int(tm["bytes"]) for nm, tm in r.tensor_meta.items()
                   if not tm["dtype"] == "text" and nm.startswith(prefix))

    emb_bytes = block_bytes("wte") + block_bytes("wpe")
    ln_bytes = block_bytes("ln_f")
    layer_bytes = 0
    attn_bytes = mlp_bytes = 0
    for i in range(n_layer):
        p = f"blocks.{i}"
        layer_bytes += block_bytes(p)
        attn_bytes += block_bytes(f"{p}.attn") + block_bytes(f"{p}.proj")
        mlp_bytes += block_bytes(f"{p}.fc") + block_bytes(f"{p}.fc2")
    weight_bytes = emb_bytes + ln_bytes + layer_bytes

    # ---- KV cache & activation memory --------------------------------------
    kv_per_seq_bytes = 2 * n_layer * n_ctx * n_embd * 2      # K and V, fp16
    kv_cache_bytes = kv_per_seq_bytes * batch
    # activation workspace: x, tmp, qkv, attn, scores + int8 q (4x) + qs
    act_per_token = 4 * n_embd + 3 * n_embd + n_embd + n_ctx + 4 * n_embd + 4
    act_bytes = act_per_token * max(n_ctx, batch) * 4
    est_peak_mb = (weight_bytes + kv_cache_bytes + act_bytes) / (1024.0 ** 2)

    # ---- operator distribution ---------------------------------------------
    gemm_count = 4 * n_layer + 1                # qkv, proj, fc, fc2 per layer + lm_head
    ln_count = 2 * n_layer + 1
    gelu_count = n_layer
    residual_count = 2 * n_layer
    attention_blocks = n_layer                  # QK^T + softmax + PV per layer

    # ---- FLOP estimates ------------------------------------------------------
    # decode: per token, per layer GEMMs ~ 2 * (params in matmuls for that token)
    #   qkv 2*E*3E, proj 2*E*E, fc 2*E*4E, fc2 2*4E*E = 2*E*E*(3+1+4+4) = 24E^2
    #   plus lm_head 2*E*V
    # attention QK^T+PV: 2 * 2 * n_ctx * E  (rough per-token) -> folded into decode
    flops_qkv = 2 * n_embd * (3 * n_embd)
    flops_proj = 2 * n_embd * n_embd
    flops_fc = 2 * n_embd * (4 * n_embd)
    flops_fc2 = 2 * (4 * n_embd) * n_embd
    flops_attn = 4 * n_embd * n_ctx              # QK^T + PV over cached context
    flops_lm = 2 * n_embd * vocab
    flops_decode_per_token = n_layer * (flops_qkv + flops_proj + flops_fc +
                                        flops_fc2 + flops_attn) + flops_lm
    flops_prefill_per_token = n_layer * (flops_qkv + flops_proj + flops_fc +
                                         flops_fc2) + flops_lm

    # ---- parameter breakdown --------------------------------------------------
    params_emb = vocab * n_embd + n_ctx * n_embd
    params_ln_total = (2 * n_layer + 2) * n_embd
    params_attn_per = (3 * n_embd + 1) * n_embd + (n_embd + 1) * n_embd
    params_mlp_per = (4 * n_embd + 1) * n_embd + (n_embd + 1) * (4 * n_embd)
    params_layers = n_layer * (params_attn_per + params_mlp_per)
    params_recomputed = params_emb + params_ln_total + params_layers

    return {
        "model_name": meta.get("model_name", "unknown"),
        "artifact": str(path),
        "file_bytes": r.path.stat().st_size,
        "architecture": "gpt2",
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "head_dim": head_dim,
        "n_ctx": n_ctx,
        "vocab": vocab,
        "eps": float(meta.get("eps", 1e-5)),
        "precision": _detect_precision(r),
        "tensor_count": total_tensors,
        "text_blob_bytes": text_bytes,
        "bytes_by_dtype": {k: v for k, v in sorted(bytes_by_dtype.items())},
        "weight_bytes": weight_bytes,
        "weight_mb": weight_bytes / (1024.0 ** 2),
        "params_total": n_params,
        "params_breakdown": {
            "embeddings": params_emb,
            "layernorms": params_ln_total,
            "transformer_layers": params_layers,
            "recomputed_total": params_recomputed,
        },
        "memory": {
            "kv_cache_bytes": kv_cache_bytes,
            "kv_cache_mb": kv_cache_bytes / (1024.0 ** 2),
            "kv_cache_mb_per_seq": kv_per_seq_bytes / (1024.0 ** 2),
            "activation_ws_bytes": act_bytes,
            "estimated_peak_mb": est_peak_mb,
        },
        "operators": {
            "gemm": gemm_count,
            "layernorm": ln_count,
            "gelu": gelu_count,
            "residual_add": residual_count,
            "attention_blocks": attention_blocks,
        },
        "flops": {
            "decode_per_token": flops_decode_per_token,
            "prefill_per_token": flops_prefill_per_token,
            "prefill_128_tokens": flops_prefill_per_token * 128,
        },
        "estimated": {
            "int8_speedup_hint": _int8_hint(weight_bytes, bytes_by_dtype),
        },
    }


def _detect_precision(r: EcmReader) -> str:
    w = r.tensor_meta.get("blocks.0.attn.w")
    if w is None:
        return "unknown"
    return "int8" if w["dtype"] == "int8" else ("fp16" if w["dtype"] == "fp16" else "fp32")


def _int8_hint(weight_bytes: int, by_dtype: dict[str, int]) -> float:
    fp32 = by_dtype.get("fp32", 0)
    fp16 = by_dtype.get("fp16", 0)
    int8 = by_dtype.get("int8", 0)
    if int8 == 0:
        return 1.0
    full = fp32 + fp16 + int8
    return round(full / max(1, (fp32 + fp16 / 2 + int8)), 3) if full else 1.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Analyze an .ecm model", prog="python -m edgecore.analyze")
    ap.add_argument("--model", required=True, help="path to .ecm model")
    ap.add_argument("--batch", type=int, default=1, help="deployment batch size (KV sizing)")
    ap.add_argument("--output", default=None, help="write JSON report to file")
    args = ap.parse_args(argv)

    rep = analyze_ecm(args.model, batch=args.batch)
    text = json.dumps(rep, indent=2)
    if args.output:
        Path(args.output).write_text(text)
        print(f"analysis written to {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
