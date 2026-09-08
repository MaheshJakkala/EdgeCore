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
    return {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}.get(dtype, 0)


def analyze_ecm(path: str | Path, batch: int = 1) -> dict[str, Any]:
    path = Path(path)
    r = EcmReader(path)
    meta = r.meta
    from .ecm import arch_from_meta
    arch = arch_from_meta(meta)
    if arch == "qwen2":
        return _analyze_qwen2(r, meta, batch)
    return _analyze_gpt2(r, meta, batch)


def _base_report(r: EcmReader, meta: dict[str, Any], arch: str,
                 n_layer: int, n_head: int, n_embd: int, n_ctx: int,
                 vocab: int, n_params: int, batch: int,
                 head_dim: int, kv_per_seq_bytes: int, act_bytes: int,
                 weight_bytes: int, gemm_count: int, ln_count: int,
                 act_count: int, residual_count: int, flops_decode: int,
                 flops_prefill: int, params_emb: int, params_ln_total: int,
                 params_layers: int) -> dict[str, Any]:
    kv_cache_bytes = kv_per_seq_bytes * batch
    est_peak_mb = (weight_bytes + kv_cache_bytes + act_bytes) / (1024.0 ** 2)

    total_tensors = len(r.tensor_meta)
    bytes_by_dtype: dict[str, int] = {}
    text_bytes = 0
    for name, tm in r.tensor_meta.items():
        if tm["dtype"] == "text":
            text_bytes += int(tm["bytes"])
        else:
            bytes_by_dtype[tm["dtype"]] = bytes_by_dtype.get(tm["dtype"], 0) + int(tm["bytes"])

    return {
        "model_name": meta.get("model_name", "unknown"),
        "artifact": str(r.path),
        "file_bytes": r.path.stat().st_size,
        "architecture": arch,
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
            "recomputed_total": params_emb + params_ln_total + params_layers,
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
            "activation": act_count,
            "residual_add": residual_count,
            "attention_blocks": n_layer,
        },
        "flops": {
            "decode_per_token": flops_decode,
            "prefill_per_token": flops_prefill,
            "prefill_128_tokens": flops_prefill * 128,
        },
        "estimated": {
            "int8_speedup_hint": _int8_hint(weight_bytes, bytes_by_dtype),
        },
    }


def _analyze_gpt2(r: EcmReader, meta: dict[str, Any], batch: int) -> dict[str, Any]:
    n_layer = int(meta["n_layer"])
    n_head = int(meta["n_head"])
    n_embd = int(meta["n_embd"])
    n_ctx = int(meta["n_ctx"])
    vocab = int(meta["vocab"])
    n_params = int(meta.get("n_params", 0))
    head_dim = n_embd // n_head

    # ---- weight footprint per component ------------------------------------
    def block_bytes(prefix: str) -> int:
        return sum(int(tm["bytes"]) for nm, tm in r.tensor_meta.items()
                   if not tm["dtype"] == "text" and nm.startswith(prefix))

    emb_bytes = block_bytes("wte") + block_bytes("wpe")
    ln_bytes = block_bytes("ln_f")
    layer_bytes = 0
    for i in range(n_layer):
        layer_bytes += block_bytes(f"blocks.{i}")
    weight_bytes = emb_bytes + ln_bytes + layer_bytes

    # ---- KV cache & activation memory --------------------------------------
    kv_per_seq_bytes = 2 * n_layer * n_ctx * n_embd * 2      # K and V, fp16
    act_per_token = 4 * n_embd + 3 * n_embd + n_embd + n_ctx + 4 * n_embd + 4
    act_bytes = act_per_token * max(n_ctx, batch) * 4

    # ---- operator distribution ---------------------------------------------
    gemm_count = 4 * n_layer + 1                # qkv, proj, fc, fc2 per layer + lm_head
    ln_count = 2 * n_layer + 1

    # ---- FLOP estimates ------------------------------------------------------
    flops_qkv = 2 * n_embd * (3 * n_embd)
    flops_proj = 2 * n_embd * n_embd
    flops_fc = 2 * n_embd * (4 * n_embd)
    flops_fc2 = 2 * (4 * n_embd) * n_embd
    flops_attn = 4 * n_embd * n_ctx              # QK^T + PV over cached context
    flops_lm = 2 * n_embd * vocab
    flops_decode = n_layer * (flops_qkv + flops_proj + flops_fc +
                              flops_fc2 + flops_attn) + flops_lm
    flops_prefill = n_layer * (flops_qkv + flops_proj + flops_fc + flops_fc2) + flops_lm

    params_emb = vocab * n_embd + n_ctx * n_embd
    params_ln_total = (2 * n_layer + 2) * n_embd
    params_attn_per = (3 * n_embd + 1) * n_embd + (n_embd + 1) * n_embd
    params_mlp_per = (4 * n_embd + 1) * n_embd + (n_embd + 1) * (4 * n_embd)
    params_layers = n_layer * (params_attn_per + params_mlp_per)

    return _base_report(r, meta, "gpt2", n_layer, n_head, n_embd, n_ctx,
                        vocab, n_params, batch, head_dim,
                        kv_per_seq_bytes, act_bytes, weight_bytes,
                        gemm_count, ln_count, n_layer, 2 * n_layer,
                        flops_decode, flops_prefill, params_emb,
                        params_ln_total, params_layers)


def _analyze_qwen2(r: EcmReader, meta: dict[str, Any], batch: int) -> dict[str, Any]:
    n_layer = int(meta["n_layer"])
    n_head = int(meta["n_head"])
    n_embd = int(meta["n_embd"])
    n_ctx = int(meta["n_ctx"])
    vocab = int(meta["vocab"])
    n_params = int(meta.get("n_params", 0))
    n_kv_head = int(meta.get("n_kv_head", n_head))
    intermediate = int(meta.get("intermediate", 4 * n_embd))
    head_dim = n_embd // n_head
    n_kv_embd = n_kv_head * head_dim
    tied = bool(int(meta.get("tie_word_embeddings", 0)))

    # ---- weight footprint ---------------------------------------------------
    def block_bytes(prefix: str) -> int:
        return sum(int(tm["bytes"]) for nm, tm in r.tensor_meta.items()
                   if not tm["dtype"] == "text" and nm.startswith(prefix))

    emb_bytes = block_bytes("wte") + block_bytes("lm_head")
    ln_bytes = block_bytes("ln_f")
    layer_bytes = sum(block_bytes(f"blocks.{i}") for i in range(n_layer))
    weight_bytes = emb_bytes + ln_bytes + layer_bytes

    # ---- KV cache & activation memory --------------------------------------
    kv_per_seq_bytes = 2 * n_layer * n_ctx * n_kv_embd * 2   # GQA: KV is n_kv_embd
    # workspace sizing mirrors gpt2.c: x + tmp(4E) + qkv(max(E+2*kv_embd, 2*inter))
    #   + attn(E) + scores(C) + ws_q(max(E, inter)) + qs
    qkv_cols = max(2 * intermediate, n_embd + 2 * n_kv_embd)
    qcols = max(n_embd, intermediate)
    act_per_token = n_embd + 4 * n_embd + qkv_cols + n_embd + n_ctx + qcols + 1
    act_bytes = act_per_token * max(n_ctx, batch) * 4

    # ---- operators ----------------------------------------------------------
    gemm_count = 7 * n_layer + 1            # q,k,v,o,gate,up,down per layer + lm_head
    ln_count = 2 * n_layer + 1              # RMSNorm

    # ---- FLOPs ---------------------------------------------------------------
    flops_q = 2 * n_embd * n_embd
    flops_k = 2 * n_embd * n_kv_embd
    flops_v = 2 * n_embd * n_kv_embd
    flops_o = 2 * n_embd * n_embd
    flops_gate = 2 * n_embd * intermediate
    flops_up = 2 * n_embd * intermediate
    flops_down = 2 * intermediate * n_embd
    flops_attn = 4 * n_embd * n_ctx
    flops_lm = 2 * n_embd * vocab
    flops_decode = n_layer * (flops_q + flops_k + flops_v + flops_o +
                              flops_gate + flops_up + flops_down + flops_attn) + flops_lm
    flops_prefill = n_layer * (flops_q + flops_k + flops_v + flops_o +
                               flops_gate + flops_up + flops_down) + flops_lm

    # ---- params ---------------------------------------------------------------
    params_emb = vocab * n_embd + (0 if tied else vocab * n_embd)
    params_ln_total = (2 * n_layer + 1) * n_embd
    params_attn_per = n_embd * n_embd + 2 * n_kv_embd * n_embd + n_embd * n_embd
    params_mlp_per = 3 * intermediate * n_embd
    params_layers = n_layer * (params_attn_per + params_mlp_per)

    return _base_report(r, meta, "qwen2", n_layer, n_head, n_embd, n_ctx,
                        vocab, n_params, batch, head_dim,
                        kv_per_seq_bytes, act_bytes, weight_bytes,
                        gemm_count, ln_count, n_layer, 2 * n_layer,
                        flops_decode, flops_prefill, params_emb,
                        params_ln_total, params_layers)


def _detect_precision(r: EcmReader) -> str:
    from .ecm import first_linear_key
    w = r.tensor_meta.get(first_linear_key(r.meta))
    if w is None:
        return "unknown"
    return {"int8": "int8", "fp16": "fp16", "bf16": "bf16"}.get(w["dtype"], "fp32")


def _int8_hint(weight_bytes: int, by_dtype: dict[str, int]) -> float:
    fp32 = by_dtype.get("fp32", 0)
    fp16 = by_dtype.get("fp16", 0)
    bf16 = by_dtype.get("bf16", 0)
    int8 = by_dtype.get("int8", 0)
    if int8 == 0:
        return 1.0
    full = fp32 + fp16 + bf16 + int8
    half = (fp16 + bf16) / 2
    return round(full / max(1, (fp32 + half + int8)), 3) if full else 1.0


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
