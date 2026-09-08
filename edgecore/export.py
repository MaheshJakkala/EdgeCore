"""EdgeCore - export HF GPT-2 safetensors checkpoints to .ecm artifacts.

Step 5: produces two artifact flavors from one HF model directory:

  <name>-fp16.ecm      linear weights stored as fp16  (native fp16 baseline)
  <name>-int8-per.ecm  linear weights stored as per-output-channel int8
                       with fp32 scales (per-channel quantization)

Embeddings (wte/wpe) are always fp16; layernorm gammas/betas and biases are
always fp32. The C runtime derives its precision from ``blocks.0.attn.w``
dtype, so no explicit precision flag is needed at load time.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .ecm import EcmWriter, _load_safetensors, quantize_int8_symmetric

SUPPORTED_LINEARS = ("attn", "proj", "fc", "fc2")


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-_") or "model"


def load_hf_config(src: Path) -> dict[str, Any]:
    cfg = json.loads((src / "config.json").read_text())
    return {
        "model_name": cfg.get("_name_or_path", "gpt2"),
        "n_layer": cfg["n_layer"],
        "n_head": cfg["n_head"],
        "n_embd": cfg["n_embd"],
        "n_ctx": cfg.get("n_positions", cfg.get("n_ctx", 1024)),
        "vocab": cfg["vocab_size"],
        "eps": cfg.get("layer_norm_epsilon", 1e-5),
    }


def build_tokenizer_blobs(src: Path) -> tuple[np.ndarray, np.ndarray]:
    """Serialize vocab.json + merges.txt into the .ecm tokenizer blob layout."""
    import struct

    vocab_json = json.loads((src / "vocab.json").read_text())
    merges_txt = (src / "merges.txt").read_text().splitlines()

    vocab_blob = bytearray(struct.pack("<I", len(vocab_json)))
    for tok, tid in vocab_json.items():
        b = tok.encode("utf-8")
        vocab_blob += struct.pack("<II", tid, len(b)) + b

    merges_blob = bytearray(struct.pack("<I", len(merges_txt)))
    for line in merges_txt:
        if not line.strip():
            continue
        left, right = line.split(" ", 1)
        la, lb = left.encode("utf-8"), right.encode("utf-8")
        merges_blob += struct.pack("<I", len(la)) + la + struct.pack("<I", len(lb)) + lb

    return (
        np.frombuffer(bytes(vocab_blob), dtype=np.uint8),
        np.frombuffer(bytes(merges_blob), dtype=np.uint8),
    )


def build_shared_tensors(src: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    """Load config, safetensors and tokenizer data shared by both artifacts."""
    cfg = load_hf_config(src)
    tensors = _load_safetensors(src / "model.safetensors")
    vocab_blob, merges_blob = build_tokenizer_blobs(src)
    return cfg, tensors, {"vocab": vocab_blob, "merges": merges_blob}


def build_layer_weights(tensors: dict[str, np.ndarray], i: int) -> dict[str, np.ndarray]:
    """Return per-layer weight arrays already transposed to [out, in] layout."""
    p = f"h.{i}"
    attn_w = tensors[f"{p}.attn.c_attn.weight"].T.copy()
    attn_b = tensors[f"{p}.attn.c_attn.bias"]
    proj_w = tensors[f"{p}.attn.c_proj.weight"].T.copy()
    proj_b = tensors[f"{p}.attn.c_proj.bias"]
    fc_w = tensors[f"{p}.mlp.c_fc.weight"].T.copy()
    fc_b = tensors[f"{p}.mlp.c_fc.bias"]
    fc2_w = tensors[f"{p}.mlp.c_proj.weight"].T.copy()
    fc2_b = tensors[f"{p}.mlp.c_proj.bias"]
    return {
        "ln_1.g": tensors[f"{p}.ln_1.weight"],
        "ln_1.b": tensors[f"{p}.ln_1.bias"],
        "ln_2.g": tensors[f"{p}.ln_2.weight"],
        "ln_2.b": tensors[f"{p}.ln_2.bias"],
        "attn.w": attn_w, "attn.b": attn_b,
        "proj.w": proj_w, "proj.b": proj_b,
        "fc.w": fc_w, "fc.b": fc_b,
        "fc2.w": fc2_w, "fc2.b": fc2_b,
    }


def count_params(cfg: dict[str, Any], tensors: dict[str, np.ndarray]) -> int:
    n = tensors["wte.weight"].size + tensors["wpe.weight"].size
    n += tensors["ln_f.weight"].size + tensors["ln_f.bias"].size
    for i in range(cfg["n_layer"]):
        for key in ("ln_1.weight", "ln_1.bias", "ln_2.weight", "ln_2.bias",
                    "attn.c_attn.weight", "attn.c_attn.bias",
                    "attn.c_proj.weight", "attn.c_proj.bias",
                    "mlp.c_fc.weight", "mlp.c_fc.bias",
                    "mlp.c_proj.weight", "mlp.c_proj.bias"):
            n += tensors[f"h.{i}.{key}"].size
    return int(n)


def build_writer(precision: str, cfg: dict[str, Any], tensors: dict[str, np.ndarray],
                 shared: dict[str, np.ndarray]) -> EcmWriter:
    """Construct an EcmWriter holding every tensor for the given precision."""
    wr = EcmWriter()
    wr.set_meta(
        model_name=cfg["model_name"], n_layer=cfg["n_layer"], n_head=cfg["n_head"],
        n_embd=cfg["n_embd"], n_ctx=cfg["n_ctx"], vocab=cfg["vocab"],
        eps=cfg["eps"], n_params=count_params(cfg, tensors),
    )

    wr.add("wte", "fp16", tensors["wte.weight"])
    wr.add("wpe", "fp16", tensors["wpe.weight"])
    wr.add("ln_f.g", "fp32", tensors["ln_f.weight"])
    wr.add("ln_f.b", "fp32", tensors["ln_f.bias"])

    for i in range(cfg["n_layer"]):
        lw = build_layer_weights(tensors, i)
        p = f"blocks.{i}"
        wr.add(f"{p}.ln_1.g", "fp32", lw["ln_1.g"])
        wr.add(f"{p}.ln_1.b", "fp32", lw["ln_1.b"])
        wr.add(f"{p}.ln_2.g", "fp32", lw["ln_2.g"])
        wr.add(f"{p}.ln_2.b", "fp32", lw["ln_2.b"])
        for suffix in SUPPORTED_LINEARS:
            w = lw[f"{suffix}.w"]
            if precision == "int8":
                q, s = quantize_int8_symmetric(w)
                wr.add(f"{p}.{suffix}.w", "int8", q)
                wr.add(f"{p}.{suffix}.s", "fp32", s)
            else:
                wr.add(f"{p}.{suffix}.w", "fp16", w)
            wr.add(f"{p}.{suffix}.b", "fp32", lw[f"{suffix}.b"])

    wr.add("tokenizer.vocab", "text", shared["vocab"])
    wr.add("tokenizer.merges", "text", shared["merges"])
    return wr


def artifact_name(model_name: str, precision: str) -> str:
    tag = "int8-per" if precision == "int8" else precision
    return f"{sanitize_name(model_name)}-{tag}.ecm"


def export(src: Path, out_dir: Path, precisions: list[str]) -> list[dict[str, Any]]:
    """Export the HF dir at ``src`` into ``out_dir`` for the given precisions.

    GPT-2 and Qwen2 / OLMo checkpoints are dispatched automatically from
    ``config.json``. Returns a list of result dicts (one per artifact).
    """
    from .ecm import detect_arch

    raw_cfg = json.loads((src / "config.json").read_text())
    arch = detect_arch(raw_cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    if arch == "qwen2":
        from .ecm import export_qwen2_from_safetensors
        for precision in precisions:
            out = out_dir / artifact_name(raw_cfg.get("_name_or_path", "qwen2"), precision)
            r = export_qwen2_from_safetensors(src, out, precision=precision)
            r["arch"] = "qwen2"
            results.append(r)
        return results

    if "bf16" in precisions:
        raise ValueError(
            "bf16 export is only supported for Qwen2 / OLMo checkpoints "
            "(native bf16 weights); GPT-2 checkpoints store fp32 weights and "
            "support fp16 / int8 only")

    cfg, tensors, shared = build_shared_tensors(src)
    for precision in precisions:
        out = out_dir / artifact_name(cfg["model_name"], precision)
        wr = build_writer(precision, cfg, tensors, shared)
        with open(out, "wb") as fh:
            wr.write(fh)
        results.append({
            "model": cfg["model_name"],
            "arch": "gpt2",
            "precision": precision,
            "n_layer": cfg["n_layer"],
            "n_head": cfg["n_head"],
            "n_embd": cfg["n_embd"],
            "n_ctx": cfg["n_ctx"],
            "vocab": cfg["vocab"],
            "n_params": cfg.get("n_params") or count_params(cfg, tensors),
            "out": str(out),
            "size_mb": out.stat().st_size / 1e6,
        })
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Export an HF GPT-2 or Qwen2 safetensors directory to .ecm artifacts",
        prog="python -m edgecore.export")
    ap.add_argument("--src", required=True, help="HF model directory (config.json, model.safetensors, vocab.json, merges.txt)")
    ap.add_argument("--out", default="models", help="output directory (default models)")
    ap.add_argument("--precision", default="both",
                    choices=["fp16", "bf16", "int8", "both", "all"],
                    help="which artifact(s) to write (default both = fp16,int8; all = fp16,bf16,int8)")
    args = ap.parse_args(argv)

    src = Path(args.src)
    if args.precision == "all":
        precisions = ["fp16", "bf16", "int8"]
    else:
        precisions = ["fp16", "int8"] if args.precision == "both" else [args.precision]
    for r in export(src, Path(args.out), precisions):
        print(f"exported {r['precision']:>5}: {r['out']} ({r['size_mb']:.2f} MB, "
              f"{r['n_params']:,} params, L{r['n_layer']} H{r['n_head']} E{r['n_embd']} V{r['vocab']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
