"""EdgeCore - verification module.

Step 9: validates one or more ``.ecm`` artifacts against reference
implementations across multiple prompt lengths using three metrics:

  cosine   mean/min cosine similarity between runtime logits rows and
           the reference logits rows
  top1     fraction of positions whose argmax prediction matches the
           reference
  ppl      perplexity ratio (runtime / reference) on the same token
           sequence

References
  hf       pure-fp32 ground truth reconstructed from the original HF
           safetensors directory (``simulate_fp16_kv=False``)
  ecm      numpy replication of the exact artifact precision, including
           the fp16 KV-cache rounding the C runtime applies
           (``simulate_fp16_kv=True``)

Pass/fail thresholds are configurable. ``--strict`` tightens the
defaults; ``--no-fail`` downgrades failures to warnings (report-only
mode).

All comparisons feed explicit token ids through ``--tokens-file`` so the
tokenization can never differ between the two implementations. A separate
optional check compares the C tokenizer against the Python fallback
scanner on a short sentence.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .ecm import EcmReader, first_linear_key
from .reference import load_ecm_reference, load_safetensors_reference
from .tokenizer import ByteLevelBPE
from .bench import default_runtime

__all__ = ["verify", "default_thresholds", "strict_thresholds"]

# Probe corpus: plain ASCII so Python's byte-scanner and the C tokenizer
# agree on tokenization (both operate byte-wise on ASCII).
_CORPUS = (
    "The rapid adoption of small language models has created a need for "
    "efficient deployment on diverse computing environments such as "
    "commodity CPUs, on-premise servers, edge devices and isolated or "
    "air-gapped infrastructures. Unlike conventional software workloads, "
    "small language model inference performance depends strongly on the "
    "interaction between the model architecture, hardware capabilities, "
    "memory hierarchy, instruction set support, workload characteristics "
    "and runtime configuration. Parameters such as quantization, kernel "
    "implementation, thread count, batch size and CPU affinity can "
    "significantly affect latency, throughput, memory consumption and "
    "overall deployment cost. This project proposes a hardware adaptive "
    "inference and deployment system that automatically analyzes the "
    "model and target hardware, searches a configurable execution space, "
    "evaluates candidate configurations using real hardware measurements, "
    "verifies the selected configuration and produces a reproducible "
    "deployment package."
)

_SHORT = "EdgeCore verifies the selected configuration on real hardware."


def default_thresholds() -> dict[str, float]:
    return {"cosine": 0.99, "top1": 0.95, "ppl_ratio": 1.05}


def strict_thresholds() -> dict[str, float]:
    return {"cosine": 0.995, "top1": 0.97, "ppl_ratio": 1.02}


# ----------------------------------------------------------------------
# Probes
# ----------------------------------------------------------------------

def _synthetic_vocab(vocab_size: int) -> dict[str, int]:
    """Byte-level vocab (token i is byte i), used for synthetic .ecm models."""
    return {chr(i): i for i in range(min(int(vocab_size), 256))}


def _tokenizer_for(hf_dir: str | Path | None, meta: dict[str, Any]) -> ByteLevelBPE | None:
    variant = 1 if _arch_of(meta) == "qwen2" else 0
    if hf_dir is not None:
        return ByteLevelBPE(hf_dir=Path(hf_dir), variant=variant)
    vocab = int(meta.get("vocab", 0))
    if 0 < vocab <= 256:
        return ByteLevelBPE(vocab_json=_synthetic_vocab(vocab), merges=[], variant=variant)
    return None


def _arch_of(meta: dict[str, Any]) -> str:
    from .ecm import arch_from_meta
    return arch_from_meta(meta)


def _build_probes(lengths: list[int], tokenizer: ByteLevelBPE | None,
                  meta: dict[str, Any]) -> dict[str, np.ndarray]:
    n_ctx = int(meta.get("n_ctx", 1024))
    if tokenizer is not None:
        ids = tokenizer.encode(_CORPUS)
    else:
        ids = list(range(min(max(lengths) + 2, n_ctx)))
    probes: dict[str, np.ndarray] = {}
    for L in lengths:
        L = min(int(L), max(1, n_ctx - 2))
        if L <= len(ids):
            probes[str(L)] = np.asarray(ids[:L], dtype=np.int64)
    return probes


# ----------------------------------------------------------------------
# Runtime interaction
# ----------------------------------------------------------------------

def _read_logits_bin(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the raw dump written by ``--logits-bin``.

    Layout: int32 n, int32 V, tokens[n] int32, logits[n*V] f32.
    """
    raw = Path(path).read_bytes()
    n, V = struct.unpack_from("<ii", raw, 0)
    toks = np.frombuffer(raw, dtype=np.int32, count=n, offset=8)
    logits = np.frombuffer(raw, dtype=np.float32, count=n * V,
                           offset=8 + 4 * n).reshape(n, V)
    return toks, logits


def _run_runtime_logits(runtime: Path, model: Path, tokens: np.ndarray,
                        threads: int = 1, affinity: str = "none",
                        timeout: int = 900) -> tuple[np.ndarray, np.ndarray]:
    """Run the C runtime in logits mode on the given token ids."""
    ids_txt = " ".join(str(int(t)) for t in tokens)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(ids_txt)
        ids_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        bin_path = f.name
    try:
        cmd = [str(runtime), "--model", str(model), "--mode", "logits",
               "--tokens-file", ids_path, "--logits-bin", bin_path,
               "--threads", str(threads), "--affinity", affinity]
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(
                f"runtime failed rc={proc.returncode}: "
                f"{proc.stderr.decode(errors='replace')[:400]}")
        return _read_logits_bin(bin_path)
    finally:
        for p in (ids_path, bin_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def _check_tokenizer(runtime: Path, model: Path, n_ctx: int,
                     tokenizer: ByteLevelBPE | None) -> dict[str, Any]:
    """Run the C tokenizer on a short sentence and compare with Python."""
    if tokenizer is None:
        return {"checked": False, "reason": "no text tokenizer available"}
    py_ids = tokenizer.encode(_SHORT)
    if len(py_ids) >= n_ctx:
        # the C runtime does not clamp prompts to n_ctx in logits mode and
        # would write the KV cache out of bounds; skip for small contexts.
        return {"checked": False, "reason": f"prompt too long for n_ctx={n_ctx}"}
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name
    try:
        cmd = [str(runtime), "--model", str(model), "--mode", "logits",
               "--prompt", _SHORT, "--output", out_path]
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE, timeout=900)
        if proc.returncode != 0:
            return {"checked": False, "reason": f"runtime rc={proc.returncode}"}
        data = json.loads(Path(out_path).read_text())
        c_ids = [int(t) for t in data.get("tokens", [])]
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass
    if len(c_ids) != len(py_ids):
        return {"checked": True, "agreement": 0.0, "python_n": len(py_ids),
                "c_n": len(c_ids), "detail": "token count mismatch"}
    agree = sum(1 for a, b in zip(c_ids, py_ids) if a == b) / max(1, len(py_ids))
    return {"checked": True, "agreement": float(agree),
            "python_n": len(py_ids), "c_n": len(c_ids)}


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def row_cosines(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    an = np.linalg.norm(a, axis=1)
    bn = np.linalg.norm(b, axis=1)
    denom = an * bn
    denom[denom == 0] = 1.0
    c = (a * b).sum(axis=1) / denom
    return float(c.mean()), float(c.min())


def top1_agreement(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.argmax(a, axis=1) == np.argmax(b, axis=1)))


def _log_softmax(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.float64)
    m = z.max(axis=1, keepdims=True)
    e = np.exp(z - m)
    lse = np.log(e.sum(axis=1, keepdims=True))
    return z - m - lse


def perplexity(logits: np.ndarray, tokens: np.ndarray) -> float:
    if tokens.shape[0] < 2:
        return math.inf
    ls = _log_softmax(logits)
    nll = -ls[:-1, tokens[1:].astype(np.int64)]
    return math.exp(float(nll.mean()))


def _compare(runtime_logits: np.ndarray, ref_logits: np.ndarray,
             tokens: np.ndarray) -> dict[str, Any]:
    cos_mean, cos_min = row_cosines(runtime_logits, ref_logits)
    top1 = top1_agreement(runtime_logits, ref_logits)
    ppl_r = perplexity(runtime_logits, tokens)
    ppl_x = perplexity(ref_logits, tokens)
    finite = math.isfinite(ppl_r) and math.isfinite(ppl_x) and ppl_x > 0
    ratio = ppl_r / ppl_x if finite else math.inf
    return {
        "cosine_mean": cos_mean,
        "cosine_min": cos_min,
        "top1": top1,
        "ppl_runtime": ppl_r,
        "ppl_reference": ppl_x,
        "ppl_ratio": ratio,
    }


def _passes(cmp: dict[str, Any], th: dict[str, float]) -> bool:
    return (cmp["cosine_mean"] >= th["cosine"]
            and cmp["top1"] >= th["top1"]
            and cmp["ppl_ratio"] <= th["ppl_ratio"])


# ----------------------------------------------------------------------
# Main verification
# ----------------------------------------------------------------------

def verify(artifacts: list[str | Path], hf_dir: str | Path | None = None,
           runtime: Path | None = None, lengths: list[int] | None = None,
           reference: str = "auto", thresholds: dict[str, float] | None = None,
           check_tokenizer: bool = True, threads: int = 1,
           affinity: str = "none") -> dict[str, Any]:
    """Verify each artifact; returns a JSON-serialisable report."""
    runtime = runtime or default_runtime()
    if not runtime.exists():
        raise FileNotFoundError(f"runtime not found: {runtime} (run `make`)")
    th = thresholds or default_thresholds()
    lengths = lengths or [4, 16, 64]

    hf_ref = None
    if hf_dir is not None and reference in ("auto", "hf"):
        hf_ref = load_safetensors_reference(hf_dir, simulate_fp16_kv=False)

    _ref_cache: dict[tuple[int, tuple[int, ...]], np.ndarray] = {}

    models: dict[str, Any] = {}
    for art_path in artifacts:
        art = Path(art_path)
        meta = EcmReader(art).meta
        tokenizer = _tokenizer_for(hf_dir, meta)
        probes = _build_probes(lengths, tokenizer, meta)
        ecm_ref = load_ecm_reference(art, simulate_fp16_kv=True)
        refs: dict[str, Any] = {"ecm": ecm_ref}
        primary = "ecm"
        if hf_ref is not None:
            refs["hf"] = hf_ref
            primary = "hf"
        if reference == "ecm":
            primary = "ecm"

        lengths_res: dict[str, Any] = {}
        for L, tokens in probes.items():
            toks, r_logits = _run_runtime_logits(runtime, art, tokens,
                                                 threads=threads,
                                                 affinity=affinity)
            comparisons: dict[str, Any] = {}
            key = (id(hf_ref) if hf_ref is not None else -1,
                   tuple(int(t) for t in tokens))
            for rname, ref in refs.items():
                if rname == "hf" and key in _ref_cache:
                    ref_logits = _ref_cache[key]
                else:
                    ref_logits = ref.forward(tokens)
                    if rname == "hf":
                        _ref_cache[key] = ref_logits
                comparisons[rname] = _compare(r_logits, ref_logits, tokens)
            lengths_res[L] = {
                "n_tokens": int(len(tokens)),
                "runtime_tokens_head": [int(t) for t in toks[:8]],
                "comparisons": comparisons,
                "primary": primary,
                "pass": _passes(comparisons[primary], th),
            }

        prim = [lengths_res[L]["comparisons"][primary] for L in lengths_res]
        tok_res = _check_tokenizer(runtime, art, int(meta.get("n_ctx", 0)),
                                   tokenizer) if check_tokenizer \
            else {"checked": False, "reason": "disabled"}
        summary = {
            "worst_cosine_mean": min(c["cosine_mean"] for c in prim) if prim else None,
            "worst_cosine_min": min(c["cosine_min"] for c in prim) if prim else None,
            "worst_top1": min(c["top1"] for c in prim) if prim else None,
            "max_ppl_ratio": max(c["ppl_ratio"] for c in prim) if prim else None,
            "pass": all(lengths_res[L]["pass"] for L in lengths_res) and bool(lengths_res),
        }
        models[str(art)] = {
            "model_name": meta.get("model_name", "unknown"),
            "precision": _precision_of(art),
            "primary_reference": primary,
            "n_ctx": int(meta.get("n_ctx", 0)),
            "lengths": lengths_res,
            "summary": summary,
            "tokenizer": tok_res,
        }

    failed = [m for m, r in models.items() if not r["summary"]["pass"]]
    overall = "pass" if not failed else "fail"
    report: dict[str, Any] = {
        "tool": "edgecore.verify",
        "version": "0.1.0",
        "runtime": str(runtime),
        "hf_dir": str(hf_dir) if hf_dir else None,
        "thresholds": th,
        "strict": th == strict_thresholds(),
        "overall": overall,
        "failed_models": failed,
        "models": models,
    }
    return report


def _precision_of(art: Path) -> str:
    r = EcmReader(art)
    key = first_linear_key(r.meta)
    w = r.tensor_meta.get(key)
    if not w:
        return "unknown"
    return {"int8": "int8", "fp16": "fp16", "bf16": "bf16"}.get(w["dtype"], w["dtype"])


# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Verify .ecm artifacts (cosine / top-1 / ppl across prompt lengths)",
        prog="python -m edgecore.verify")
    ap.add_argument("--model", action="append", required=True,
                    help=".ecm artifact(s); repeatable")
    ap.add_argument("--hf-dir", default=None,
                    help="original HF safetensors directory (fp32 ground truth)")
    ap.add_argument("--runtime", default=None, help="path to edgecore-runtime")
    ap.add_argument("--lengths", default="4,16,64",
                    help="comma-separated prompt lengths in tokens")
    ap.add_argument("--reference", default="auto", choices=["auto", "hf", "ecm"],
                    help="primary reference (default auto: hf if --hf-dir given)")
    ap.add_argument("--cosine-threshold", type=float, default=None)
    ap.add_argument("--top1-threshold", type=float, default=None)
    ap.add_argument("--ppl-ratio-threshold", type=float, default=None)
    ap.add_argument("--strict", action="store_true", help="use tight thresholds")
    ap.add_argument("--no-fail", action="store_true",
                    help="never exit non-zero (report-only)")
    ap.add_argument("--threads", type=int, default=1,
                    help="threads used for runtime logits runs")
    ap.add_argument("--affinity", default="none",
                    help="affinity policy for runtime logits runs")
    ap.add_argument("--check-tokenizer", dest="check_tokenizer",
                    action="store_true", default=True)
    ap.add_argument("--no-check-tokenizer", dest="check_tokenizer",
                    action="store_false")
    ap.add_argument("--output", default=None, help="write JSON report to file")
    args = ap.parse_args(argv)

    th = strict_thresholds() if args.strict else default_thresholds()
    if args.cosine_threshold is not None:
        th["cosine"] = args.cosine_threshold
    if args.top1_threshold is not None:
        th["top1"] = args.top1_threshold
    if args.ppl_ratio_threshold is not None:
        th["ppl_ratio"] = args.ppl_ratio_threshold

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    report = verify(
        args.model,
        hf_dir=args.hf_dir,
        runtime=Path(args.runtime) if args.runtime else None,
        lengths=lengths,
        reference=args.reference,
        thresholds=th,
        check_tokenizer=args.check_tokenizer,
        threads=args.threads,
        affinity=args.affinity,
    )
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text)
        print(f"verification report written to {args.output}")
    else:
        print(text)

    print(f"\noverall: {report['overall'].upper()}", file=sys.stderr)
    for m, r in report["models"].items():
        s = r["summary"]
        print(f"  {m}: {r['precision']:>4} ref={r['primary_reference']:>3} "
              f"cosine={s['worst_cosine_mean']:.4f} top1={s['worst_top1']:.4f} "
              f"ppl_ratio={s['max_ppl_ratio']:.4f} -> "
              f"{'PASS' if s['pass'] else 'FAIL'}", file=sys.stderr)

    if report["overall"] == "fail" and not args.no_fail:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
