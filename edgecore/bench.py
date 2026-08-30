"""EdgeCore - benchmark runner and two-stage auto-tuner.

Step 8: measures real performance of the C inference runtime across a
configuration grid and then applies a two-stage search:

  Stage 1 (grid): enumerate (precision, threads, batch, affinity) configs
                  and run controlled benchmarks (TTFT / TPOT / p50-p99 /
                  throughput / peak RSS).

  Stage 2 (Pareto): compute the non-dominated set across objectives
                  (latency, TTFT, throughput, memory) and then map
                  Pareto-optimal configs to deployment scenarios
                  (low-latency interactive, high-throughput batch,
                  low-memory edge, balanced).

The chosen configuration(s) are emitted as a machine-readable tuned
config used by the deployment packager.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_PROMPT = ("The rapid adoption of small language models has created a need for "
                  "efficient deployment on diverse computing environments such as "
                  "commodity CPUs, on-premise servers and edge devices.")

SCENARIOS: dict[str, dict[str, Any]] = {
    "low-latency-interactive": {
        "desc": "Interactive assistant / chat: minimize per-token latency and TTFT.",
        "objectives": {"avg_token_ms": "min", "ttft_avg_ms": "min"},
    },
    "high-throughput-batch": {
        "desc": "Offline batch / RAG embedding: maximize tokens generated per second.",
        "objectives": {"tokens_per_sec": "max"},
    },
    "low-memory-edge": {
        "desc": "Memory-constrained edge device: minimize peak resident set.",
        "objectives": {"peak_rss_gb": "min"},
    },
    "balanced": {
        "desc": "General-purpose server: best compromise across all objectives.",
        "objectives": {"avg_token_ms": "min", "tokens_per_sec": "max",
                       "peak_rss_gb": "min", "ttft_avg_ms": "min"},
    },
}

METRIC_KEYS = ["avg_token_ms", "ttft_avg_ms", "p50_ms", "p95_ms", "p99_ms",
               "tokens_per_sec", "peak_rss_gb"]


def default_runtime() -> Path:
    env = os.environ.get("EDGECORE_RUNTIME")
    if env:
        return Path(env)
    cand = Path(__file__).resolve().parent.parent / "build" / "edgecore-runtime"
    return cand if cand.exists() else Path("build/edgecore-runtime")


def default_hwprof() -> Path:
    env = os.environ.get("EDGECORE_HWPROF")
    if env:
        return Path(env)
    cand = Path(__file__).resolve().parent.parent / "build" / "edgecore-hwprof"
    return cand if cand.exists() else Path("build/edgecore-hwprof")


# ----------------------------------------------------------------------
def run_hw_profile(hwprof: Path) -> dict[str, Any]:
    out = subprocess.run([str(hwprof)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def build_grid(artifacts: list[Path], hw: dict[str, Any],
               batch_values: list[int] | None = None,
               thread_values: list[int] | None = None,
               affinities: list[str] | None = None) -> list[dict[str, Any]]:
    """Enumerate the configuration grid over all artifacts."""
    logical = int(hw.get("logical_cores", os.cpu_count() or 2))
    threads = thread_values or _default_threads(logical)
    batches = batch_values or [1, 2, 4]
    affs = affinities or ["none", "physical"]
    grid: list[dict[str, Any]] = []
    for art in artifacts:
        for th in threads:
            for b in batches:
                for aff in affs:
                    grid.append({
                        "model": str(art),
                        "precision": _precision_of(art),
                        "threads": th,
                        "batch": b,
                        "affinity": aff,
                    })
    return grid


def _default_threads(logical: int) -> list[int]:
    if logical <= 1:
        return [1]
    if logical == 2:
        return [1, 2]
    return [1, 2, 4, logical]


def _precision_of(art: Path) -> str:
    from .ecm import EcmReader
    try:
        r = EcmReader(art)
        w = r.tensor_meta.get("blocks.0.attn.w")
        return "int8" if w and w["dtype"] == "int8" else "fp16"
    except Exception:
        return "?"


# ----------------------------------------------------------------------
def run_one(runtime: Path, cfg: dict[str, Any], prompt: str,
            n_tokens: int, warmup: int, iters: int, timeout: int = 600) -> dict[str, Any]:
    """Run a single benchmark config; returns the merged config + metrics."""
    cmd = [
        str(runtime),
        "--model", cfg["model"],
        "--mode", "bench",
        "--prompt", prompt,
        "--n-tokens", str(n_tokens),
        "--warmup", str(warmup),
        "--iters", str(iters),
        "--batch", str(cfg["batch"]),
        "--threads", str(cfg["threads"]),
        "--affinity", cfg["affinity"],
        "--no-print",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        return {**cfg, "error": proc.stderr.strip() or f"exit {proc.returncode}"}
    try:
        m = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {**cfg, "error": "invalid runtime JSON output"}
    merged = {**cfg}
    for k in METRIC_KEYS:
        merged[k] = m.get(k)
    merged["n_generated_tokens"] = m.get("n_generated_tokens")
    merged["iterations"] = m.get("iterations")
    return merged


# ----------------------------------------------------------------------
def pareto_front(results: list[dict[str, Any]],
                 minimize: list[str], maximize: list[str]) -> list[dict[str, Any]]:
    """Non-dominated set. A config A dominates B if A is >= on every
    maximized objective, <= on every minimized objective, and strictly
    better on at least one."""
    dominated: set[int] = set()
    n = len(results)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = results[i], results[j]
            if _dominates(a, b, minimize, maximize):
                dominated.add(j)
    return [r for k, r in enumerate(results) if k not in dominated]


def _dominates(a: dict, b: dict, minimize: list[str], maximize: list[str]) -> bool:
    better = False
    for k in minimize:
        va, vb = a.get(k), b.get(k)
        if va is None or vb is None:
            return False
        if va > vb:
            return False
        if va < vb:
            better = True
    for k in maximize:
        va, vb = a.get(k), b.get(k)
        if va is None or vb is None:
            return False
        if va < vb:
            return False
        if va > vb:
            better = True
    return better


# ----------------------------------------------------------------------
def score_for_scenario(cfg: dict[str, Any], scenario: str,
                       bounds: dict[str, tuple[float, float]]) -> float:
    """Weighted normalized score (lower is better) for a scenario."""
    total = 0.0
    weights = {"avg_token_ms": 1.0, "ttft_avg_ms": 1.0,
               "tokens_per_sec": 1.0, "peak_rss_gb": 1.0}
    for k, direction in SCENARIOS[scenario]["objectives"].items():
        lo, hi = bounds[k]
        v = cfg.get(k)
        if v is None or hi <= lo:
            total += 1.0
            continue
        norm = (v - lo) / (hi - lo)
        total += (norm if direction == "min" else (1.0 - norm)) * weights[k]
    return total


def recommend(results: list[dict[str, Any]],
              pareto: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the best config per deployment scenario (preferring Pareto set)."""
    valid = [r for r in results if not r.get("error")]
    if not valid:
        return {"error": "no valid benchmark results"}

    bounds: dict[str, tuple[float, float]] = {}
    for k in ["avg_token_ms", "ttft_avg_ms", "p50_ms", "p95_ms", "p99_ms",
              "tokens_per_sec", "peak_rss_gb"]:
        vals = [r.get(k) for r in valid if r.get(k) is not None]
        bounds[k] = (min(vals), max(vals)) if vals else (0.0, 0.0)

    pool = pareto if pareto else valid
    recs: dict[str, Any] = {}
    for name, spec in SCENARIOS.items():
        scored = sorted(pool, key=lambda c: score_for_scenario(c, name, bounds))
        best = scored[0]
        recs[name] = {
            "description": spec["desc"],
            "objectives": spec["objectives"],
            "config": {k: best.get(k) for k in
                       ["precision", "threads", "batch", "affinity", "model"]},
            "metrics": {k: best.get(k) for k in METRIC_KEYS},
        }
    return {"recommendations": recs, "bounds": bounds}


# ----------------------------------------------------------------------
def run_grid(runtime: Path, grid: list[dict[str, Any]], prompt: str,
             n_tokens: int, warmup: int, iters: int,
             only: list[tuple[int, int, str]] | None = None) -> list[dict[str, Any]]:
    """Stage 1: measure every config. `only` is a subset filter (threads,batch,affinity)."""
    results: list[dict[str, Any]] = []
    for i, cfg in enumerate(grid):
        if only:
            sel = (cfg["threads"], cfg["batch"], cfg["affinity"])
            if sel not in only:
                continue
        print(f"  [{i+1}/{len(grid)}] {cfg['precision']:>4} threads={cfg['threads']} "
              f"batch={cfg['batch']} affinity={cfg['affinity']} ...", file=sys.stderr, end=" ")
        r = run_one(runtime, cfg, prompt, n_tokens, warmup, iters)
        if "error" in r:
            print(f"ERROR: {r['error']}", file=sys.stderr)
        else:
            print(f"avg={r['avg_token_ms']:.3f}ms ttft={r['ttft_avg_ms']:.3f}ms "
                  f"tps={r['tokens_per_sec']:.1f} rss={r['peak_rss_gb']:.3f}GB",
                  file=sys.stderr)
        results.append(r)
    return results


def autotune(artifacts: list[Path], prompt: str = DEFAULT_PROMPT, n_tokens: int = 32,
             warmup: int = 1, iters: int = 3, runtime: Path | None = None,
             hwprof: Path | None = None, batch_values: list[int] | None = None,
             thread_values: list[int] | None = None, affinities: list[str] | None = None,
             only: list[tuple[int, int, str]] | None = None) -> dict[str, Any]:
    """Two-stage auto-tune: grid search then Pareto + scenario recommendations."""
    runtime = runtime or default_runtime()
    hwprof = hwprof or default_hwprof()
    if not runtime.exists():
        raise FileNotFoundError(f"runtime not found: {runtime} (run `make`)")
    if not hwprof.exists():
        raise FileNotFoundError(f"hwprof not found: {hwprof} (run `make`)")

    hw = run_hw_profile(hwprof)
    grid = build_grid(artifacts, hw, batch_values, thread_values, affinities)
    print(f"hardware: {hw.get('cpu_model')} | {hw.get('logical_cores')} logical cores "
          f"| {hw.get('ram_gb')} GB", file=sys.stderr)
    print(f"grid: {len(grid)} configs", file=sys.stderr)

    results = run_grid(runtime, grid, prompt, n_tokens, warmup, iters, only)

    pareto = pareto_front([r for r in results if not r.get("error")],
                          minimize=["avg_token_ms", "ttft_avg_ms", "peak_rss_gb"],
                          maximize=["tokens_per_sec"])
    rec = recommend(results, pareto)

    return {
        "hardware": {
            "cpu_model": hw.get("cpu_model"),
            "physical_cores": hw.get("physical_cores"),
            "logical_cores": hw.get("logical_cores"),
            "ram_gb": hw.get("ram_gb"),
            "simd": hw.get("simd"),
        },
        "bench_config": {
            "prompt_len_tokens": len(prompt.split()),
            "prompt": prompt,
            "n_tokens": n_tokens,
            "warmup": warmup,
            "iters": iters,
        },
        "grid_size": len(grid),
        "measurements": results,
        "pareto_front": pareto,
        "pareto_configs": [{k: c.get(k) for k in
                            ["precision", "threads", "batch", "affinity", "model"]}
                           for c in pareto],
        "recommendations": rec.get("recommendations", {}),
        "bounds": rec.get("bounds", {}),
    }


# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Benchmark and auto-tune an EdgeCore model (grid + Pareto + scenario recs)",
        prog="python -m edgecore.bench")
    ap.add_argument("--model", action="append", default=[], help=".ecm artifact(s); repeatable")
    ap.add_argument("--runtime", default=None, help="path to edgecore-runtime")
    ap.add_argument("--hwprof", default=None, help="path to edgecore-hwprof")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="benchmark prompt")
    ap.add_argument("--n-tokens", type=int, default=32, help="generated tokens per iteration")
    ap.add_argument("--warmup", type=int, default=1, help="warmup iterations")
    ap.add_argument("--iters", type=int, default=3, help="measured iterations")
    ap.add_argument("--threads", type=str, default=None,
                    help="comma-separated thread counts (default auto from hw)")
    ap.add_argument("--batch", type=str, default=None, help="comma-separated batch sizes")
    ap.add_argument("--affinity", type=str, default=None,
                    help="comma-separated affinity policies (none,physical,<cpulist>)")
    ap.add_argument("--output", default=None, help="write JSON report to file")
    args = ap.parse_args(argv)

    if not args.model:
        print("error: at least one --model artifact is required", file=sys.stderr)
        return 2

    thread_values = [int(x) for x in args.threads.split(",")] if args.threads else None
    batch_values = [int(x) for x in args.batch.split(",")] if args.batch else None
    affinities = [a for a in args.affinity.split(",")] if args.affinity else None

    report = autotune(
        [Path(m) for m in args.model],
        prompt=args.prompt,
        n_tokens=args.n_tokens,
        warmup=args.warmup,
        iters=args.iters,
        runtime=Path(args.runtime) if args.runtime else None,
        hwprof=Path(args.hwprof) if args.hwprof else None,
        batch_values=batch_values,
        thread_values=thread_values,
        affinities=affinities,
    )
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text)
        print(f"benchmark report written to {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
