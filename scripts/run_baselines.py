#!/usr/bin/env python3
"""EdgeCore baseline pipeline.

Runs the full EdgeCore toolchain (fetch/generate -> export -> analyze ->
verify -> bench) for one model and writes a self-contained baseline report
under ``models/baselines/`` that the baseline-comparison step
(``scripts/compare_baselines.py``) merges with PyTorch / llama.cpp numbers.

Models are addressed by a short key:

  tiny-gpt2               2-layer synthetic GPT-2 (make_synthetic.py)
  tiny-qwen2              2-layer synthetic Qwen2  (make_synthetic.py)
  <any HF id>             real model, e.g. gpt2 or Qwen/Qwen2.5-0.5B-Instruct
                           (fetched with scripts/fetch_model.py)

Every report uses the SAME benchmark prompt / token count / warmup / iters
so the EdgeCore column is directly comparable to the torch / llama.cpp
columns produced by ``compare_baselines.py``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPT = ("The rapid adoption of small language models has created a need for "
          "efficient deployment on diverse computing environments such as "
          "commodity CPUs, on-premise servers and edge devices.")


def _py() -> list[str]:
    env = {"PYTHONPATH": str(REPO_ROOT)}
    import os
    return [sys.executable]  # cwd=REPO_ROOT keeps edgecore importable


def _run_tool(*argv: str) -> None:
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    subprocess.run(list(argv), cwd=REPO_ROOT, check=True, env=env)


def _import_main(module: str, argv: list[str]) -> None:
    import importlib
    importlib.import_module(module).main(argv)


def run(model: str, workdir: Path, runtime: Path, hwprof: Path,
        lengths: list[int], threads: list[int], batch: list[int],
        affinities: list[str], n_tokens: int, warmup: int, iters: int,
        force_fetch: bool = False) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    bl = workdir / "baselines"
    bl.mkdir(parents=True, exist_ok=True)
    key = model.replace("/", "__")

    src: Path | None = None
    artifacts: dict[str, Path] = {}
    if model in ("tiny-gpt2", "tiny-qwen2"):
        _run_tool("python3", "scripts/make_synthetic.py",
                  "--arch", model.split("-")[1], "--precision", "both")
        tag = model.split("-")[1]
        artifacts = {
            "int8": workdir / f"tiny-{tag}-int8.ecm",
            "fp16": workdir / f"tiny-{tag}-fp16.ecm",
        }
    else:
        src = workdir / "downloads" / key
        _run_tool("python3", "scripts/fetch_model.py",
                  "--name", model, "--out", str(src))
        if force_fetch:
            _run_tool("python3", "scripts/fetch_model.py",
                      "--name", model, "--out", str(src), "--force")
        from edgecore.export import export
        results = export(src, workdir, ["fp16", "int8"])
        artifacts = {r["precision"]: Path(r["out"]) for r in results}

    from edgecore.analyze import analyze_ecm
    a_path = bl / f"{key}-analyze.json"
    a_path.write_text(json.dumps(analyze_ecm(artifacts["fp16"]), indent=2))

    from edgecore.verify import verify as _verify
    v_path = bl / f"{key}-verify.json"
    report = _verify([artifacts["fp16"], artifacts["int8"]],
                     hf_dir=src, runtime=runtime, lengths=lengths,
                     reference="ecm" if src is None else "auto",
                     threads=1, affinity="none")
    v_path.write_text(json.dumps(report, indent=2))

    from edgecore.bench import autotune
    bench = autotune(
        [artifacts["fp16"], artifacts["int8"]],
        prompt=PROMPT, n_tokens=n_tokens, warmup=warmup, iters=iters,
        runtime=runtime, hwprof=hwprof,
        thread_values=threads, batch_values=batch, affinities=affinities)
    b_path = bl / f"{key}-bench.json"
    b_path.write_text(json.dumps(bench, indent=2))

    summary = {
        "model": model,
        "artifacts": {p: str(a) for p, a in artifacts.items()},
        "verification": {
            m: {"precision": v.get("precision"),
                "pass": v["summary"]["pass"],
                "cosine": v["summary"].get("worst_cosine_mean"),
                "top1": v["summary"].get("worst_top1"),
                "ppl_ratio": v["summary"].get("max_ppl_ratio")}
            for m, v in report.get("models", {}).items()},
        "benchmark": {
            "recommendations": bench.get("recommendations"),
            "grid_size": bench.get("grid_size")},
        "reports": {"analyze": str(a_path), "verify": str(v_path),
                    "bench": str(b_path)},
    }
    out = bl / f"{key}-baseline.json"
    out.write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="tiny-gpt2",
                    help="tiny-gpt2 | tiny-qwen2 | <HF id> (default tiny-gpt2)")
    ap.add_argument("--workdir", default="models")
    ap.add_argument("--runtime", default=None)
    ap.add_argument("--hwprof", default=None)
    ap.add_argument("--lengths", default="4,16")
    ap.add_argument("--threads", default="1,2")
    ap.add_argument("--batch", default="1,2")
    ap.add_argument("--affinity", default="none,physical")
    ap.add_argument("--n-tokens", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--force-fetch", action="store_true")
    args = ap.parse_args(argv)

    from edgecore.bench import default_runtime, default_hwprof
    runtime = Path(args.runtime) if args.runtime else default_runtime()
    hwprof = Path(args.hwprof) if args.hwprof else default_hwprof()
    if not runtime.exists() or not hwprof.exists():
        subprocess.run(["make"], cwd=REPO_ROOT, check=True)
    if not runtime.exists() or not hwprof.exists():
        print(f"error: runtime/hwprof missing ({runtime}, {hwprof})",
              file=sys.stderr)
        return 1

    summary = run(
        args.model, Path(args.workdir), runtime, hwprof,
        [int(x) for x in args.lengths.split(",") if x.strip()],
        [int(x) for x in args.threads.split(",") if x.strip()],
        [int(x) for x in args.batch.split(",") if x.strip()],
        [a for a in args.affinity.split(",") if a.strip()],
        args.n_tokens, args.warmup, args.iters, args.force_fetch)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
