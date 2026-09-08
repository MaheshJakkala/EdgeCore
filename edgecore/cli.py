"""EdgeCore - unified command-line interface.

Step 10/11: a single entry point over the toolchain modules plus an
end-to-end pipeline driver.

    python -m edgecore.cli export|analyze|bench|verify|package [args...]
    python -m edgecore.cli hw [--hwprof PATH]
    python -m edgecore.cli e2e [options]

Each toolchain subcommand forwards its arguments to the corresponding
``edgecore.<module>`` CLI (see their ``--help``).
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

from . import __version__

__all__ = ["main"]

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _env() -> dict[str, str]:
    """Environment for subprocesses: ensure the repo root is on PYTHONPATH."""
    env = dict(os.environ)
    root = str(_REPO_ROOT)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = root if not existing else root + os.pathsep + existing
    return env

_MODULES = {
    "export": "edgecore.export",
    "analyze": "edgecore.analyze",
    "bench": "edgecore.bench",
    "verify": "edgecore.verify",
    "package": "edgecore.package",
}

SCENARIOS = ("low-latency-interactive", "high-throughput-batch",
             "low-memory-edge", "balanced")


def _run(module: str, argv: list[str]) -> int:
    return importlib.import_module(module).main(argv)


def _log(msg: str) -> None:
    print(f"[e2e] {msg}", file=sys.stderr, flush=True)


# ----------------------------------------------------------------------
def _cmd_hw(args: argparse.Namespace) -> int:
    from .bench import default_hwprof, run_hw_profile
    p = Path(args.hwprof) if args.hwprof else default_hwprof()
    if not p.exists():
        print(f"error: hwprof not found: {p} (run `make`)", file=sys.stderr)
        return 1
    print(json.dumps(run_hw_profile(p), indent=2))
    return 0


# ----------------------------------------------------------------------
def _cmd_e2e(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m edgecore.cli e2e",
        description="End-to-end run: fetch -> export -> verify -> bench -> package")
    ap.add_argument("--hf-id", default="gpt2", help="HF model id")
    ap.add_argument("--fetch", action="store_true",
                    help="download the HF model first (else --src must exist)")
    ap.add_argument("--src", default="models/downloads/gpt2",
                    help="HF model directory (config.json + model.safetensors + vocab)")
    ap.add_argument("--workdir", default="models", help="artifacts / reports dir")
    ap.add_argument("--runtime", default=None, help="path to edgecore-runtime")
    ap.add_argument("--hwprof", default=None, help="path to edgecore-hwprof")
    ap.add_argument("--scenario", default="balanced", choices=list(SCENARIOS))
    ap.add_argument("--lengths", default="4,16,64",
                    help="verify prompt lengths (tokens)")
    ap.add_argument("--threads", default="1,2", help="bench thread counts (csv)")
    ap.add_argument("--batch", default="1,2", help="bench batch sizes (csv)")
    ap.add_argument("--affinity", default="none,physical",
                    help="bench affinity policies (csv)")
    ap.add_argument("--n-tokens", type=int, default=16,
                    help="benchmark tokens per iteration")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--precisions", default="fp16,int8",
                    help="artifacts to export (csv)")
    ap.add_argument("--bench-models", default="int8,fp16",
                    help="which artifacts to autotune (int8,fp16, csv)")
    ap.add_argument("--out", default="dist", help="deployment bundle output dir")
    ap.add_argument("--skip-tiny", action="store_true",
                    help="skip the tiny-synthetic sanity pass")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--no-package", action="store_true")
    ap.add_argument("--verify-no-fail", action="store_true",
                    help="report verification results without failing the run")
    args = ap.parse_args(argv)

    from .bench import default_runtime, default_hwprof, run_hw_profile

    runtime = Path(args.runtime) if args.runtime else default_runtime()
    hwprof = Path(args.hwprof) if args.hwprof else default_hwprof()
    if not runtime.exists() or not hwprof.exists():
        _log("build artifacts missing; running `make`")
        subprocess.run(["make"], check=True, env=_env())
    if not runtime.exists() or not hwprof.exists():
        print(f"error: runtime/hwprof still missing ({runtime}, {hwprof})",
              file=sys.stderr)
        return 1

    hw = run_hw_profile(hwprof)
    _log(f"hardware: {hw.get('cpu_model')} | {hw.get('logical_cores')} logical "
         f"cores | {hw.get('ram_gb')} GB | simd: {hw.get('simd')}")

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # ---- 1. tiny synthetic sanity ------------------------------------
    if not args.skip_tiny:
        _log("tiny synthetic sanity pass")
        subprocess.run([sys.executable, "scripts/make_synthetic.py"],
                       cwd=_REPO_ROOT, check=True, env=_env())
        tiny_int8 = Path("models/tiny-gpt2-int8.ecm")
        tiny_fp16 = Path("models/tiny-gpt2-fp16.ecm")
        _run("edgecore.verify",
             ["--model", str(tiny_int8), "--model", str(tiny_fp16),
              "--runtime", str(runtime), "--output", str(workdir / "tiny-verify.json")])
        _run("edgecore.bench",
             ["--model", str(tiny_int8), "--runtime", str(runtime),
              "--hwprof", str(hwprof), "--threads", "1,2", "--batch", "1",
              "--affinity", "none", "--n-tokens", "8", "--warmup", "1",
              "--iters", "2", "--output", str(workdir / "tiny-bench.json")])
        _log("tiny sanity OK")

    # ---- 2. fetch -----------------------------------------------------
    src = Path(args.src)
    if args.fetch:
        _log(f"fetching HF model '{args.hf_id}' into {src}")
        subprocess.run([sys.executable, "scripts/fetch_model.py",
                        "--name", args.hf_id, "--out", str(src)],
                       check=True, env=_env())
    if not (src / "config.json").exists():
        print(f"error: {src}/config.json not found "
              "(use --fetch or point --src at an HF model dir)", file=sys.stderr)
        return 1

    # ---- 3. export -----------------------------------------------------
    from .export import export as _export
    precisions = [p.strip() for p in args.precisions.split(",") if p.strip()]
    _log(f"exporting {precisions} artifacts from {src}")
    results = _export(src, workdir, precisions)
    artifacts = {r["precision"]: Path(r["out"]) for r in results}
    for prec, path in artifacts.items():
        _log(f"  exported {prec}: {path}")

    bench_precs = [p.strip() for p in args.bench_models.split(",") if p.strip()]
    bench_precs = [p for p in bench_precs if p in artifacts]
    if not bench_precs:
        print("error: --bench-models does not match exported precisions",
              file=sys.stderr)
        return 1
    primary = artifacts[bench_precs[0]]

    # ---- 4. analyze -----------------------------------------------------
    analyze_rep = workdir / f"{primary.stem}-analyze.json"
    _run("edgecore.analyze", ["--model", str(primary),
                              "--output", str(analyze_rep)])
    _log(f"analyzed {primary.name} -> {analyze_rep.name}")

    # ---- 5. verify -----------------------------------------------------
    verify_rep = workdir / f"{primary.stem}-verify.json"
    if not args.skip_verify:
        _log(f"verifying {bench_precs} artifacts vs fp32 ground truth")
        cmd = ["--hf-dir", str(src), "--runtime", str(runtime),
               "--lengths", args.lengths, "--output", str(verify_rep)]
        for prec in bench_precs:
            cmd += ["--model", str(artifacts[prec])]
        if args.verify_no_fail:
            cmd += ["--no-fail"]
        rc = _run("edgecore.verify", cmd)
        vr0 = json.loads(verify_rep.read_text()) if verify_rep.exists() else {}
        n_pass = sum(1 for r in vr0.get("models", {}).values()
                     if r.get("summary", {}).get("pass"))
        if rc != 0 and n_pass == 0:
            _log("VERIFICATION FAILED for every artifact - aborting before packaging")
            return 1
        if rc != 0:
            _log(f"verification: {n_pass}/{len(vr0.get('models', {}))} artifacts "
                 f"passed; packaging will prefer a verified config")
        else:
            _log(f"verification PASS -> {verify_rep.name}")
    else:
        verify_rep.write_text(json.dumps({"overall": "skipped", "models": {}}))

    # ---- 6. benchmark / autotune ----------------------------------------
    bench_rep = workdir / f"{primary.stem}-bench.json"
    if not args.skip_bench:
        _log(f"benchmarking {bench_precs} artifacts "
             f"(threads={args.threads} batch={args.batch} "
             f"affinity={args.affinity})")
        cmd = ["--runtime", str(runtime), "--hwprof", str(hwprof),
               "--threads", args.threads, "--batch", args.batch,
               "--affinity", args.affinity, "--n-tokens", str(args.n_tokens),
               "--warmup", str(args.warmup), "--iters", str(args.iters),
               "--output", str(bench_rep)]
        for prec in bench_precs:
            cmd += ["--model", str(artifacts[prec])]
        rc = _run("edgecore.bench", cmd)
        if rc != 0:
            return rc
        _log(f"benchmark report -> {bench_rep.name}")
    else:
        bench_rep.write_text(json.dumps({
            "recommendations": {args.scenario: {
                "config": {"precision": artifacts[bench_precs[0]].stem,
                           "threads": 1, "batch": 1, "affinity": "none",
                           "model": str(artifacts[bench_precs[0]])},
                "metrics": {}, "description": "skipped", "objectives": {}}},
            "hardware": {k: hw.get(k) for k in
                         ("cpu_model", "logical_cores", "physical_cores",
                          "ram_gb", "simd")},
            "grid_size": 0}))

    # ---- 7. package -------------------------------------------------------
    # The deployment config must be VERIFIED. If the fastest scenario
    # recommendation (e.g. int8) failed the accuracy gates, fall back to the
    # best-scoring VERIFIED model for the same scenario so we never ship a
    # config that failed verification.
    vr = json.loads(verify_rep.read_text()) if verify_rep.exists() else {}
    verified_models = {
        Path(m) for m, r in vr.get("models", {}).items()
        if r.get("summary", {}).get("pass")
    }

    pkg_model = primary
    pkg_bench = bench_rep
    br = json.loads(bench_rep.read_text()) if bench_rep.exists() else {}
    rec = br.get("recommendations", {}).get(args.scenario, {})
    rec_model = Path(rec.get("config", {}).get("model", "")) if rec.get("config") else None
    if (verified_models and rec_model is not None and rec_model not in verified_models
            and not args.no_package):
        from .bench import pareto_front, recommend
        meas = [r for r in br.get("measurements", [])
                if not r.get("error") and Path(r.get("model", "")) in verified_models]
        if meas:
            pareto = pareto_front(meas,
                                  minimize=["avg_token_ms", "ttft_avg_ms", "peak_rss_gb"],
                                  maximize=["tokens_per_sec"])
            r2 = recommend(meas, pareto).get("recommendations", {}).get(args.scenario, {})
            alt_model = Path(r2.get("config", {}).get("model", ""))
            if alt_model.exists():
                _log(f"scenario '{args.scenario}' recommended "
                     f"{rec_model.name} (UNVERIFIED); falling back to verified "
                     f"config: {alt_model.name} ({r2['config']['precision']}, "
                     f"threads={r2['config']['threads']}, "
                     f"batch={r2['config']['batch']}, "
                     f"affinity={r2['config']['affinity']})")
                pkg_model = alt_model
                br["recommendations"] = {**br.get("recommendations", {}),
                                         args.scenario: r2}
                bench_rep.write_text(json.dumps(br, indent=2))

    if not args.no_package:
        from .package import build_package
        pkg_analyze = analyze_rep
        if pkg_model != primary:
            alt_analyze = workdir / f"{pkg_model.stem}-analyze.json"
            if alt_analyze.exists():
                pkg_analyze = alt_analyze
            elif analyze_rep.exists():
                _run("edgecore.analyze", ["--model", str(pkg_model),
                                          "--output", str(alt_analyze)])
                pkg_analyze = alt_analyze
        _log(f"building deployment bundle (scenario={args.scenario}, "
             f"model={pkg_model.name})")
        manifest = build_package(
            pkg_model, bench_rep, scenario=args.scenario,
            verify_report=verify_rep if verify_rep.exists() else None,
            analyze_report=pkg_analyze if pkg_analyze.exists() else None,
            runtime=runtime, out_dir=args.out)
        _log(f"package ready: {manifest.get('archive', str(Path(args.out)/manifest['package_name']))}")

    # ---- summary ------------------------------------------------------------
    print("\n================ EdgeCore e2e summary ================\n")
    print(f"model:            {pkg_model} ({pkg_model.stem})")
    print(f"hardware:         {hw.get('cpu_model')} | {hw.get('logical_cores')} "
          f"cores | {hw.get('ram_gb')} GB")
    print(f"verification:     {vr.get('overall', 'n/a').upper()}")
    if vr.get("models"):
        for m, r in vr["models"].items():
            s = r.get("summary", {})
            print(f"  {Path(m).name}: {r.get('precision'):>4} cosine={s.get('worst_cosine_mean', 0):.4f} "
                  f"top1={s.get('worst_top1', 0):.4f} ppl={s.get('max_ppl_ratio', 0):.4f} "
                  f"-> {'PASS' if s.get('pass') else 'FAIL'}")
    br = json.loads(bench_rep.read_text()) if bench_rep.exists() else {}
    rec = br.get("recommendations", {}).get(args.scenario, {})
    c = rec.get("config", {})
    m = rec.get("metrics", {})
    print(f"tuned config:     precision={c.get('precision')} threads={c.get('threads')} "
          f"batch={c.get('batch')} affinity={c.get('affinity')}")
    print(f"tuned metrics:    avg_token={m.get('avg_token_ms')}ms "
          f"tps={m.get('tokens_per_sec')} rss={m.get('peak_rss_gb')}GB")
    if not args.no_package:
        print(f"package:          {Path(args.out) / manifest['package_name']}")
    print("\n======================================================")
    return 0


# ----------------------------------------------------------------------
def _usage() -> str:
    cmds = ", ".join(list(_MODULES) + ["hw", "e2e"])
    return (f"usage: python -m edgecore.cli <command> [args...]\n"
            f"commands: {cmds}\n"
            f"  export|analyze|bench|verify|package   toolchain modules (see --help)\n"
            f"  hw [--hwprof PATH]                    print hardware profile\n"
            f"  e2e [options]                         end-to-end pipeline\n")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(_usage(), end="")
        return 2
    if argv[0] in ("-h", "--help"):
        print(_usage(), end="")
        return 0
    if argv[0] == "--version":
        print(f"EdgeCore {__version__}")
        return 0

    cmd, rest = argv[0], argv[1:]
    if cmd in _MODULES:
        return _run(_MODULES[cmd], rest)
    if cmd == "hw":
        ap = argparse.ArgumentParser(prog="python -m edgecore.cli hw")
        ap.add_argument("--hwprof", default=None)
        return _cmd_hw(ap.parse_args(rest))
    if cmd == "e2e":
        return _cmd_e2e(rest)
    print(f"error: unknown command '{cmd}'", file=sys.stderr)
    print(_usage(), end="")
    return 2


if __name__ == "__main__":
    sys.exit(main())
