"""EdgeCore - deployment packager.

Step 10: assembles a reproducible, self-contained deployment bundle for
offline / on-premise / air-gapped targets from the artifacts produced by
the earlier steps:

  model.ecm                quantized model artifact
  config.json              tuned execution config (from the bench report)
  bin/edgecore-runtime     compiled C runtime
  bin/run.sh               launch helper embedding the tuned config
  reports/                 analyze / benchmark / verification JSON
  hardware.json            hardware profile of the build environment
  manifest.json            provenance: hashes, versions, git commit

The bundle directory is written to ``dist/<name>-<scenario>/`` and also
packed as ``dist/<name>-<scenario>.tar.gz``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

from . import __version__
from .bench import default_runtime, default_hwprof, run_hw_profile
from .ecm import EcmReader

__all__ = ["build_package"]

SCENARIOS = ("low-latency-interactive", "high-throughput-batch",
             "low-memory-edge", "balanced")


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_commit(repo: str | Path) -> dict[str, Any]:
    repo = Path(repo)
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        return {"commit": sha, "dirty": bool(dirty)}
    except Exception:
        return {"commit": None, "dirty": None}


def _model_info(model: Path, analyze_report: dict[str, Any] | None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "artifact": str(model),
        "size_mb": round(model.stat().st_size / 1e6, 3),
        "sha256": sha256_file(model),
    }
    meta = EcmReader(model).meta
    info["model_name"] = meta.get("model_name", "unknown")
    info["n_layer"] = int(meta.get("n_layer", 0))
    info["n_head"] = int(meta.get("n_head", 0))
    info["n_embd"] = int(meta.get("n_embd", 0))
    info["n_ctx"] = int(meta.get("n_ctx", 0))
    info["vocab"] = int(meta.get("vocab", 0))
    info["params_total"] = int(meta.get("n_params", 0))
    info["precision"] = _detect_precision(model)
    # enrich with the analysis report only when it belongs to this artifact
    if analyze_report:
        same_artifact = str(analyze_report.get("artifact")) in (str(model),
                                                                model.name)
        if not same_artifact:
            analyze_report = None
    if analyze_report:
        for k in ("model_name", "n_layer", "n_head", "n_embd", "n_ctx",
                  "vocab", "params_total", "weight_mb"):
            if k in analyze_report and analyze_report[k] is not None:
                info[k] = analyze_report[k]
        if analyze_report.get("precision"):
            info["precision"] = analyze_report["precision"]
    return info


def _detect_precision(model: Path) -> str:
    try:
        from .ecm import first_linear_key
        r = EcmReader(model)
        w = r.tensor_meta.get(first_linear_key(r.meta))
        if not w:
            return "unknown"
        return {"int8": "int8", "fp16": "fp16", "bf16": "bf16"}.get(w["dtype"], w["dtype"])
    except Exception:
        return "unknown"


def build_package(model: str | Path, bench_report: str | Path,
                  scenario: str = "balanced",
                  verify_report: str | Path | None = None,
                  analyze_report: str | Path | None = None,
                  decision_report: str | Path | None = None,
                  runtime: str | Path | None = None,
                  hw_profile: dict[str, Any] | None = None,
                  name: str | None = None, out_dir: str | Path = "dist",
                  include_tar: bool = True) -> dict[str, Any]:
    """Assemble the deployment bundle; returns the manifest dict.
    
    If decision_report is provided, the package will only be created if
    the decision status is VERIFIED. This prevents deploying invalid
    configurations (NO_VALID_CONFIGURATION or FAILED).
    """
    model = Path(model)
    if not model.exists():
        raise FileNotFoundError(f"model not found: {model}")
    bench = json.loads(Path(bench_report).read_text())
    if scenario not in bench.get("recommendations", {}):
        raise ValueError(
            f"scenario '{scenario}' not in bench report "
            f"(have: {list(bench.get('recommendations', {}))})")

    rec = bench["recommendations"][scenario]
    cfg = rec["config"]
    metrics = rec.get("metrics", {})

    # Check decision report if provided
    if decision_report:
        decision = json.loads(Path(decision_report).read_text())
        if decision.get("status") != "VERIFIED":
            raise ValueError(
                f"Cannot create deployment package: decision status is "
                f"'{decision.get('status')}', not VERIFIED. "
                f"Deployment requires a VERIFIED decision record."
            )

    runtime = runtime or default_runtime()
    runtime = Path(runtime)
    if not runtime.exists():
        raise FileNotFoundError(f"runtime not found: {runtime} (run `make`)")

    hwprof = default_hwprof()
    if hw_profile is None:
        hw_profile = run_hw_profile(hwprof)

    ar = json.loads(Path(analyze_report).read_text()) if analyze_report else None
    vr = json.loads(Path(verify_report).read_text()) if verify_report else None

    base = (name or model.stem).strip().replace(" ", "-")
    bundle_dir = Path(out_dir) / f"{base}-{scenario}"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bin_dir = bundle_dir / "bin"
    rep_dir = bundle_dir / "reports"
    bin_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    import shutil as _shutil

    # model + tuned config
    _shutil.copy2(model, bundle_dir / "model.ecm")
    (bundle_dir / "config.json").write_text(json.dumps({
        "scenario": scenario,
        "description": rec.get("description"),
        "objectives": rec.get("objectives"),
        "config": cfg,
        "metrics": metrics,
    }, indent=2))

    # runtime + launcher
    _shutil.copy2(runtime, bin_dir / "edgecore-runtime")
    os.chmod(bin_dir / "edgecore-runtime", 0o755)
    run_sh = f"""#!/usr/bin/env bash
# EdgeCore runtime launcher - generated by edgecore.package
# Usage: bin/run.sh --mode generate --prompt "..." [runtime args...]
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
exec "$DIR/bin/edgecore-runtime" \\
  --model "$DIR/model.ecm" \\
  --threads {cfg.get("threads", 1)} \\
  --batch {cfg.get("batch", 1)} \\
  --affinity {cfg.get("affinity", "none")} \\
  "$@"
"""
    (bin_dir / "run.sh").write_text(run_sh)
    os.chmod(bin_dir / "run.sh", 0o755)

    # reports
    (rep_dir / "benchmark_report.json").write_text(json.dumps(bench, indent=2))
    if vr:
        (rep_dir / "verification_report.json").write_text(json.dumps(vr, indent=2))
    if ar:
        (rep_dir / "analysis_report.json").write_text(json.dumps(ar, indent=2))
    (bundle_dir / "hardware.json").write_text(json.dumps(hw_profile, indent=2))

    # provenance
    model_info = _model_info(model, ar)
    manifest: dict[str, Any] = {
        "package_name": f"{base}-{scenario}",
        "edgecore_version": __version__,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "toolchain": {
            "python": platform.python_version(),
            "numpy": _numpy_version(),
        },
        "git": git_commit(Path(__file__).resolve().parent.parent),
        "model": model_info,
        "runtime": {
            "binary": "bin/edgecore-runtime",
            "sha256": sha256_file(runtime),
        },
        "config": cfg,
        "scenario": {
            "name": scenario,
            "description": rec.get("description"),
            "objectives": rec.get("objectives"),
        },
        "hardware": {k: hw_profile.get(k) for k in
                     ("cpu_model", "physical_cores", "logical_cores",
                      "ram_gb", "simd", "memory_bandwidth_gbps",
                      "numa_nodes")},
        "benchmark": {
            "grid_size": bench.get("grid_size"),
            "scenario_metrics": metrics,
        },
        "verification": _verification_summary(vr, model=model),
        "deployment_decision": json.loads(Path(decision_report).read_text()) if decision_report else None,
        "files": _file_manifest(bundle_dir),
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if include_tar:
        tarpath = Path(out_dir) / f"{base}-{scenario}.tar.gz"
        with tarfile.open(tarpath, "w:gz") as tf:
            tf.add(bundle_dir, arcname=f"{base}-{scenario}")
        manifest["archive"] = str(tarpath)
        manifest["archive_sha256"] = sha256_file(tarpath)

    return manifest


def _numpy_version() -> str:
    try:
        import numpy as _np
        return _np.__version__
    except Exception:
        return "unknown"


def _verification_summary(vr: dict[str, Any] | None,
                          model: Path | None = None) -> dict[str, Any] | None:
    if not vr:
        return None
    out: dict[str, Any] = {"overall": vr.get("overall"), "thresholds": vr.get("thresholds")}
    out["models"] = {
        k: {"precision": v.get("precision"),
            "primary_reference": v.get("primary_reference"),
            "pass": v["summary"]["pass"]}
        for k, v in vr.get("models", {}).items()
    }
    if model is not None:
        key = str(model)
        entry = out["models"].get(key)
        if entry is None:
            for k, v in vr.get("models", {}).items():
                if Path(k).name == model.name:
                    entry = out["models"][k]
                    key = k
                    break
        if entry is not None:
            out["packaged_model"] = key
            out["packaged_pass"] = entry["pass"]
            out["overall"] = "pass" if entry["pass"] else "fail"
    return out


def _file_manifest(bundle_dir: Path) -> list[dict[str, Any]]:
    files = []
    for p in sorted(bundle_dir.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(bundle_dir))
            files.append({"path": rel, "bytes": p.stat().st_size,
                          "sha256": sha256_file(p)})
    return files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Assemble an offline deployment bundle",
        prog="python -m edgecore.package")
    ap.add_argument("--model", required=True, help=".ecm artifact to deploy")
    ap.add_argument("--bench-report", required=True,
                    help="JSON report from edgecore.bench (contains tuned config)")
    ap.add_argument("--scenario", default="balanced", choices=list(SCENARIOS))
    ap.add_argument("--verify-report", default=None,
                    help="optional JSON report from edgecore.verify")
    ap.add_argument("--analyze-report", default=None,
                    help="optional JSON report from edgecore.analyze")
    ap.add_argument("--decision-report", default=None,
                    help="optional JSON report from edgecore.decide (required for VERIFIED package)")
    ap.add_argument("--runtime", default=None, help="path to edgecore-runtime")
    ap.add_argument("--hw-profile", default=None,
                    help="optional hardware JSON (else hwprof is run)")
    ap.add_argument("--name", default=None, help="bundle base name")
    ap.add_argument("--out", default="dist", help="output directory (default dist)")
    ap.add_argument("--no-tar", action="store_true",
                    help="write the bundle directory only")
    args = ap.parse_args(argv)

    hw_profile = None
    if args.hw_profile:
        hw_profile = json.loads(Path(args.hw_profile).read_text())

    manifest = build_package(
        args.model, args.bench_report, scenario=args.scenario,
        verify_report=args.verify_report, analyze_report=args.analyze_report,
        decision_report=args.decision_report,
        runtime=Path(args.runtime) if args.runtime else None,
        hw_profile=hw_profile,
        name=args.name, out_dir=args.out, include_tar=not args.no_tar,
    )
    print(json.dumps(manifest, indent=2))
    if manifest.get("archive"):
        print(f"\npackage written to {manifest['archive']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
