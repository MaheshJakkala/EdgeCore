#!/usr/bin/env python3
"""
EdgeCore - Deployment Decision Engine (Upgrade 3)

Implements explicit, deterministic deployment decisions with:
- Constraint filtering (P95 < X ms, accuracy loss < Y%, RAM < Z GB)
- Rejection reasons for each failed candidate
- "Best measured" vs "Best valid" distinction
- NO_VALID_CONFIGURATION status when no candidate passes
- Decision records for auditability
- Integration with tokenizer verification
"""

from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
from datetime import datetime


@dataclass
class Constraint:
    """A single deployment constraint."""
    name: str
    limit: float
    unit: str
    higher_is_better: bool  # e.g., throughput (higher better), latency (lower better)


@dataclass
class Candidate:
    """A deployment candidate with its metrics."""
    precision: str
    threads: int
    batch: int
    affinity: str = "none"
    
    # Measured metrics
    tokens_per_second: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    ttft_ms: float = 0.0
    ram_mb: float = 0.0
    accuracy_loss_percent: float = 0.0
    
    # Status
    tokenizer_verified: bool = False
    correctness_verified: bool = False


@dataclass
class RejectionReason:
    """Why a candidate was rejected."""
    constraint: str
    measured: float
    limit: float
    unit: str
    
    def __str__(self) -> str:
        return f"REJECTED: {self.constraint} ({self.measured:.2f} {self.unit} vs limit {self.limit:.2f} {self.unit})"


@dataclass
class DecisionRecord:
    """Complete deployment decision record."""
    status: str  # VERIFIED, NO_VALID_CONFIGURATION, FAILED
    constraints: dict[str, Any]
    candidates_evaluated: int
    valid_candidates: int
    best_measured: dict[str, Any] | None
    best_valid: dict[str, Any] | None
    candidates: list[dict[str, Any]]
    timestamp: str
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# Default constraints from the proposal
DEFAULT_CONSTRAINTS = {
    "max_p95_ms": Constraint("max_p95_ms", 100.0, "ms", False),
    "max_accuracy_loss_percent": Constraint("max_accuracy_loss_percent", 1.0, "%", False),
    "max_ram_mb": Constraint("max_ram_mb", 6144.0, "MB", False),  # 6 GB
}

# Minimum required checks
REQUIRED_CHECKS = ["tokenizer_verified", "correctness_verified"]


def evaluate_candidate(candidate: Candidate, 
                       constraints: dict[str, Constraint]) -> tuple[bool, list[RejectionReason]]:
    """
    Evaluate a candidate against constraints.
    Returns (passes_all, list_of_rejection_reasons).
    """
    reasons = []
    
    # Check required checks
    if not candidate.tokenizer_verified:
        reasons.append(RejectionReason(
            constraint="tokenizer_verified",
            measured=0.0, limit=1.0, unit="bool"
        ))
    if not candidate.correctness_verified:
        reasons.append(RejectionReason(
            constraint="correctness_verified",
            measured=0.0, limit=1.0, unit="bool"
        ))
    
    # Check constraints - map constraint names to candidate attributes
    attr_map = {
        "max_p95_ms": "p95_ms",
        "max_accuracy_loss_percent": "accuracy_loss_percent",
        "max_ram_mb": "ram_mb",
    }
    
    for name, constraint in constraints.items():
        attr = attr_map.get(name, name)
        measured = getattr(candidate, attr, None)
        if measured is None:
            continue
            
        passes = (measured <= constraint.limit) if not constraint.higher_is_better \
                 else (measured >= constraint.limit)
        
        if not passes:
            reasons.append(RejectionReason(
                constraint=name,
                measured=measured,
                limit=constraint.limit,
                unit=constraint.unit
            ))
    
    return len(reasons) == 0, reasons


def make_decision(candidates: list[Candidate],
                  constraints: dict[str, Constraint] = DEFAULT_CONSTRAINTS) -> DecisionRecord:
    """
    Make a deployment decision from evaluated candidates.
    
    Logic:
    1. Filter valid candidates (pass all constraints + required checks)
    2. If no valid candidates: NO_VALID_CONFIGURATION
    3. If valid candidates: rank by composite score, pick best valid
    4. Also report best measured (regardless of validity)
    """
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    candidate_records = []
    valid_candidates = []
    
    for c in candidates:
        passes, reasons = evaluate_candidate(c, constraints)
        
        record = {
            "precision": c.precision,
            "threads": c.threads,
            "batch": c.batch,
            "affinity": c.affinity,
            "metrics": {
                "tokens_per_second": c.tokens_per_second,
                "p95_ms": c.p95_ms,
                "p99_ms": c.p99_ms,
                "ttft_ms": c.ttft_ms,
                "ram_mb": c.ram_mb,
                "accuracy_loss_percent": c.accuracy_loss_percent,
            },
            "checks": {
                "tokenizer_verified": c.tokenizer_verified,
                "correctness_verified": c.correctness_verified,
            },
            "valid": passes,
            "rejection_reasons": [str(r) for r in reasons] if reasons else []
        }
        candidate_records.append(record)
        
        if passes:
            valid_candidates.append(c)
    
    # Best measured (regardless of validity) - highest throughput
    best_measured = max(candidates, key=lambda c: c.tokens_per_second) if candidates else None
    best_measured_record = {
        "precision": best_measured.precision,
        "threads": best_measured.threads,
        "batch": best_measured.batch,
        "affinity": best_measured.affinity,
        "metrics": {
            "tokens_per_second": best_measured.tokens_per_second,
            "p95_ms": best_measured.p95_ms,
            "p99_ms": best_measured.p99_ms,
            "ttft_ms": best_measured.ttft_ms,
            "ram_mb": best_measured.ram_mb,
            "accuracy_loss_percent": best_measured.accuracy_loss_percent,
        },
        "valid": best_measured in valid_candidates
    } if best_measured else None
    
    # Best valid - rank valid candidates by composite score
    best_valid = None
    if valid_candidates:
        # Composite score: higher throughput, lower latency, lower memory
        # Normalize and weight
        best_valid = min(valid_candidates, key=lambda c: (
            -c.tokens_per_second * 0.5 +  # throughput (higher better)
            c.p95_ms * 0.3 +              # latency (lower better)
            c.ram_mb / 1024 * 0.2         # memory (lower better)
        ))
        best_valid_record = {
            "precision": best_valid.precision,
            "threads": best_valid.threads,
            "batch": best_valid.batch,
            "affinity": best_valid.affinity,
            "metrics": {
                "tokens_per_second": best_valid.tokens_per_second,
                "p95_ms": best_valid.p95_ms,
                "p99_ms": best_valid.p99_ms,
                "ttft_ms": best_valid.ttft_ms,
                "ram_mb": best_valid.ram_mb,
                "accuracy_loss_percent": best_valid.accuracy_loss_percent,
            }
        }
    else:
        best_valid_record = None
    
    # Determine overall status
    if not valid_candidates:
        status = "NO_VALID_CONFIGURATION"
    elif len(valid_candidates) > 0:
        status = "VERIFIED"
    else:
        status = "FAILED"
    
    # Constraints dict for output
    constraints_dict = {name: {"limit": c.limit, "unit": c.unit, "higher_is_better": c.higher_is_better}
                        for name, c in constraints.items()}
    
    return DecisionRecord(
        status=status,
        constraints=constraints_dict,
        candidates_evaluated=len(candidates),
        valid_candidates=len(valid_candidates),
        best_measured=best_measured_record,
        best_valid=best_valid_record,
        candidates=candidate_records,
        timestamp=timestamp
    )


def format_decision_cli(decision: DecisionRecord) -> str:
    """Format decision for CLI output."""
    lines = []
    lines.append("=" * 60)
    lines.append("EDGECORE DEPLOYMENT DECISION")
    lines.append("=" * 60)
    lines.append("")
    
    lines.append("CONSTRAINTS")
    for name, info in decision.constraints.items():
        direction = ">=" if info["higher_is_better"] else "<="
        lines.append(f"  {name} {direction} {info['limit']} {info['unit']}")
    lines.append("")
    
    lines.append("CANDIDATES")
    for c in decision.candidates:
        status = "VALID" if c["valid"] else "REJECTED"
        lines.append(f"  {c['precision']} / {c['threads']}T / batch={c['batch']} / {c['affinity']}")
        lines.append(f"    {status}")
        if not c["valid"]:
            for reason in c["rejection_reasons"]:
                lines.append(f"    {reason}")
        lines.append("")
    
    lines.append("DECISION")
    if decision.status == "NO_VALID_CONFIGURATION":
        lines.append("  No valid configuration found.")
        lines.append("")
        if decision.best_measured:
            bm = decision.best_measured
            lines.append("  BEST MEASURED (INVALID)")
            lines.append(f"    {bm['precision']} / {bm['threads']}T / batch={bm['batch']}")
            lines.append(f"    {bm['metrics']['tokens_per_second']:.1f} tok/s")
            lines.append(f"    P95: {bm['metrics']['p95_ms']:.1f} ms")
            lines.append(f"    RAM: {bm['metrics']['ram_mb']:.0f} MB")
    elif decision.status == "VERIFIED":
        lines.append("  Valid configuration found.")
        lines.append("")
        if decision.best_valid:
            bv = decision.best_valid
            lines.append("  RECOMMENDED")
            lines.append(f"    {bv['precision']} / {bv['threads']}T / batch={bv['batch']}")
            lines.append(f"    {bv['metrics']['tokens_per_second']:.1f} tok/s")
            lines.append(f"    P95: {bv['metrics']['p95_ms']:.1f} ms")
            lines.append(f"    RAM: {bv['metrics']['ram_mb']:.0f} MB")
    
    lines.append("")
    lines.append(f"STATUS: {decision.status}")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def save_decision_record(decision: DecisionRecord, output_path: Path) -> None:
    """Save decision record to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(decision.to_json())


def load_candidates_from_bench_report(bench_report_path: Path) -> list[Candidate]:
    """Load candidates from a benchmark report JSON."""
    with open(bench_report_path) as f:
        bench = json.load(f)
    
    candidates = []
    
    # Handle both formats: edgecore.json has "runs", bench.py report has "measurements"
    runs = bench.get("runs", bench.get("measurements", []))
    
    # Get common metadata
    precision = bench.get("precision", "unknown")
    batch = bench.get("batch", 1)
    
    for m in runs:
        if "error" in m:
            continue
        c = Candidate(
            precision=precision,
            threads=m.get("threads", 1),
            batch=batch,
            affinity="none",  # Not in current format
            tokens_per_second=m.get("tokens_per_second", 0.0),
            p95_ms=m.get("p95_ms", 0.0),
            p99_ms=m.get("p99_ms", 0.0),
            ttft_ms=m.get("ttft_ms", 0.0),
            ram_mb=m.get("ram_mb", 0.0),
            accuracy_loss_percent=0.0,  # Would need separate accuracy eval
        )
        candidates.append(c)
    return candidates


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for deployment decision."""
    import argparse
    ap = argparse.ArgumentParser(
        description="EdgeCore Deployment Decision Engine",
        prog="python -m edgecore.decide")
    ap.add_argument("--bench-report", required=True, help="Benchmark report JSON")
    ap.add_argument("--verify-report", help="Verification report JSON")
    ap.add_argument("--tokenizer-report", help="Tokenizer verification report JSON")
    ap.add_argument("--constraints", help="Constraints JSON (optional)")
    ap.add_argument("--output", help="Output decision record JSON")
    ap.add_argument("--format", choices=["cli", "json"], default="cli")
    args = ap.parse_args(argv)
    
    # Load candidates from bench report
    candidates = load_candidates_from_bench_report(Path(args.bench_report))
    
    # Load verification results
    tokenizer_verified = False
    correctness_verified = False
    
    if args.tokenizer_report:
        with open(args.tokenizer_report) as f:
            vr = json.load(f)
            tokenizer_verified = vr.get("overall", {}).get("status") == "VERIFIED"
    
    if args.verify_report:
        with open(args.verify_report) as f:
            vr = json.load(f)
            correctness_verified = vr.get("overall") == "pass"
    
    # Apply verification flags to all candidates
    for c in candidates:
        c.tokenizer_verified = tokenizer_verified
        c.correctness_verified = correctness_verified
    
    # Load custom constraints if provided
    constraints = DEFAULT_CONSTRAINTS.copy()
    if args.constraints:
        with open(args.constraints) as f:
            custom = json.load(f)
            for name, info in custom.items():
                constraints[name] = Constraint(
                    name=name,
                    limit=info["limit"],
                    unit=info.get("unit", ""),
                    higher_is_better=info.get("higher_is_better", False)
                )
    
    # Make decision
    decision = make_decision(candidates, constraints)
    
    # Output
    if args.format == "json":
        output = decision.to_json()
    else:
        output = format_decision_cli(decision)
    
    if args.output:
        Path(args.output).write_text(output)
        print(f"Decision record written to {args.output}")
    else:
        print(output)
    
    # Exit code based on status
    if decision.status == "VERIFIED":
        return 0
    elif decision.status == "NO_VALID_CONFIGURATION":
        return 1
    else:
        return 2


if __name__ == "__main__":
    import sys
    sys.exit(main())