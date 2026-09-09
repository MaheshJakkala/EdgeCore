#!/usr/bin/env python3
"""
Unified comparison table generator for Upgrade 2 benchmarks.
Reads results from all three runtimes and produces a comparison table.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

def load_results(runtime_dir):
    """Load results from a runtime directory."""
    results_file = runtime_dir / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    return None

def format_table(results_dict):
    """Format results as a comparison table."""
    # Collect all thread configurations
    all_threads = set()
    for runtime, data in results_dict.items():
        for run in data.get("runs", []):
            all_threads.add(run.get("threads", 0))
    
    sorted_threads = sorted(all_threads)
    
    # Header
    header = f"{'Runtime':<12} {'Threads':<8} {'tok/s':<8} {'P95 (ms)':<10} {'RAM (MB)':<10} {'Status':<10}"
    separator = "-" * len(header)
    
    lines = [header, separator]
    
    for runtime in ["pytorch", "llamacpp", "edgecore"]:
        if runtime not in results_dict:
            for t in sorted_threads:
                lines.append(f"{runtime:<12} {t:<8} {'—':<8} {'—':<10} {'—':<10} {'Missing':<10}")
            continue
        
        data = results_dict[runtime]
        runs_by_thread = {r["threads"]: r for r in data.get("runs", [])}
        
        for t in sorted_threads:
            if t in runs_by_thread:
                r = runs_by_thread[t]
                tok_sec = r.get("tokens_per_second", 0)
                p95 = r.get("p95_ms", 0)
                ram = r.get("ram_mb", 0)
                
                # Determine status for EdgeCore
                if runtime == "edgecore":
                    if t == 2:
                        status = "Accept"
                    else:
                        status = "Reject"
                else:
                    status = "—"
                
                lines.append(f"{runtime:<12} {t:<8} {tok_sec:<8.1f} {p95:<10.1f} {ram:<10.0f} {status:<10}")
            else:
                lines.append(f"{runtime:<12} {t:<8} {'—':<8} {'—':<10} {'—':<10} {'—':<10}")
    
    return "\n".join(lines)

def main():
    results_dir = Path(__file__).parent.parent / "results" / "upgrade2"
    
    results_dict = {}
    
    for runtime in ["pytorch", "llamacpp", "edgecore"]:
        result_file = results_dir / f"{runtime}.json"
        if result_file.exists():
            with open(result_file) as f:
                results_dict[runtime] = json.load(f)
                print(f"Loaded {runtime} results")
        else:
            print(f"Warning: {runtime} results not found at {result_file}")
    
    if not results_dict:
        print("No results found. Run benchmarks first.")
        sys.exit(1)
    
    # Generate comparison table
    table = format_table(results_dict)
    print("\n" + "=" * 70)
    print("UPGRADE 2 BENCHMARK COMPARISON")
    print("=" * 70)
    print(f"Model: Qwen2.5-0.5B-Instruct")
    print(f"Hardware: Intel i3-1115G4")
    print(f"Precision: FP16")
    print(f"Batch: 1, Context: 2048, Generation: 32 tokens")
    print(f"Generated: {datetime.utcnow().isoformat()}Z")
    print("=" * 70)
    print(table)
    print("=" * 70)
    
    # Save table to file
    output_file = results_dir / "comparison_table.txt"
    with open(output_file, "w") as f:
        f.write("UPGRADE 2 BENCHMARK COMPARISON\n")
        f.write("=" * 70 + "\n")
        f.write(f"Model: Qwen2.5-0.5B-Instruct\n")
        f.write(f"Hardware: Intel i3-1115G4\n")
        f.write(f"Precision: FP16\n")
        f.write(f"Batch: 1, Context: 2048, Generation: 32 tokens\n")
        f.write(f"Generated: {datetime.utcnow().isoformat()}Z\n")
        f.write("=" * 70 + "\n")
        f.write(table + "\n")
        f.write("=" * 70 + "\n")
    
    print(f"\nTable saved to {output_file}")
    
    # Also save as JSON for programmatic access
    comparison = {
        "model": "Qwen2.5-0.5B-Instruct",
        "hardware": "Intel i3-1115G4",
        "precision": "fp16",
        "batch": 1,
        "context": 2048,
        "generation_tokens": 32,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "results": results_dict
    }
    
    json_file = results_dir / "comparison.json"
    with open(json_file, "w") as f:
        json.dump(comparison, f, indent=2)
    
    print(f"JSON saved to {json_file}")

if __name__ == "__main__":
    main()