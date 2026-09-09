#!/usr/bin/env python3
"""
EdgeCore CPU benchmark for Qwen2.5-0.5B-Instruct.
Uses native .ecm model with FP16 precision.
"""

import json
import subprocess
import sys
import os
import tempfile
from pathlib import Path
from datetime import datetime

def load_config():
    config_path = Path(__file__).parent.parent / "common" / "benchmark_config.json"
    with open(config_path) as f:
        return json.load(f)

def load_prompts():
    prompt_path = Path(__file__).parent.parent / "common" / "prompts.json"
    with open(prompt_path) as f:
        return json.load(f)

def run_benchmark():
    config = load_config()
    prompts = load_prompts()
    
    MODEL = config["model"]
    GENERATION_TOKENS = config["generation_tokens"]
    THREADS_LIST = config["threads"]
    REPETITIONS = config["repetitions"]
    BATCH = config["batch"]
    CONTEXT = config["context"]
    PRECISION = config["precision"]
    
    FORMATTED_PROMPT = prompts["formatted_prompt"]
    
    MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "qwen2-fp16.ecm"
    RUNTIME = Path(__file__).parent.parent.parent / "build" / "edgecore-runtime"
    
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        return 1
    
    if not RUNTIME.exists():
        print(f"Error: Runtime not found at {RUNTIME}")
        print("Please build EdgeCore first: make")
        return 1
    
    print(f"Model: {MODEL}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Prompt: {FORMATTED_PROMPT}")
    print(f"Generation tokens: {GENERATION_TOKENS}")
    print(f"Threads: {THREADS_LIST}")
    print(f"Repetitions: {REPETITIONS}")
    
    # Build results JSON
    TIMESTAMP = datetime.utcnow().isoformat() + "Z"
    RESULTS_DIR = Path(__file__).parent.parent / "results" / "upgrade2"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE = RESULTS_DIR / "edgecore.json"
    
    results = {
        "runtime": "edgecore",
        "model": MODEL,
        "precision": PRECISION,
        "batch": BATCH,
        "context": CONTEXT,
        "generation_tokens": GENERATION_TOKENS,
        "hardware": "Intel i3-1115G4",
        "timestamp": TIMESTAMP,
        "software": {
            "edgecore": "v0.2"
        },
        "runs": []
    }
    
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    
    for THREADS in THREADS_LIST:
        print(f"\n=== Running with {THREADS} thread(s) ===")
        
        TTFT_TIMES = []
        TPOT_TIMES = []
        TOK_PER_SEC_TIMES = []
        P50_TIMES = []
        P95_TIMES = []
        P99_TIMES = []
        RAM_MB_MAX = 0
        
        for run in range(1, REPETITIONS + 1):
            print(f"  Run {run}/{REPETITIONS}...")
            
            # Run EdgeCore with JSON output to a temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                tmp_path = tmp.name
            
            try:
                cmd = [
                    str(RUNTIME),
                    "--model", str(MODEL_PATH),
                    "--mode", "generate",
                    "--prompt", FORMATTED_PROMPT,
                    "--n-tokens", str(GENERATION_TOKENS),
                    "--threads", str(THREADS),
                    "--batch", str(BATCH),
                    "--temp", "0.0",
                    "--output", tmp_path,
                    "--no-print"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                
                # Read JSON output
                with open(tmp_path) as f:
                    output = f.read()
                
                os.unlink(tmp_path)
                
                if not output.strip():
                    print(f"    Empty output")
                    continue
                
                # Parse JSON output
                data = json.loads(output)
                
                TTFT_MS = data.get("ttft_ms", 0)
                TPOT_AVG_MS = data.get("tpot_avg_ms", 0)
                TOK_PER_SEC = data.get("tokens_per_sec", 0)
                P50_MS = data.get("p50_ms", 0)
                P95_MS = data.get("p95_ms", 0)
                P99_MS = data.get("p99_ms", 0)
                PEAK_RSS_GB = data.get("peak_rss_gb", 0)
                PEAK_RSS_MB = int(PEAK_RSS_GB * 1024)
                
                TTFT_TIMES.append(TTFT_MS)
                TPOT_TIMES.append(TPOT_AVG_MS)
                TOK_PER_SEC_TIMES.append(TOK_PER_SEC)
                
                if PEAK_RSS_MB > RAM_MB_MAX:
                    RAM_MB_MAX = PEAK_RSS_MB
                    
            except subprocess.TimeoutExpired:
                print(f"    Timeout!")
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                continue
            except json.JSONDecodeError as e:
                print(f"    JSON parse error: {e}")
                print(f"    Output: {output[:500]}")
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                continue
            except Exception as e:
                print(f"    Error: {e}")
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                continue
        
        if not TPOT_TIMES:
            print("  No successful runs!")
            continue
        
        # Calculate percentiles
        SORTED_TPOT = sorted(TPOT_TIMES)
        COUNT = len(SORTED_TPOT)
        P50 = SORTED_TPOT[COUNT // 2]
        P95 = SORTED_TPOT[int(COUNT * 0.95)]
        P99 = SORTED_TPOT[int(COUNT * 0.99)]
        
        AVG_TTFT = sum(TTFT_TIMES) / len(TTFT_TIMES)
        AVG_TPOT = sum(TPOT_TIMES) / len(TPOT_TIMES)
        AVG_TOK_SEC = sum(TOK_PER_SEC_TIMES) / len(TOK_PER_SEC_TIMES)
        
        print(f"  Results: {AVG_TOK_SEC:.1f} tok/s, P95: {P95:.1f} ms, RAM: {RAM_MB_MAX} MB")
        
        # Append to results
        run_data = {
            "threads": THREADS,
            "ttft_ms": AVG_TTFT,
            "tpot_ms": AVG_TPOT,
            "tokens_per_second": AVG_TOK_SEC,
            "p50_ms": P50,
            "p95_ms": P95,
            "p99_ms": P99,
            "ram_mb": RAM_MB_MAX
        }
        
        # Read existing results and append
        with open(RESULTS_FILE, "r") as f:
            results = json.load(f)
        results["runs"].append(run_data)
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {RESULTS_FILE}")
    with open(RESULTS_FILE) as f:
        print(json.dumps(json.load(f), indent=2))
    
    return 0

if __name__ == "__main__":
    sys.exit(run_benchmark())