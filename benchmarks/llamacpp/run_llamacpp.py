#!/usr/bin/env python3
"""
llama.cpp CPU benchmark for Qwen2.5-0.5B-Instruct.
Uses GGUF FP16 model.

MEASUREMENT METHODOLOGY (as currently implemented):
- TTFT: Measured by running a separate single-token generation (includes model load)
- TPOT: Calculated as (total_time - ttft) / (tokens - 1) -- NOTE: TTFT from separate run
- tok/s: Total tokens / total wall time (includes model load)
- RAM: Peak RSS of llama-cli subprocess (measured via psutil during run)
- All measurements are "as currently measured" and may not match other methodologies
"""

import json
import subprocess
import time
import os
import sys
import threading
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

def monitor_memory(pid, stop_event, max_rss):
    """Monitor memory of a process in a separate thread."""
    try:
        import psutil
        proc = psutil.Process(pid)
        while not stop_event.is_set():
            try:
                rss = proc.memory_info().rss / 1024 / 1024
                if rss > max_rss[0]:
                    max_rss[0] = rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(0.001)  # 1ms polling
    except ImportError:
        pass

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

    MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "baselines" / "qwen2.5-0.5b-instruct" / "qwen2.5-0.5b-instruct-fp16.gguf"
    LLAMA_CLI = Path(__file__).parent.parent.parent / "build" / "llama.cpp" / "build" / "bin" / "llama-cli"

    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        return 1

    if not LLAMA_CLI.exists():
        print(f"Error: llama-cli not found at {LLAMA_CLI}")
        print("Please build llama.cpp first.")
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
    RESULTS_FILE = RESULTS_DIR / "llamacpp.json"

    # Get llama.cpp version
    try:
        version_output = subprocess.run([str(LLAMA_CLI), "--version"], capture_output=True, text=True, timeout=10)
        llama_version = version_output.stdout.strip().split('\n')[0] if version_output.stdout else "unknown"
    except Exception:
        llama_version = "unknown"

    results = {
        "runtime": "llamacpp",
        "model": MODEL,
        "precision": PRECISION,
        "batch": BATCH,
        "context": CONTEXT,
        "generation_tokens": GENERATION_TOKENS,
        "hardware": "Intel i3-1115G4",
        "timestamp": TIMESTAMP,
        "software": {
            "llama_cpp": llama_version
        },
        "measurement_notes": {
            "ttft_method": "separate_single_token_run_includes_model_load",
            "tpot_method": "total_minus_separate_ttft_div_tokens_minus_1",
            "tok_per_sec_method": "tokens_div_total_wall_time_includes_model_load",
            "ram_method": "peak_rss_of_llama_cli_subprocess",
            "caveat": "TTFT includes model load; TPOT uses separate-run TTFT; tok/s includes load time; RAM is subprocess peak RSS"
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
        RAM_MB_MAX = 0

        for run in range(1, REPETITIONS + 1):
            print(f"  Run {run}/{REPETITIONS}...")

            # Warmup run (not measured) - loads model into memory
            if run == 1:
                print(f"    Warmup (model load)...")
                warmup_cmd = [
                    str(LLAMA_CLI),
                    "-m", str(MODEL_PATH),
                    "-t", str(THREADS),
                    "-n", "1",
                    "-c", str(CONTEXT),
                    "-b", str(BATCH),
                    "--temp", "0.0",
                    "-p", FORMATTED_PROMPT,
                    "--no-display-prompt",
                    "--simple-io"
                ]
                subprocess.run(warmup_cmd, capture_output=True, timeout=60)

            # Measured run: full generation
            cmd = [
                str(LLAMA_CLI),
                "-m", str(MODEL_PATH),
                "-t", str(THREADS),
                "-n", str(GENERATION_TOKENS),
                "-c", str(CONTEXT),
                "-b", str(BATCH),
                "--temp", "0.0",
                "--top-p", "1.0",
                "--top-k", "0",
                "--repeat-penalty", "1.0",
                "-p", FORMATTED_PROMPT,
                "--no-display-prompt",
                "--simple-io"
            ]

            # Start process and monitor memory
            max_rss = [0]
            stop_monitor = threading.Event()
            
            START_TIME = time.perf_counter()
            
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Start memory monitoring thread
            monitor_thread = threading.Thread(target=monitor_memory, args=(proc.pid, stop_monitor, max_rss))
            monitor_thread.start()
            
            try:
                stdout, stderr = proc.communicate(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                print(f"    Timeout!")
                stop_monitor.set()
                monitor_thread.join()
                continue
            
            END_TIME = time.perf_counter()
            TOTAL_MS = (END_TIME - START_TIME) * 1000
            
            # Stop memory monitoring
            stop_monitor.set()
            monitor_thread.join()
            
            if max_rss[0] > RAM_MB_MAX:
                RAM_MB_MAX = max_rss[0]

            OUTPUT = stdout.strip()

            # Estimate generated token count (use target as approximation since tokenize is unreliable)
            # For consistent comparison, use the target generation_tokens
            TOKEN_COUNT = GENERATION_TOKENS

            # Measure TTFT separately (with warm model now)
            TTFT_START = time.perf_counter()
            ttft_cmd = [
                str(LLAMA_CLI),
                "-m", str(MODEL_PATH),
                "-t", str(THREADS),
                "-n", "1",
                "-c", str(CONTEXT),
                "-b", str(BATCH),
                "--temp", "0.0",
                "-p", FORMATTED_PROMPT,
                "--no-display-prompt",
                "--simple-io"
            ]
            try:
                subprocess.run(ttft_cmd, capture_output=True, timeout=30)
            except subprocess.TimeoutExpired:
                pass
            TTFT_END = time.perf_counter()
            TTFT_MS = (TTFT_END - TTFT_START) * 1000

            # Calculate TPOT: (total - ttft) / (tokens - 1)
            # Note: TTFT measured separately with warm model, so this is approximate
            if TOKEN_COUNT > 1:
                TPOT_MS = (TOTAL_MS - TTFT_MS) / (TOKEN_COUNT - 1)
                # Clamp to non-negative
                if TPOT_MS < 0:
                    TPOT_MS = TOTAL_MS / TOKEN_COUNT
            else:
                TPOT_MS = TOTAL_MS

            TOK_PER_SEC = TOKEN_COUNT * 1000 / TOTAL_MS if TOTAL_MS > 0 else 0

            TTFT_TIMES.append(TTFT_MS)
            TPOT_TIMES.append(TPOT_MS)
            TOK_PER_SEC_TIMES.append(TOK_PER_SEC)

            print(f"    Total: {TOTAL_MS:.0f}ms, TTFT: {TTFT_MS:.0f}ms, TPOT: {TPOT_MS:.2f}ms, tok/s: {TOK_PER_SEC:.2f}, RAM: {max_rss[0]:.0f}MB")

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

        print(f"  Results: {AVG_TOK_SEC:.2f} tok/s, P95: {P95:.1f} ms, RAM: {RAM_MB_MAX:.0f} MB")

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