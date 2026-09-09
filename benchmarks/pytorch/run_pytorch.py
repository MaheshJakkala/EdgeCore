#!/usr/bin/env python3
"""
PyTorch CPU benchmark for Qwen2.5-0.5B-Instruct.
Uses Hugging Face transformers with FP16 precision on CPU.
"""

import json
import time
import torch
import gc
import psutil
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Error: transformers not installed. Run: pip install transformers torch accelerate")
    sys.exit(1)

def load_config():
    config_path = Path(__file__).parent.parent / "common" / "benchmark_config.json"
    with open(config_path) as f:
        return json.load(f)

def load_prompts():
    prompt_path = Path(__file__).parent.parent / "common" / "prompts.json"
    with open(prompt_path) as f:
        return json.load(f)

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def run_benchmark(model, tokenizer, prompt, generation_tokens, threads):
    torch.set_num_threads(threads)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"]
    prompt_tokens = input_ids.shape[1]
    
    # Warmup
    with torch.no_grad():
        for _ in range(2):
            _ = model.generate(
                input_ids,
                max_new_tokens=generation_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
    
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    # Benchmark runs
    ttft_times = []
    tpot_times = []
    tokens_per_sec_list = []
    peak_rams = []
    
    for run in range(config["repetitions"]):
        gc.collect()
        start_mem = get_memory_mb()
        
        # Measure TTFT (time to first token)
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=generation_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True
            )
        end = time.perf_counter()
        
        peak_mem = get_memory_mb()
        peak_rams.append(peak_mem - start_mem)
        
        total_time = (end - start) * 1000  # ms
        generated_tokens = outputs.sequences.shape[1] - prompt_tokens
        
        # For TTFT, we need to measure time to first generated token
        # We'll approximate by running single-token generation
        ttft_start = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(
                input_ids,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        ttft_ms = (time.perf_counter() - ttft_start) * 1000
        ttft_times.append(ttft_ms)
        
        # TPOT = (total_time - ttft) / (generated_tokens - 1)
        if generated_tokens > 1:
            tpot = (total_time - ttft_ms) / (generated_tokens - 1)
        else:
            tpot = total_time
        tpot_times.append(tpot)
        
        tokens_per_sec = (generated_tokens / total_time) * 1000
        tokens_per_sec_list.append(tokens_per_sec)
    
    return {
        "ttft_ms": sum(ttft_times) / len(ttft_times),
        "tpot_ms": sum(tpot_times) / len(tpot_times),
        "tokens_per_second": sum(tokens_per_sec_list) / len(tokens_per_sec_list),
        "p50_ms": sorted(tpot_times)[len(tpot_times) // 2],
        "p95_ms": sorted(tpot_times)[int(len(tpot_times) * 0.95)],
        "p99_ms": sorted(tpot_times)[int(len(tpot_times) * 0.99)],
        "ram_mb": max(peak_rams)
    }

def main():
    global config
    config = load_config()
    prompts = load_prompts()
    
    prompt = prompts["formatted_prompt"]
    generation_tokens = config["generation_tokens"]
    threads_list = config["threads"]
    
    print(f"Loading Qwen2.5-0.5B-Instruct model...")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    model.eval()
    
    print(f"Model loaded. Prompt tokens: {len(tokenizer.encode(prompt, add_special_tokens=False))}")
    
    results = {
        "runtime": "pytorch",
        "model": config["model"],
        "precision": config["precision"],
        "batch": config["batch"],
        "context": config["context"],
        "generation_tokens": config["generation_tokens"],
        "hardware": "Intel i3-1115G4",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "software": {
            "pytorch": torch.__version__,
            "transformers": "4.40+"
        },
        "runs": []
    }
    
    for threads in threads_list:
        print(f"\nRunning benchmark with {threads} thread(s)...")
        result = run_benchmark(model, tokenizer, prompt, generation_tokens, threads)
        result["threads"] = threads
        results["runs"].append(result)
        print(f"  {threads}T: {result['tokens_per_second']:.1f} tok/s, P95: {result['p95_ms']:.1f} ms, RAM: {result['ram_mb']:.0f} MB")
    
    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "upgrade2"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "pytorch.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()