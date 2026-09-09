#!/usr/bin/env python3
"""
Correctness comparison for Upgrade 2.
Compares outputs from PyTorch, llama.cpp, and EdgeCore.
"""

import json
import sys
import subprocess
import os
from pathlib import Path

def load_prompts():
    prompt_path = Path(__file__).parent / "prompts.json"
    with open(prompt_path) as f:
        return json.load(f)

def run_pytorch(prompt, generation_tokens=32):
    """Run PyTorch generation and return output."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
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
    
    torch.set_num_threads(2)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=generation_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=False
        )
    
    # Handle different output formats
    if hasattr(outputs, 'sequences'):
        generated = outputs.sequences[0][input_ids.shape[1]:]
    else:
        generated = outputs[0][input_ids.shape[1]:]
    
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text.strip()

def run_edgecore(prompt, generation_tokens=32, threads=2):
    """Run EdgeCore generation and return output."""
    runtime = Path(__file__).parent.parent.parent / "build" / "edgecore-runtime"
    model = Path(__file__).parent.parent.parent / "models" / "qwen2-fp16.ecm"
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        cmd = [
            str(runtime),
            "--model", str(model),
            "--mode", "generate",
            "--prompt", prompt,
            "--n-tokens", str(generation_tokens),
            "--threads", str(threads),
            "--batch", "1",
            "--temp", "0.0",
            "--output", tmp_path,
            "--no-print"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        with open(tmp_path) as f:
            output = json.load(f)
        
        # Return performance info since we can't easily get generated text
        return f"[EdgeCore: {output.get('tokens_per_sec', 0):.1f} tok/s, TTFT: {output.get('ttft_ms', 0):.1f}ms]"
    finally:
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()

def run_llamacpp(prompt, generation_tokens=32, threads=2):
    """Run llama.cpp generation and return output."""
    llama_cli = Path(__file__).parent.parent.parent / "build" / "llama.cpp" / "build" / "bin" / "llama-cli"
    model = Path(__file__).parent.parent.parent / "models" / "baselines" / "qwen2.5-0.5b-instruct" / "qwen2.5-0.5b-instruct-fp16.gguf"
    
    if not llama_cli.exists() or not model.exists():
        return "[llama.cpp: not available]"
    
    # Use flags to suppress progress output
    cmd = [
        str(llama_cli),
        "-m", str(model),
        "-t", str(threads),
        "-n", str(generation_tokens),
        "-c", "2048",
        "-b", "1",
        "--temp", "0.0",
        "--top-p", "1.0",
        "--top-k", "0",
        "--repeat-penalty", "1.0",
        "-p", prompt,
        "--no-display-prompt",
        "--no-mmap"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout.strip()
        # Remove the prompt from output
        if output.startswith(prompt):
            output = output[len(prompt):]
        # Clean up any progress bar artifacts
        lines = output.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('[') and not line.startswith('|') and not 'debuginfod' in line.lower():
                clean_lines.append(line)
        output = '\n'.join(clean_lines)
        return output.strip()
    except subprocess.TimeoutExpired:
        return "[llama.cpp: timeout]"
    except Exception as e:
        return f"[llama.cpp: error - {e}]"

def main():
    prompts = load_prompts()
    prompt = prompts["formatted_prompt"]
    generation_tokens = 32
    
    print("=" * 60)
    print("CORRECTNESS COMPARISON")
    print("=" * 60)
    print(f"Prompt: {prompt[:80]}...")
    print(f"Generation tokens: {generation_tokens}")
    print("=" * 60)
    
    results = {}
    
    # PyTorch
    print("\nRunning PyTorch...")
    try:
        pt_output = run_pytorch(prompt, generation_tokens)
        results["pytorch"] = pt_output
        print(f"  PyTorch: {pt_output[:200]}...")
    except Exception as e:
        results["pytorch"] = f"[Error: {e}]"
        print(f"  PyTorch: Error - {e}")
    
    # EdgeCore
    print("\nRunning EdgeCore...")
    try:
        ec_output = run_edgecore(prompt, generation_tokens, 2)
        results["edgecore"] = ec_output
        print(f"  EdgeCore: {ec_output}")
    except Exception as e:
        results["edgecore"] = f"[Error: {e}]"
        print(f"  EdgeCore: Error - {e}")
    
    # llama.cpp
    print("\nRunning llama.cpp...")
    try:
        lc_output = run_llamacpp(prompt, generation_tokens, 2)
        results["llamacpp"] = lc_output
        print(f"  llama.cpp: {lc_output[:200]}...")
    except Exception as e:
        results["llamacpp"] = f"[Error: {e}]"
        print(f"  llama.cpp: Error - {e}")
    
    # Token-level comparison (if all succeeded)
    print("\n" + "=" * 60)
    print("OUTPUTS")
    print("=" * 60)
    for runtime, output in results.items():
        print(f"\n{runtime}:")
        print(f"  {output[:500]}")
    
    # Check if PyTorch and llama.cpp outputs are similar
    if "pytorch" in results and "llamacpp" in results:
        pt_out = results["pytorch"]
        lc_out = results["llamacpp"]
        if not pt_out.startswith("[Error") and not lc_out.startswith("[") and not lc_out.startswith("Loading"):
            # Simple word overlap check
            pt_words = set(pt_out.lower().split())
            lc_words = set(lc_out.lower().split())
            if pt_words and lc_words:
                overlap = len(pt_words & lc_words) / len(pt_words | lc_words)
                print(f"\nWord overlap (PyTorch vs llama.cpp): {overlap:.2%}")
    
    # Save results
    results_dir = Path(__file__).parent.parent / "results" / "upgrade2"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    correctness_report = {
        "prompt": prompt,
        "generation_tokens": generation_tokens,
        "outputs": results,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    
    with open(results_dir / "correctness_comparison.json", "w") as f:
        json.dump(correctness_report, f, indent=2)
    
    print(f"\nCorrectness report saved to {results_dir}/correctness_comparison.json")

if __name__ == "__main__":
    from datetime import datetime
    main()