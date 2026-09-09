#!/usr/bin/env python3
"""
EdgeCore Tokenizer Verification - Upgrade 3
Compares EdgeCore tokenizer against Hugging Face reference tokenizer.

Three checks:
A. Vocabulary/tokenizer metadata
B. Tokenization agreement (plain text)
C. Chat-template agreement (Qwen Instruct)
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Any

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Error: transformers not installed. Run: pip install transformers")
    sys.exit(1)

from .tokenizer import ByteLevelBPE
from .chat_template import apply_qwen_chat_template


def load_hf_tokenizer(hf_dir: Path) -> AutoTokenizer:
    """Load Hugging Face tokenizer as reference."""
    return AutoTokenizer.from_pretrained(
        str(hf_dir),
        trust_remote_code=True,
        use_fast=True
    )


def load_edgecore_tokenizer(hf_dir: Path) -> ByteLevelBPE:
    """Load EdgeCore tokenizer from HF directory."""
    return ByteLevelBPE(hf_dir=hf_dir, variant=1)


def check_vocabulary_metadata(hf_tok: AutoTokenizer, ec_tok: ByteLevelBPE) -> dict[str, Any]:
    """Check A: Vocabulary/tokenizer metadata."""
    results = {}
    
    # Vocab size
    hf_vocab_size = len(hf_tok.get_vocab())
    ec_vocab_size = len(ec_tok.encoder)
    results["vocab_size"] = {
        "hf": hf_vocab_size,
        "edgecore": ec_vocab_size,
        "match": hf_vocab_size == ec_vocab_size
    }
    
    # Special tokens - compare added_tokens_decoder (token -> id) with EdgeCore's special_tokens
    hf_special = {}
    if hasattr(hf_tok, 'added_tokens_decoder'):
        for tid, info in hf_tok.added_tokens_decoder.items():
            # AddedToken objects have a .content attribute
            content = getattr(info, 'content', None)
            if content is None and isinstance(info, dict):
                content = info.get('content')
            if isinstance(content, str):
                hf_special[content] = int(tid)
    
    ec_special = dict(ec_tok.special_tokens)  # Already token -> id
    
    results["special_tokens"] = {
        "hf": hf_special,
        "edgecore": ec_special,
        "match": hf_special == ec_special
    }
    
    # Key special token IDs
    key_tokens = ["eos_token", "bos_token", "pad_token", "unk_token"]
    special_ids = {}
    for key in key_tokens:
        hf_id = getattr(hf_tok, key, None)
        if hf_id is not None and hasattr(hf_id, '__int__'):
            hf_id = int(hf_id)
        special_ids[key] = {
            "hf": hf_id,
            "edgecore": ec_tok.encoder.get(hf_id) if hf_id is not None else None
        }
    results["key_special_token_ids"] = special_ids
    
    return results


def check_tokenization_agreement(hf_tok: AutoTokenizer, ec_tok: ByteLevelBPE, 
                                  test_cases: list[dict]) -> dict[str, Any]:
    """Check B: Tokenization agreement on plain text."""
    results = {
        "tests": [],
        "total": 0,
        "passed": 0
    }
    
    for case in test_cases:
        if "messages" in case:
            # Skip chat template cases here
            continue
            
        text = case["input"]
        name = case["name"]
        
        # HF tokenizer
        hf_ids = hf_tok.encode(text, add_special_tokens=False)
        
        # EdgeCore tokenizer
        ec_ids = ec_tok.encode(text)
        
        match = hf_ids == ec_ids
        
        test_result = {
            "name": name,
            "input": text,
            "hf_ids": hf_ids,
            "ec_ids": ec_ids,
            "hf_count": len(hf_ids),
            "ec_count": len(ec_ids),
            "match": match
        }
        
        if not match:
            # Find first mismatch
            min_len = min(len(hf_ids), len(ec_ids))
            mismatch_pos = None
            for i in range(min_len):
                if hf_ids[i] != ec_ids[i]:
                    mismatch_pos = i
                    break
            if mismatch_pos is None:
                mismatch_pos = min_len
            
            test_result["mismatch"] = {
                "position": mismatch_pos,
                "expected": hf_ids[mismatch_pos] if mismatch_pos < len(hf_ids) else None,
                "actual": ec_ids[mismatch_pos] if mismatch_pos < len(ec_ids) else None
            }
        
        results["tests"].append(test_result)
        results["total"] += 1
        if match:
            results["passed"] += 1
    
    return results


def check_chat_template_agreement(hf_tok: AutoTokenizer, ec_tok: ByteLevelBPE,
                                   test_cases: list[dict]) -> dict[str, Any]:
    """Check C: Chat-template agreement for Qwen Instruct."""
    results = {
        "tests": [],
        "total": 0,
        "passed": 0
    }
    
    for case in test_cases:
        if "messages" not in case:
            continue
            
        name = case["name"]
        messages = case["messages"]
        add_gen_prompt = case.get("add_generation_prompt", True)
        
        # HF chat template
        hf_formatted = hf_tok.apply_chat_template(
            messages,
            add_generation_prompt=add_gen_prompt,
            tokenize=False
        )
        hf_ids = hf_tok.encode(hf_formatted, add_special_tokens=False)
        
        # EdgeCore chat template
        ec_formatted = apply_qwen_chat_template(messages, add_generation_prompt=add_gen_prompt)
        ec_ids = ec_tok.encode(ec_formatted)
        
        match = hf_ids == ec_ids
        
        test_result = {
            "name": name,
            "messages": messages,
            "hf_formatted": hf_formatted,
            "ec_formatted": ec_formatted,
            "hf_ids": hf_ids,
            "ec_ids": ec_ids,
            "hf_count": len(hf_ids),
            "ec_count": len(ec_ids),
            "match": match
        }
        
        if not match:
            min_len = min(len(hf_ids), len(ec_ids))
            mismatch_pos = None
            for i in range(min_len):
                if hf_ids[i] != ec_ids[i]:
                    mismatch_pos = i
                    break
            if mismatch_pos is None:
                mismatch_pos = min_len
            
            test_result["mismatch"] = {
                "position": mismatch_pos,
                "expected": hf_ids[mismatch_pos] if mismatch_pos < len(hf_ids) else None,
                "actual": ec_ids[mismatch_pos] if mismatch_pos < len(ec_ids) else None
            }
        
        results["tests"].append(test_result)
        results["total"] += 1
        if match:
            results["passed"] += 1
    
    return results


def generate_golden_file(hf_tok: AutoTokenizer, test_cases: list[dict], 
                         output_path: Path) -> dict[str, Any]:
    """Generate golden reference file with HF token IDs."""
    golden = {
        "model": "Qwen2.5-0.5B-Instruct",
        "tokenizer": "Qwen2Tokenizer",
        "vocab_size": len(hf_tok.get_vocab()),
        "tests": []
    }
    
    for case in test_cases:
        if "messages" in case:
            # Chat template case
            messages = case["messages"]
            add_gen_prompt = case.get("add_generation_prompt", True)
            formatted = hf_tok.apply_chat_template(
                messages,
                add_generation_prompt=add_gen_prompt,
                tokenize=False
            )
            ids = hf_tok.encode(formatted, add_special_tokens=False)
            golden["tests"].append({
                "name": case["name"],
                "type": "chat_template",
                "messages": messages,
                "formatted": formatted,
                "reference_ids": ids
            })
        else:
            # Plain text case
            text = case["input"]
            ids = hf_tok.encode(text, add_special_tokens=False)
            golden["tests"].append({
                "name": case["name"],
                "type": "plain_text",
                "input": text,
                "reference_ids": ids
            })
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(golden, f, indent=2)
    
    return golden


def run_verification(hf_dir: Path, test_cases_path: Path, 
                     output_dir: Path) -> dict[str, Any]:
    """Run complete tokenizer verification."""
    print(f"Loading HF tokenizer from {hf_dir}...")
    hf_tok = load_hf_tokenizer(hf_dir)
    
    print(f"Loading EdgeCore tokenizer from {hf_dir}...")
    ec_tok = load_edgecore_tokenizer(hf_dir)
    
    with open(test_cases_path) as f:
        test_cases = json.load(f)
    
    print("\n" + "=" * 60)
    print("TOKENIZER VERIFICATION")
    print("=" * 60)
    
    # Check A: Vocabulary metadata
    print("\nA. Vocabulary/Tokenizer Metadata")
    print("-" * 40)
    vocab_results = check_vocabulary_metadata(hf_tok, ec_tok)
    print(f"  Vocab size: HF={vocab_results['vocab_size']['hf']}, EC={vocab_results['vocab_size']['edgecore']} -> {'MATCH' if vocab_results['vocab_size']['match'] else 'MISMATCH'}")
    print(f"  Special tokens match: {'YES' if vocab_results['special_tokens']['match'] else 'NO'}")
    
    # Check B: Tokenization agreement
    print("\nB. Tokenization Agreement (Plain Text)")
    print("-" * 40)
    tokenization_results = check_tokenization_agreement(hf_tok, ec_tok, test_cases)
    for test in tokenization_results["tests"]:
        status = "PASS" if test["match"] else "FAIL"
        print(f"  {test['name']}: {status} (HF={test['hf_count']}, EC={test['ec_count']})")
        if not test["match"] and "mismatch" in test:
            m = test["mismatch"]
            print(f"    First mismatch at pos {m['position']}: expected {m['expected']}, got {m['actual']}")
    
    # Check C: Chat template agreement
    print("\nC. Chat-Template Agreement (Qwen Instruct)")
    print("-" * 40)
    chat_results = check_chat_template_agreement(hf_tok, ec_tok, test_cases)
    for test in chat_results["tests"]:
        status = "PASS" if test["match"] else "FAIL"
        print(f"  {test['name']}: {status} (HF={test['hf_count']}, EC={test['ec_count']})")
        if not test["match"] and "mismatch" in test:
            m = test["mismatch"]
            print(f"    First mismatch at pos {m['position']}: expected {m['expected']}, got {m['actual']}")
    
    # Generate golden file
    print("\nGenerating golden reference file...")
    golden_path = output_dir / "tokenizer_reference.json"
    golden = generate_golden_file(hf_tok, test_cases, golden_path)
    print(f"  Saved to {golden_path}")
    
    # Overall summary
    total_tests = tokenization_results["total"] + chat_results["total"]
    total_passed = tokenization_results["passed"] + chat_results["passed"]
    vocab_pass = vocab_results["vocab_size"]["match"] and vocab_results["special_tokens"]["match"]
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Vocabulary metadata: {'PASS' if vocab_pass else 'FAIL'}")
    print(f"  Tokenization agreement: {tokenization_results['passed']}/{tokenization_results['total']} passed")
    print(f"  Chat template agreement: {chat_results['passed']}/{chat_results['total']} passed")
    print(f"  Overall: {total_passed}/{total_tests} tokenization tests passed")
    
    overall_pass = vocab_pass and (total_passed == total_tests)
    print(f"  STATUS: {'VERIFIED' if overall_pass else 'FAILED'}")
    
    # Save full report
    report = {
        "model": "Qwen2.5-0.5B-Instruct",
        "vocabulary": vocab_results,
        "tokenization": tokenization_results,
        "chat_template": chat_results,
        "overall": {
            "vocab_pass": vocab_pass,
            "tokenization_passed": tokenization_results["passed"],
            "tokenization_total": tokenization_results["total"],
            "chat_passed": chat_results["passed"],
            "chat_total": chat_results["total"],
            "status": "VERIFIED" if overall_pass else "FAILED"
        }
    }
    
    report_path = output_dir / "tokenizer_verification.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to {report_path}")
    
    return report


def main():
    hf_dir = Path(__file__).parent.parent / "models" / "downloads" / "qwen2.5-0.5b"
    test_cases_path = Path(__file__).parent.parent / "tests" / "tokenizer" / "test_cases.json"
    output_dir = Path(__file__).parent.parent / "benchmarks" / "results" / "upgrade3"
    
    if not hf_dir.exists():
        print(f"Error: HF model directory not found: {hf_dir}")
        return 1
    
    if not test_cases_path.exists():
        print(f"Error: Test cases not found: {test_cases_path}")
        return 1
    
    report = run_verification(hf_dir, test_cases_path, output_dir)
    
    if report["overall"]["status"] == "VERIFIED":
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())