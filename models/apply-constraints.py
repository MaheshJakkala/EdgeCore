#!/usr/bin/env python3
import json

with open("models/qwen25-autotune-bench.json") as f:
    bench = json.load(f)

with open("models/qwen25-fp16-verify.json") as f:
    fp16_verify = json.load(f)

with open("models/qwen2-int8-per-verify.json") as f:
    int8_verify = json.load(f)

fp16_p95 = min(m["p95_ms"] for m in bench["measurements"] if m["precision"] == "fp16")
p95_threshold = fp16_p95 * 1.2

print(f"FP16 best P95: {fp16_p95:.1f}ms")
print(f"P95 threshold (1.2x FP16 best): {p95_threshold:.1f}ms")

constraints = {
    "max_ram_gb": 6.0,
    "max_p95_ms": p95_threshold,
    "max_accuracy_loss_pct": 1.0,
    "batch": 1,
    "context": 2048
}

verification_data = {
    "fp16": {
        "cosine_mean": fp16_verify["models"]["models/qwen2-fp16.ecm"]["summary"]["worst_cosine_mean"],
        "top1": fp16_verify["models"]["models/qwen2-fp16.ecm"]["summary"]["worst_top1"],
        "ppl_ratio": fp16_verify["models"]["models/qwen2-fp16.ecm"]["summary"]["max_ppl_ratio"]
    },
    "int8": {
        "cosine_mean": int8_verify["models"]["models/qwen2-int8-per.ecm"]["summary"]["worst_cosine_mean"],
        "top1": int8_verify["models"]["models/qwen2-int8-per.ecm"]["summary"]["worst_top1"],
        "ppl_ratio": int8_verify["models"]["models/qwen2-int8-per.ecm"]["summary"]["max_ppl_ratio"]
    }
}

fp16_cos = verification_data["fp16"]["cosine_mean"]
int8_cos = verification_data["int8"]["cosine_mean"]
accuracy_loss_pct = (1 - int8_cos / fp16_cos) * 100

print(f"\nAccuracy loss: {accuracy_loss_pct:.2f}% (threshold: {constraints['max_accuracy_loss_pct']}%)")

results = []

for m in bench["measurements"]:
    prec = m["precision"]
    threads = m["threads"]
    
    ram_ok = m["peak_rss_gb"] < constraints["max_ram_gb"]
    p95_ok = m["p95_ms"] < constraints["max_p95_ms"]
    batch_ok = m["batch"] == constraints["batch"]
    
    if prec == "fp16":
        acc_loss = 0.0
        acc_ok = True
    else:
        acc_loss = accuracy_loss_pct
        acc_ok = acc_loss < constraints["max_accuracy_loss_pct"]
    
    accepted = ram_ok and p95_ok and acc_ok and batch_ok
    
    results.append({
        "precision": prec.upper(),
        "threads": threads,
        "batch": m["batch"],
        "affinity": m["affinity"],
        "avg_token_ms": m["avg_token_ms"],
        "ttft_avg_ms": m["ttft_avg_ms"],
        "p95_ms": m["p95_ms"],
        "tokens_per_sec": m["tokens_per_sec"],
        "peak_rss_gb": m["peak_rss_gb"],
        "accuracy_loss_pct": round(acc_loss, 2),
        "ram_ok": ram_ok,
        "p95_ok": p95_ok,
        "accuracy_ok": acc_ok,
        "batch_ok": batch_ok,
        "accepted": accepted
    })
    
    status = "ACCEPTED" if accepted else "REJECTED"
    print(f"\n{prec.upper()} / {threads}T: {status}")
    print(f"  RAM: {m['peak_rss_gb']:.2f}GB ({'OK' if ram_ok else 'FAIL'} < {constraints['max_ram_gb']}GB)")
    print(f"  P95: {m['p95_ms']:.1f}ms ({'OK' if p95_ok else 'FAIL'} < {constraints['max_p95_ms']:.1f}ms)")
    print(f"  Accuracy loss: {acc_loss:.2f}% ({'OK' if acc_ok else 'FAIL'} < {constraints['max_accuracy_loss_pct']}%)")
    print(f"  Batch: {m['batch']} ({'OK' if batch_ok else 'FAIL'} == {constraints['batch']})")

accepted = [r for r in results if r["accepted"]]
if accepted:
    best = min(accepted, key=lambda x: x["avg_token_ms"])
    print(f"\n✅ RECOMMENDED: {best['precision']} / {best['threads']} threads")
    print(f"   {best['tokens_per_sec']:.1f} tok/s, P95={best['p95_ms']:.1f}ms, RAM={best['peak_rss_gb']:.2f}GB")
else:
    print("\n❌ NO CONFIGURATION ACCEPTED")

with open("models/qwen25-constraint-results.json", "w") as f:
    json.dump({
        "constraints": constraints,
        "fp16_baseline_p95_ms": fp16_p95,
        "p95_threshold_ms": p95_threshold,
        "accuracy_loss_pct": round(accuracy_loss_pct, 2),
        "candidates": results,
        "recommended": accepted[0] if accepted else None
    }, f, indent=2)

print("\nWritten to models/qwen25-constraint-results.json")