#!/usr/bin/env bash
# EdgeCore smoke test: build, generate tiny synthetic models (int8 + fp16),
# and exercise the runtime (logits / generate / bench) plus the hardware profiler.
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHONPATH=. python3 scripts/make_synthetic.py

INT8=models/tiny-gpt2-int8.ecm
FP16=models/tiny-gpt2-fp16.ecm
RUNTIME=build/edgecore-runtime
HWPROF=build/edgecore-hwprof

fail() { echo "SMOKE FAIL: $1" >&2; exit 1; }

echo "[1/6] hardware profiler"
$HWPROF | python3 -c "
import json,sys
d = json.load(sys.stdin)
assert d['logical_cores'] >= 1 and d['ram_gb'] > 0, d
assert d['physical_cores'] >= 1 and d['threads_per_core'] >= 1, d
assert any(c['level'] == 1 for c in d['caches']), d
print('  ok:', d['cpu_model'], '| phys', d['physical_cores'], 'logical', d['logical_cores'],
      '| simd', d['simd'], '| caches', [(c['level'], c['type']) for c in d['caches']])"

echo "[2/6] logits mode (int8, auto precision)"
$RUNTIME --model $INT8 --mode logits --prompt "hello world" \
  > /tmp/ec_logits.json 2>/dev/null
python3 -c "
import json
d = json.load(open('/tmp/ec_logits.json'))
assert d['precision'] == 'int8'
assert d['n_tokens'] == 11 and len(d['logits']) == 11
assert all(len(r) == 256 for r in d['logits'])
print('  ok: %d tokens, %d dims, precision %s' % (d['n_tokens'], len(d['logits'][0]), d['precision']))"

echo "[3/6] logits mode (fp16, auto precision)"
$RUNTIME --model $FP16 --mode logits --prompt "hello world" \
  > /tmp/ec_logits_fp16.json 2>/dev/null
python3 -c "
import json
d = json.load(open('/tmp/ec_logits_fp16.json'))
assert d['precision'] == 'fp16' and len(d['logits']) == 11
assert 'affinity_cpus' in d
print('  ok: fp16 forward ran; affinity_cpus', d['affinity_cpus'])"

echo "[4/6] logits-bin (raw float32 dump)"
printf "0 1 2 3 4 5 6 7 8 9 10\n" > /tmp/prompt_tokens.txt
$RUNTIME --model $INT8 --mode logits --tokens-file /tmp/prompt_tokens.txt \
  --logits-bin /tmp/ec_logits.bin --logits-limit 64 > /tmp/ec_lb.json 2>/dev/null
python3 -c "
import numpy as np, json
b = open('/tmp/ec_logits.bin','rb')
n, V = np.frombuffer(b.read(8), dtype=np.int32)
toks = np.frombuffer(b.read(4*n), dtype=np.int32)
logits = np.frombuffer(b.read(4*n*V), dtype=np.float32).reshape(n, V)
assert n == 11 and V == 256 and len(toks) == 11
assert abs(float(logits[0].sum())) > 0
print('  ok: bin dump n=%d V=%d' % (n, V))"

echo "[5/6] generate mode (int8)"
$RUNTIME --model $INT8 --mode generate --prompt "hello world" --n-tokens 16 \
  --temp 0.7 --top-k 40 --seed 1 --no-print > /tmp/ec_gen.json 2>/dev/null
python3 -c "
import json
d = json.load(open('/tmp/ec_gen.json'))
assert d['n_generated_tokens'] == 16
assert d['ttft_ms'] >= 0 and d['tpot_avg_ms'] > 0
assert d['tokens_per_sec'] > 0 and d['p50_ms'] > 0
assert d['affinity_requested'] == 'none'
print('  ok: ttft %.2fms tpot %.2fms tps %.1f' % (d['ttft_ms'], d['tpot_avg_ms'], d['tokens_per_sec']))"

echo "[6/6] bench mode (batch=2, threads=2, fp16, affinity physical)"
$RUNTIME --model $FP16 --mode bench --prompt "hello world" \
  --n-tokens 16 --warmup 1 --iters 3 --batch 2 --threads 2 --affinity physical \
  > /tmp/ec_bench.json 2>/dev/null
python3 -c "
import json
d = json.load(open('/tmp/ec_bench.json'))
assert d['n_generated_tokens'] == 32
assert 0 < d['avg_token_ms'] <= d['p95_ms']
assert d['tokens_per_sec'] > 0
assert d['affinity_applied'] is True and len(d['affinity_cpus']) >= 1
assert 'ttft_p95_ms' in d and 'ttft_p99_ms' in d
print('  ok: avg %.3fms p50 %.3f p95 %.3f tps %.1f | ttft_p95 %.2fms' % (
    d['avg_token_ms'], d['p50_ms'], d['p95_ms'], d['tokens_per_sec'], d['ttft_p95_ms']))"

echo "SMOKE OK"
