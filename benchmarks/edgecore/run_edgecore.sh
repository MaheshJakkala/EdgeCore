#!/bin/bash
# EdgeCore CPU benchmark for Qwen2.5-0.5B-Instruct
# Uses native .ecm model with FP16 precision

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_DIR="${SCRIPT_DIR}/../common"
RESULTS_DIR="${SCRIPT_DIR}/../results/upgrade2"

CONFIG_FILE="${COMMON_DIR}/benchmark_config.json"
PROMPTS_FILE="${COMMON_DIR}/prompts.json"

mkdir -p "${RESULTS_DIR}"

# Load config
MODEL=$(jq -r '.model' "${CONFIG_FILE}")
GENERATION_TOKENS=$(jq -r '.generation_tokens' "${CONFIG_FILE}")
THREADS_LIST=$(jq -r '.threads[]' "${CONFIG_FILE}")
REPETITIONS=$(jq -r '.repetitions' "${CONFIG_FILE}")
BATCH=$(jq -r '.batch' "${CONFIG_FILE}")
CONTEXT=$(jq -r '.context' "${CONFIG_FILE}")
PRECISION=$(jq -r '.precision' "${CONFIG_FILE}")

# Load prompt
FORMATTED_PROMPT=$(jq -r '.formatted_prompt' "${PROMPTS_FILE}")

# Model and runtime paths
MODEL_PATH="${SCRIPT_DIR}/../../models/qwen2-fp16.ecm"
RUNTIME="${SCRIPT_DIR}/../../build/edgecore-runtime"

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "Error: Model not found at ${MODEL_PATH}"
    exit 1
fi

if [[ ! -f "${RUNTIME}" ]]; then
    echo "Error: Runtime not found at ${RUNTIME}"
    echo "Please build EdgeCore first: make"
    exit 1
fi

echo "Model: ${MODEL}"
echo "Model path: ${MODEL_PATH}"
echo "Prompt: ${FORMATTED_PROMPT}"
echo "Generation tokens: ${GENERATION_TOKENS}"
echo "Threads: ${THREADS_LIST}"
echo "Repetitions: ${REPETITIONS}"

# Build results JSON
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
RESULTS_FILE="${RESULTS_DIR}/edgecore.json"

cat > "${RESULTS_FILE}" <<EOF
{
  "runtime": "edgecore",
  "model": "${MODEL}",
  "precision": "${PRECISION}",
  "batch": ${BATCH},
  "context": ${CONTEXT},
  "generation_tokens": ${GENERATION_TOKENS},
  "hardware": "Intel i3-1115G4",
  "timestamp": "${TIMESTAMP}",
  "software": {
    "edgecore": "$(${RUNTIME} --version 2>&1 | head -1 || echo 'v0.2')"
  },
  "runs": []
}
EOF

for THREADS in ${THREADS_LIST}; do
    echo ""
    echo "=== Running with ${THREADS} thread(s) ==="
    
    TTFT_TIMES=()
    TPOT_TIMES=()
    TOK_PER_SEC_TIMES=()
    P50_TIMES=()
    P95_TIMES=()
    P99_TIMES=()
    RAM_MB_MAX=0
    
    for ((run=1; run<=REPETITIONS; run++)); do
        echo "  Run ${run}/${REPETITIONS}..."
        
        # Run EdgeCore with JSON output
        OUTPUT=$(${RUNTIME} \
            --model "${MODEL_PATH}" \
            --mode generate \
            --prompt "${FORMATTED_PROMPT}" \
            --n-tokens "${GENERATION_TOKENS}" \
            --threads "${THREADS}" \
            --batch "${BATCH}" \
            --temp 0.0 \
            --json 2>&1)
        
        # Parse JSON output
        TTFT_MS=$(echo "${OUTPUT}" | jq -r '.ttft_ms // 0')
        TPOT_AVG_MS=$(echo "${OUTPUT}" | jq -r '.tpot_avg_ms // 0')
        TOK_PER_SEC=$(echo "${OUTPUT}" | jq -r '.tokens_per_sec // 0')
        P50_MS=$(echo "${OUTPUT}" | jq -r '.p50_ms // 0')
        P95_MS=$(echo "${OUTPUT}" | jq -r '.p95_ms // 0')
        P99_MS=$(echo "${OUTPUT}" | jq -r '.p99_ms // 0')
        PEAK_RSS_GB=$(echo "${OUTPUT}" | jq -r '.peak_rss_gb // 0')
        PEAK_RSS_MB=$(echo "${PEAK_RSS_GB} * 1024" | bc -l | cut -d'.' -f1)
        
        TTFT_TIMES+=(${TTFT_MS})
        TPOT_TIMES+=(${TPOT_AVG_MS})
        TOK_PER_SEC_TIMES+=(${TOK_PER_SEC})
        
        if [[ ${PEAK_RSS_MB} -gt ${RAM_MB_MAX} ]]; then
            RAM_MB_MAX=${PEAK_RSS_MB}
        fi
    done
    
    # Calculate percentiles
    IFS=$'\n' SORTED_TPOT=($(sort -n <<<"${TPOT_TIMES[*]}"))
    COUNT=${#SORTED_TPOT[@]}
    P50=${SORTED_TPOT[$((COUNT / 2))]}
    P95=${SORTED_TPOT[$((COUNT * 95 / 100))]}
    P99=${SORTED_TPOT[$((COUNT * 99 / 100))]}
    
    AVG_TTFT=$(echo "${TTFT_TIMES[@]}" | tr ' ' '\n' | awk '{sum+=$1} END {print sum/NR}')
    AVG_TPOT=$(echo "${TPOT_TIMES[@]}" | tr ' ' '\n' | awk '{sum+=$1} END {print sum/NR}')
    AVG_TOK_SEC=$(echo "${TOK_PER_SEC_TIMES[@]}" | tr ' ' '\n' | awk '{sum+=$1} END {print sum/NR}')
    
    echo "  Results: ${AVG_TOK_SEC} tok/s, P95: ${P95} ms, RAM: ${RAM_MB_MAX} MB"
    
    # Append to results
    jq --argjson threads "${THREADS}" \
       --argjson ttft "${AVG_TTFT}" \
       --argjson tpot "${AVG_TPOT}" \
       --argjson tok_sec "${AVG_TOK_SEC}" \
       --argjson p50 "${P50}" \
       --argjson p95 "${P95}" \
       --argjson p99 "${P99}" \
       --argjson ram "${RAM_MB_MAX}" \
       '.runs += [{
           "threads": $threads,
           "ttft_ms": $ttft,
           "tpot_ms": $tpot,
           "tokens_per_second": $tok_sec,
           "p50_ms": $p50,
           "p95_ms": $p95,
           "p99_ms": $p99,
           "ram_mb": $ram
       }]' "${RESULTS_FILE}" > "${RESULTS_FILE}.tmp" && mv "${RESULTS_FILE}.tmp" "${RESULTS_FILE}"
done

echo ""
echo "Results saved to ${RESULTS_FILE}"
cat "${RESULTS_FILE}"