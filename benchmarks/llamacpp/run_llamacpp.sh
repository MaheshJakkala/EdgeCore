#!/bin/bash
# llama.cpp CPU benchmark for Qwen2.5-0.5B-Instruct
# Uses GGUF FP16 model

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

# Model path
MODEL_PATH="${SCRIPT_DIR}/../../models/baselines/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-fp16.gguf"
LLAMA_CLI="${SCRIPT_DIR}/../../build/llama.cpp/build/bin/llama-cli"

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "Error: Model not found at ${MODEL_PATH}"
    echo "Please download the GGUF model first."
    exit 1
fi

if [[ ! -f "${LLAMA_CLI}" ]]; then
    echo "Error: llama-cli not found at ${LLAMA_CLI}"
    echo "Please build llama.cpp first."
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
RESULTS_FILE="${RESULTS_DIR}/llamacpp.json"

cat > "${RESULTS_FILE}" <<EOF
{
  "runtime": "llamacpp",
  "model": "${MODEL}",
  "precision": "${PRECISION}",
  "batch": ${BATCH},
  "context": ${CONTEXT},
  "generation_tokens": ${GENERATION_TOKENS},
  "hardware": "Intel i3-1115G4",
  "timestamp": "${TIMESTAMP}",
  "software": {
    "llama_cpp": "$(${LLAMA_CLI} --version 2>&1 | head -1 || echo 'unknown')"
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
        
        # Use llama-cli with timing
        # We use --prompt-cache to get consistent results, but disable for fair comparison
        START_TIME=$(date +%s%3N)
        
        OUTPUT=$(${LLAMA_CLI} \
            -m "${MODEL_PATH}" \
            -t "${THREADS}" \
            -n "${GENERATION_TOKENS}" \
            -c "${CONTEXT}" \
            -b "${BATCH}" \
            --temp 0.0 \
            --top-p 1.0 \
            --top-k 0 \
            --repeat-penalty 1.0 \
            -p "${FORMATTED_PROMPT}" \
            --no-display-prompt \
            2>&1)
        
        END_TIME=$(date +%s%3N)
        TOTAL_MS=$((END_TIME - START_TIME))
        
        # Extract generated tokens (approximate)
        # Count tokens in output (rough estimate)
        GENERATED_TEXT=$(echo "${OUTPUT}" | tail -n +2 | head -n -1)
        # Use llama-tokenize to count tokens accurately
        TOKEN_COUNT=$(${LLAMA_CLI} -m "${MODEL_PATH}" -t 1 -c "${CONTEXT}" --tokenize -p "${GENERATED_TEXT}" 2>&1 | grep -o 'tokens = [0-9]*' | cut -d' ' -f3)
        if [[ -z "${TOKEN_COUNT}" ]]; then
            TOKEN_COUNT=${GENERATION_TOKENS}
        fi
        
        # For TTFT, run single token generation
        TTFT_START=$(date +%s%3N)
        ${LLAMA_CLI} \
            -m "${MODEL_PATH}" \
            -t "${THREADS}" \
            -n 1 \
            -c "${CONTEXT}" \
            -b "${BATCH}" \
            --temp 0.0 \
            -p "${FORMATTED_PROMPT}" \
            --no-display-prompt \
            > /dev/null 2>&1
        TTFT_END=$(date +%s%3N)
        TTFT_MS=$((TTFT_END - TTFT_START))
        
        # Calculate TPOT
        if [[ ${TOKEN_COUNT} -gt 1 ]]; then
            TPOT_MS=$(( (TOTAL_MS - TTFT_MS) / (TOKEN_COUNT - 1) ))
        else
            TPOT_MS=${TOTAL_MS}
        fi
        
        TOK_PER_SEC=$(echo "scale=2; ${TOKEN_COUNT} * 1000 / ${TOTAL_MS}" | bc)
        
        TTFT_TIMES+=(${TTFT_MS})
        TPOT_TIMES+=(${TPOT_MS})
        TOK_PER_SEC_TIMES+=(${TOK_PER_SEC})
        
        # Memory usage (approximate)
        RSS_KB=$(ps -o rss= -p $$)
        RSS_MB=$((RSS_KB / 1024))
        if [[ ${RSS_MB} -gt ${RAM_MB_MAX} ]]; then
            RAM_MB_MAX=${RSS_MB}
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