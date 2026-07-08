#!/bin/bash
# DeepCollector Master Suite: RUN 2 (FINAL RECOVERY)

RUN_ID="run2"
PROJECTS=("UTSD" "LOTSA" "TIMEBENCH" "TEMPO" "TSFM" "KAGGLETS" "M2" "M6")
MODELS=("gemma4" "qwen" "deepseek")

echo "========================================================"
echo "🚑 RECOVERING DEEPCOLLECTOR MASTER SUITE ($RUN_ID)"
echo "========================================================"

run_job() {
    local MODEL=$1
    local CONTEXT=$2
    local CONCURRENCY=$3
    local PROJECT=$4

    local LOG_FILE="Bench_${MODEL}_${CONTEXT}_${PROJECT}_${RUN_ID}_RECOVERY.log"

    echo -e "\n--------------------------------------------------------"
    echo "⚡ STARTING: $MODEL | Context: $CONTEXT | Concurrency: $CONCURRENCY | Project: $PROJECT"
    echo "--------------------------------------------------------"

    # Force kill container between jobs to prevent VRAM fragmentation
    docker rm -f vllm_engine >/dev/null 2>&1
    sleep 5

    ./start.sh "$MODEL" "$CONTEXT" "$CONCURRENCY" "$PROJECT" > "$LOG_FILE" 2>&1

    # Give the system time to cool down before the next allocation
    sleep 10
}

# -------------------------------------------------------------
# 1. RECOVER THE FAILED 64K JOB (M6 on DeepSeek)
# We lower concurrency to 4 to prevent the OOM crash that killed it.
# -------------------------------------------------------------
echo "🔄 Retrying crashed DeepSeek 64K job on M6..."
run_job "deepseek" "65536" "4" "M6"


# -------------------------------------------------------------
# 2. PROCEED TO 131K TITAN TIER (Single Concurrency: 1)
# -------------------------------------------------------------
echo "🏔️ Entering Titan Tier (131K Context)..."
for MODEL in "${MODELS[@]}"; do
    for PROJ in "${PROJECTS[@]}"; do
        run_job "$MODEL" "131072" "1" "$PROJ"
    done
done

echo -e "\n========================================================"
echo "✅ MASTER SUITE ($RUN_ID) FULLY COMPLETED"
echo "========================================================"
