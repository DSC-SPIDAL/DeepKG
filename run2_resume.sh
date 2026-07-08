#!/bin/bash
# DeepCollector Master Suite: RUN 2 (RESUME AFTER OUTAGE)

RUN_ID="run2"
PROJECTS=("UTSD" "LOTSA" "TIMEBENCH" "TEMPO" "TSFM" "KAGGLETS" "M2" "M6")
MODELS=("gemma4" "qwen" "deepseek")

echo "========================================================"
echo "🚀 RESUMING DEEPCOLLECTOR MASTER SUITE ($RUN_ID)"
echo "========================================================"

run_job() {
    local MODEL=$1
    local CONTEXT=$2
    local CONCURRENCY=$3
    local PROJECT=$4

    local LOG_FILE="Bench_${MODEL}_${CONTEXT}_${PROJECT}_${RUN_ID}.log"

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

# 1. 32K Context Tier (High Concurrency: 32)
# ✅ COMPLETED. Commented out to prevent overwriting your good data.
# for MODEL in "${MODELS[@]}"; do
#     for PROJ in "${PROJECTS[@]}"; do
#         run_job "$MODEL" "32768" "32" "$PROJ"
#     done
# done

# 2. 64K Context Tier (Medium Concurrency: 8)
# Resuming here from the exact drop point (gemma4 + UTSD)
for MODEL in "${MODELS[@]}"; do
    for PROJ in "${PROJECTS[@]}"; do
        run_job "$MODEL" "65536" "8" "$PROJ"
    done
done

# 3. 131K Titan Tier (Single Concurrency: 1)
for MODEL in "${MODELS[@]}"; do
    for PROJ in "${PROJECTS[@]}"; do
        run_job "$MODEL" "131072" "1" "$PROJ"
    done
done

echo -e "\n========================================================"
echo "✅ MASTER SUITE ($RUN_ID) RESUME COMPLETE"
echo "========================================================"
