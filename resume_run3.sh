#!/bin/bash
ROOT_DIR="$HOME/Desktop/DeepKG"
RUN_ID="run3"

MODELS=("deepseek" "qwen" "gemma4")
PROJECTS=("UTSD" "TIMEBENCH" "LOTSA" "M2" "M6" "TEMPO" "TSFM" "KAGGLETS")

echo "========================================================"
echo "🚀 RESUMING DEEPCOLLECTOR (Root Execution Mode)"
echo "========================================================"

run_job() {
    local MODEL=$1
    local CONTEXT=$2
    local CONCURRENCY=$3
    local PROJECT=$4

    local LOG_FILE="$ROOT_DIR/Bench_${MODEL}_${CONTEXT}_${PROJECT}_${RUN_ID}.log"
    local VRAM_FILE="$ROOT_DIR/VRAM_${MODEL}_${CONTEXT}_${PROJECT}_${RUN_ID}.csv"

    echo -e "\n--------------------------------------------------------"
    echo "⚡ STARTING: $MODEL | Context: $CONTEXT | Concurrency: $CONCURRENCY | Project: $PROJECT"
    echo "--------------------------------------------------------"

    docker rm -f vllm_engine >/dev/null 2>&1
    sleep 5

    # Executing natively from ROOT_DIR
    ./vram_monitor.sh "$VRAM_FILE" &
    VRAM_PID=$!

    # 🟢 Run start.sh directly from root and capture the true Linux Exit Code
    ./start.sh "$MODEL" "$CONTEXT" "$CONCURRENCY" "$PROJECT" > "$LOG_FILE" 2>&1
    EXIT_CODE=$?

    kill $VRAM_PID 2>/dev/null

    # 🛑 BULLETPROOF EXIT-CODE KILL-SWITCH
    if [ $EXIT_CODE -ne 0 ]; then
        echo "❌ ERROR: start.sh failed for $MODEL $PROJECT."
        echo "Linux Exit Code: $EXIT_CODE. Aborting master script to prevent runaway loop!"
        cat "$LOG_FILE"
        exit 1
    fi

    # 🛑 FATAL EXCEPTION SCANNER
    if grep -iqE "Traceback|SyntaxError:|IndentationError:" "$LOG_FILE"; then
        echo "❌ ERROR: Python exception detected in $LOG_FILE!"
        echo "Aborting master script to prevent data corruption."
        exit 1
    fi

    sleep 10
}

cd "$ROOT_DIR" || exit 1

# --- 1. RESUME 64K CONTEXT TIER FOR DEEPSEEK (Post-LOTSA) ---
# DeepSeek already finished UTSD, TIMEBENCH, and LOTSA. Resuming at M2.
DEEPSEEK_REMAINING=("M2" "M6" "TEMPO" "TSFM" "KAGGLETS")
for PROJ in "${DEEPSEEK_REMAINING[@]}"; do
    run_job "deepseek" "65536" "8" "$PROJ"
done

# --- 2. COMPLETE 64K CONTEXT TIER FOR QWEN & GEMMA4 ---
for MODEL in "qwen" "gemma4"; do
    for PROJ in "${PROJECTS[@]}"; do
        run_job "$MODEL" "65536" "8" "$PROJ"
    done
done

# --- 3. 131K CONTEXT TIER (Single Concurrency: 1) - ALL MODELS & PROJECTS ---
for MODEL in "${MODELS[@]}"; do
    for PROJ in "${PROJECTS[@]}"; do
        run_job "$MODEL" "131072" "1" "$PROJ"
    done
done

echo -e "\n========================================================"
echo "✅ RUN 3 RESUME COMPLETE"
echo "========================================================"
