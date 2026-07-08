#!/bin/bash
# Qwen Suite: Dense Hallucination Baseline
export DC_TEMP="0.0"
export DC_TOKENS="4096"

PROJECTS=("UTSD" "TimeBench" "LOTSA" "TEMPO" "TSFM" "KaggleTS" "M2" "M6")

echo "🔥 Launching Master Benchmark Suite for Qwen 2.5..."
for PROJ in "${PROJECTS[@]}"; do
    echo "▶️ STARTING PROJECT: $PROJ"
    # Using 'qwen' as the model choice (mapped to qwen-2.5-32b-it in your start.sh)
    ./start.sh qwen 65536 8 "$PROJ" > "Bench_Qwen_${PROJ}.log" 2>&1
    echo "✅ Finished: $PROJ"
done
