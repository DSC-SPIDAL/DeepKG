#!/bin/bash
# DeepSeek 64K Suite: Isolated Re-run
export DC_TEMP="0.6"
export DC_TOKENS="8192"

PROJECTS=("UTSD" "TimeBench" "LOTSA" "TEMPO" "TSFM" "KaggleTS" "M2" "M6")

echo "🔥 Launching Master Benchmark Suite for DeepSeek (Tokens: $DC_TOKENS)..."
for PROJ in "${PROJECTS[@]}"; do
    echo "▶️ STARTING PROJECT: $PROJ"
    ./start.sh deepseek 65536 8 "$PROJ" > "Bench_DeepSeek_${PROJ}.log" 2>&1
    echo "✅ Finished: $PROJ"
done
