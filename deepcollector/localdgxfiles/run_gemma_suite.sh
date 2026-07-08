#!/bin/bash

# 1. Export strict factual extraction parameters for Gemma
export DC_TEMP="0.0"
export DC_TOKENS="4096"

# 2. Define the "Elite 8" Master Benchmark Suite
PROJECTS=("UTSD" "TimeBench" "LOTSA" "TEMPO" "TSFM" "KaggleTS" "M2" "M6")

echo "🔥 Launching Master Benchmark Suite for Gemma 4..."
for PROJ in "${PROJECTS[@]}"; do
    echo "========================================================"
    echo "▶️ STARTING PROJECT: $PROJ"
    echo "========================================================"
    
    # THE FIX: Changed 'gemma' to 'gemma4'
    # (Context: 65536 | Concurrency: 8)
    ./start.sh gemma4 65536 8 "$PROJ" > "Bench_Gemma_${PROJ}.log" 2>&1
    
    echo "✅ Finished: $PROJ"
done

echo "🎉 ALL GEMMA BENCHMARK JOBS COMPLETE!"
