#!/bin/bash
export DC_TEMP="0.0"
export DC_TOKENS="4096"

# The Stress Test Parameters
CONTEXT="32768"
CONCURRENCY="32"

# The target ablation projects
PROJECTS=("UTSD" "TimeBench" "LOTSA" "TEMPO" "TSFM" "KaggleTS" "M2" "M6")
# The models to cycle through
MODELS=("gemma4" "qwen" "deepseek")

echo "====================================================="
echo "🚀 STARTING OVERNIGHT ABLATION SWEEP"
echo "   Context: $CONTEXT | Concurrency: $CONCURRENCY"
echo "====================================================="

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "====================================================="
    echo "🧠 SWAPPING TO MODEL: $MODEL"
    echo "====================================================="
    
    # Force clear VRAM before loading the new model
    docker rm -f vllm_engine >/dev/null 2>&1
    sleep 5

    for PROJ in "${PROJECTS[@]}"; do
        echo "▶️ RUNNING: $MODEL on $PROJ..."
        ./start.sh "$MODEL" "$CONTEXT" "$CONCURRENCY" "$PROJ" > "Ablation_${MODEL}_${PROJ}.log" 2>&1
        echo "✅ Finished: $MODEL on $PROJ"
    done
done

echo "====================================================="
echo "🏁 ABLATION SWEEP COMPLETE. HAVE A GOOD MORNING!"
echo "====================================================="
