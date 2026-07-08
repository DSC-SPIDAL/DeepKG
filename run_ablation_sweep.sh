#!/bin/bash
# DeepSeek 32K Ablation: Isolated Re-run
export DC_TEMP="0.6"
export DC_TOKENS="8192"

CONTEXT="32768"
CONCURRENCY="32"

PROJECTS=("UTSD" "TimeBench" "LOTSA" "TEMPO" "TSFM" "KaggleTS" "M2" "M6")
MODELS=("deepseek") # Gemma and Qwen removed

echo "====================================================="
echo "🚀 STARTING ABLATION SWEEP (DeepSeek ONLY)"
echo "   Context: $CONTEXT | Concurrency: $CONCURRENCY"
echo "====================================================="

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "====================================================="
    echo "🧠 SWAPPING TO MODEL: $MODEL"
    echo "====================================================="
    
    docker rm -f vllm_engine >/dev/null 2>&1
    sleep 5

    for PROJ in "${PROJECTS[@]}"; do
        echo "▶️ RUNNING: $MODEL on $PROJ..."
        ./start.sh "$MODEL" "$CONTEXT" "$CONCURRENCY" "$PROJ" > "Ablation_${MODEL}_${PROJ}.log" 2>&1
        echo "✅ Finished: $MODEL on $PROJ"
    done
done

echo "====================================================="
echo "🏁 DEEPSEEK ABLATION SWEEP COMPLETE."
echo "====================================================="
