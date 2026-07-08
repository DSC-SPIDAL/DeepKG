#!/bin/bash

# Define the remaining benchmark matrix with the correct model keys
JOBS=(
    "gemma4 UTSD"
    "deepseek UTSD"
    "qwen LOTSA"
    "gemma4 LOTSA"
    "deepseek LOTSA"
)

CONTEXT="131072"
CONCURRENCY="1"

echo "========================================================"
echo "🚀 INITIALIZING CORRECTED TITAN ABLATION SUITE (131K)"
echo "========================================================"

for JOB in "${JOBS[@]}"; do
    read -r MODEL DATASET <<< "$JOB"
    
    # Keep the log names clean but use the correct model variable for execution
    LOG_FILE="Titan_${MODEL}_${DATASET}_128K.log"
    VRAM_CSV="vram_${MODEL}_${DATASET}.csv"
    
    echo -e "\n--------------------------------------------------------"
    echo "⚡ STARTING JOB: Model=$MODEL | Dataset=$DATASET"
    echo "--------------------------------------------------------"
    
    docker rm -f vllm_engine >/dev/null 2>&1
    sleep 5
    
    nohup ./start.sh "$MODEL" "$CONTEXT" "$CONCURRENCY" "$DATASET" > "$LOG_FILE" 2>&1 &
    PIPELINE_PID=$!
    
    echo "⏳ Waiting up to 180s for vllm_engine to initialize..."
    CONTAINER_READY=false
    
    for i in {1..180}; do
        STATUS=$(docker inspect -f '{{.State.Status}}' vllm_engine 2>/dev/null)
        
        if [ "$STATUS" == "running" ]; then
            CONTAINER_READY=true
            break
        elif [ "$STATUS" == "exited" ] || [ "$STATUS" == "dead" ]; then
            echo "💥 Container crashed prematurely!"
            break
        fi
        sleep 1
    done
    
    if [ "$CONTAINER_READY" == "true" ]; then
        echo "🚀 Container running! Tracking VRAM..."
        echo "Timestamp,Elapsed_Seconds,Total_VRAM_GB" > "$VRAM_CSV"
        START_TIME=$(date +%s)
        
        while true; do
            CURRENT_TIME=$(date +%s)
            ELAPSED=$((CURRENT_TIME - START_TIME))
            
            if [ "$(docker inspect -f '{{.State.Running}}' vllm_engine 2>/dev/null)" == "true" ]; then
                TOTAL_MEM_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
                TOTAL_MEM_GB=$(echo "scale=2; $TOTAL_MEM_MB / 1024" | bc)
                echo "$(date '+%H:%M:%S'),$ELAPSED,$TOTAL_MEM_GB" >> "$VRAM_CSV"
            else
                echo "$(date '+%H:%M:%S'),$ELAPSED,CONTAINER_DEAD" >> "$VRAM_CSV"
                break
            fi
            sleep 5
        done
    else
        echo "❌ CRITICAL: Engine failed to boot. Skipping job."
        echo "0,0,LAUNCH_FAILED" > "$VRAM_CSV"
    fi
    
    wait $PIPELINE_PID 2>/dev/null
    sleep 10
done

echo -e "\n========================================================"
echo "✅ ALL MATRIX RUNS COMPLETE"
echo "========================================================"
