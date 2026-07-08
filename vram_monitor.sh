#!/bin/bash

LOG_FILE="vram_time_series.csv"
echo "Timestamp,Elapsed_Seconds,Total_VRAM_GB" > $LOG_FILE

echo "📊 VRAM Monitor Started. Logging to $LOG_FILE..."
echo "Press [CTRL+C] to stop monitoring."

echo "⏳ Waiting for vllm_engine container to spin up..."

# Wait loop: pause until the container actually exists and is running
while [ "$(docker inspect -f '{{.State.Running}}' vllm_engine 2>/dev/null)" != "true" ]; do
    sleep 1
done

echo "🚀 Container found! Recording VRAM every 5 seconds..."
START_TIME=$(date +%s)

while true; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    
    if [ "$(docker inspect -f '{{.State.Running}}' vllm_engine 2>/dev/null)" == "true" ]; then
        TOTAL_MEM_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
        TOTAL_MEM_GB=$(echo "scale=2; $TOTAL_MEM_MB / 1024" | bc)
        
        echo "$(date '+%H:%M:%S'),$ELAPSED,$TOTAL_MEM_GB" >> $LOG_FILE
    else
        echo "$(date '+%H:%M:%S'),$ELAPSED,CONTAINER_DEAD" >> $LOG_FILE
        echo -e "\n⚠️ Container vllm_engine crashed or finished. Monitoring ended."
        break
    fi
    sleep 5
done
