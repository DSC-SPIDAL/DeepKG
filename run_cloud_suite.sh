#!/bin/bash

# Define the 8 target datasets
PROJECTS=(
    "KAGGLETS"
    "LOTSA"
    "M2"
    "M6"
    "TEMPO"
    "TIMEBENCH"
    "TSFM"
    "UTSD"
)

# Cloud configuration parameters mapping exactly to start.sh
MODEL="cloud"
CONTEXT="32768"
CONCURRENCY="1"

echo "========================================================"
echo "☁️ INITIALIZING CLOUD VERIFICATION SUITE ON DGX"
echo "========================================================"

for DATASET in "${PROJECTS[@]}"; do
    LOG_FILE="Cloud_${DATASET}_Verification.log"
    
    echo -e "\n--------------------------------------------------------"
    echo "⚡ STARTING JOB: Model=$MODEL | Dataset=$DATASET"
    echo "--------------------------------------------------------"
    
    nohup ./start.sh "$MODEL" "$CONTEXT" "$CONCURRENCY" "$DATASET" > "$LOG_FILE" 2>&1 &
    
    wait $!
    
    echo "🏁 Finished Job: $MODEL on $DATASET"
    sleep 5
done

echo -e "\n========================================================"
echo "✅ ALL CLOUD VERIFICATION RUNS COMPLETE"
echo "========================================================"
