#!/bin/bash
# Accept the first argument as the output file, fallback to default if missing
OUT_FILE="${1:-vram_time_series.csv}"

# Write the CSV header
echo "Timestamp,GPU,Memory_Used_MB,Memory_Total_MB,GPU_Utilization_Pct" > "$OUT_FILE"

# Loop and append hardware metrics every 1 second
while true; do
    nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv,noheader >> "$OUT_FILE"
    sleep 1
done
