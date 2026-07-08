#!/bin/bash

echo "====================================================="
echo "🚀 STARTING PHASE 1: QWEN 2.5 SUITE"
echo "====================================================="
# 1. Force clear VRAM before starting Qwen
docker rm -f vllm_engine >/dev/null 2>&1
sleep 5

# 2. Execute Qwen Suite (this blocks until all 8 finish)
./run_qwen_suite.sh

echo "====================================================="
echo "🧠 STARTING PHASE 2: DEEPSEEK SUITE"
echo "====================================================="
# 3. Force clear VRAM again to dump Qwen's weights
docker rm -f vllm_engine >/dev/null 2>&1
sleep 5

# 4. Execute DeepSeek Suite
./run_deepseek_suite.sh

echo "====================================================="
echo "🏁 MASTER GAUNTLET COMPLETE. ALL MODELS FINISHED."
echo "====================================================="
