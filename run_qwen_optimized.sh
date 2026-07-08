#!/bin/bash

# 1. Load API Keys from your environment
set -a
[ -f .env ] && source .env
set +a

if [ -z "$GEMINI_API_KEY" ] && [ -z "$GOOGLE_API_KEY" ]; then
    echo "❌ ERROR: GEMINI_API_KEY is not set. Please export it or add it to .env"
    exit 1
fi

MODEL="Qwen/Qwen2.5-32B-Instruct"
CONTEXT=65536
CONCURRENCY=8

echo "🧹 Cleaning up old vLLM containers..."
docker rm -f vllm_engine >/dev/null 2>&1
pkill -9 -f run_agent.py >/dev/null 2>&1
rm -rf /dev/shm/vllm* /dev/shm/nccl* /dev/shm/core* 2>/dev/null

echo "🚀 Booting Optimized vLLM for Qwen 2.5 (Context: $CONTEXT)..."
docker run -d --name vllm_engine --gpus all --ipc=host \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model $MODEL \
  --max-model-len $CONTEXT \
  --max-num-seqs $CONCURRENCY \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.95 \
  --tensor-parallel-size 4

echo "⏳ Waiting for vLLM API to spin up..."
while ! curl -s http://localhost:8000/v1/models > /dev/null; do sleep 5; echo -n "."; done
echo -e "\n✅ vLLM is Ready!"

# 2. Export the strict parameters for Qwen
export DC_TEMP="0.0"
export DC_TOKENS="4096"
export DEEPCOLLECTOR_USE_VLLM="True"

echo "🔥 Launching DeepCollector..."
# 3. THE FIX: Pass the arguments so it doesn't default to Gemma!
nohup python3 run_agent.py qwen 65536 8 LOTSA > LOTSA_Qwen_Optimized.log 2>&1 &

echo "✅ Job submitted! Run 'tail -f LOTSA_Qwen_Optimized.log' to watch."
