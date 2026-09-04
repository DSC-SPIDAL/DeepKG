#!/bin/bash
MODEL_CHOICE=${1:-gemma4}
export LOCAL_MAX_CONTEXT=${2:-32768}
export VLLM_CONCURRENCY=${3:-16}
export TARGET_PROJECT=${4:-LOTSA}

export BENCHMARK_MODE="LOCAL"
export TP_SIZE=4 

if [ "$MODEL_CHOICE" == "cloud" ]; then
    export BENCHMARK_MODE="CLOUD"
    export LOCAL_MODEL_ID="gemini-cascade"
    export LOCAL_MAX_CONTEXT=2000000
    export VLLM_CONCURRENCY=50
elif [ "$MODEL_CHOICE" == "gemma2" ]; then
    export LOCAL_MODEL_ID="google/gemma-2-27b-it"
elif [ "$MODEL_CHOICE" == "qwen" ]; then
    export LOCAL_MODEL_ID="Qwen/Qwen2.5-32B-Instruct"
elif [ "$MODEL_CHOICE" == "deepseek" ]; then
    export LOCAL_MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
elif [ "$MODEL_CHOICE" == "gemma4" ]; then
    export LOCAL_MODEL_ID="google/gemma-4-31b-it" 
elif [ "$MODEL_CHOICE" == "llama3" ]; then
    export LOCAL_MODEL_ID="meta-llama/Llama-3.3-70B-Instruct"
elif [ "$MODEL_CHOICE" == "command-r" ]; then
    export LOCAL_MODEL_ID="CohereForAI/c4ai-command-r-08-2024"
    export TP_SIZE=2
else
    echo "❌ Unknown model choice."
    exit 1
fi

export ENABLE_DEEP_RESEARCH_FLAG="True"
export APPLY_AMNESIA="True"

cleanup() {
    echo -e "\n🛑 Stopping script and cleaning up..."
    if [ "$BENCHMARK_MODE" == "LOCAL" ]; then docker rm -f vllm_engine >/dev/null 2>&1 || true; fi
    stty sane 2>/dev/null
}
trap 'cleanup; exit 130' INT TERM
trap cleanup EXIT

cd /home/geoffrey/Desktop/DeepKG || exit 1

# 🔥 CRITICAL FIX: 'set -a' forces ALL variables in .env to be exported to Python!
if [ -f ".env" ]; then 
    set -a
    source .env
    set +a
else 
    exit 1
fi

export KB_SHEET_ID="1-PuWrHO30E4WPM-rOed03n42gfo5AlEtscKqqtjznA0"
export PROJECT_LIST_ID="1gJ6oHZj0NzCHNOeFNyJTBTtlmS0b7gBSHF3iOqJrFwE"

docker rm -f vllm_engine >/dev/null 2>&1
pkill -9 -f run_agent.py >/dev/null 2>&1 || true
rm -rf /dev/shm/vllm* /dev/shm/nccl* /dev/shm/core* 2>/dev/null

if [ "$BENCHMARK_MODE" == "LOCAL" ]; then
    DOCKER_ARGS=(--model "$LOCAL_MODEL_ID" --tensor-parallel-size "$TP_SIZE" --max-model-len "$LOCAL_MAX_CONTEXT" --enable-prefix-caching --gpu-memory-utilization 0.95 --trust-remote-code --enforce-eager)
    
    if [ "$MODEL_CHOICE" == "gemma2" ]; then DOCKER_ARGS+=( --rope-scaling '{"type":"dynamic","factor":16.0}' ); fi

    echo "🧠 Starting vLLM ($LOCAL_MODEL_ID) | Context: $LOCAL_MAX_CONTEXT | Concurrency: $VLLM_CONCURRENCY | TP: $TP_SIZE..."
    
    docker run -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -d --name vllm_engine --gpus all --shm-size 32g -e NVIDIA_VISIBLE_DEVICES="0,1,2,4" -e HF_TOKEN="$HF_TOKEN" -v /raid/huggingface_cache:/root/.cache/huggingface -p 8000:8000 --ipc=host vllm/vllm-openai:latest "${DOCKER_ARGS[@]}" > /dev/null

    echo -n "⏳ Waiting for vLLM to load weights "
    while [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health)" != "200" ]; do 
        if [ "$(docker inspect -f '{{.State.Running}}' vllm_engine 2>/dev/null)" == "false" ]; then
            echo -e "\n\n❌ [FATAL] vLLM Docker container crashed! (Try lowering Context or Concurrency)."
            docker logs vllm_engine | tail -n 25
            exit 1
        fi
        echo -n "."
        sleep 5
    done
    echo -e "\n✅ vLLM is Ready!"
    docker exec vllm_engine pip install arxiv pypdf >/dev/null 2>&1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M)
LOG_FILE="${TARGET_PROJECT}_${MODEL_CHOICE}_Ctx${LOCAL_MAX_CONTEXT}_Con${VLLM_CONCURRENCY}_${TIMESTAMP}.log"

echo "🚀 Starting DeepCollector Benchmark (Saving output to $LOG_FILE)..."

# PYTHONPATH injection guarantees your local code edits run
PYTHONPATH="$(pwd)" PYTHONUNBUFFERED=1 python3 run_agent.py 2>&1 | tee "$LOG_FILE"
