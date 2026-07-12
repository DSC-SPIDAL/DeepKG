## 🚀 Hybrid DGX Architecture (Local vLLM + Cloud Gemini)

DeepCollector supports a "Hybrid" execution mode optimized for enterprise hardware (NVIDIA DGX systems). This allows you to offload external web searches to Google Cloud APIs to prevent IP bans, while keeping heavy RAG text extraction on local A100 GPUs for maximum throughput and privacy.

### Handling Enterprise Driver Mismatches
Enterprise supercomputers often run stable, older NVIDIA drivers (e.g., CUDA 12.0) which clash with bleeding-edge PyTorch libraries. DeepCollector sidesteps this entirely by using the official `vLLM` Docker Container with NVIDIA Container Toolkit compatibility.

The provided `start.sh` automatically:
1. Bypasses the weak DGX Display GPU, mapping specifically to the A100 compute cards (`NVIDIA_VISIBLE_DEVICES="0,1,2,4"`).
2. Mounts the local Hugging Face cache to prevent re-downloading massive models (e.g., the 55GB `Gemma-2-27B`).
3. Executes a "Smart Polling" health check against `http://localhost:8000/health` to pause Python execution until the AI model is actively mapped into VRAM and ready to receive requests.

### Google Drive OAuth Export
To bypass Google Cloud's 0-byte Service Account quota limits for Google Drive uploads, this system uses User OAuth (`token.json`). The application authenticates natively as the user to drop exported CSVs directly into personal Google Drive folders.
