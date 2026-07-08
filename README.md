import os
import sys
import gspread
from google.auth import default

from deepcollector.config.settings import AppConfig
from deepcollector.core.executor import execute_jobs

# =============================================================================
# DEEPCOLLECTOR: LOCAL HARDWARE AGENT DRIVER
# =============================================================================

# 1. Authenticate with Google
try:
    print("🔑 [Local Agent] Authenticating with Google Cloud...")
    gc = gspread.service_account(filename="/home/geoffrey/Desktop/DeepKG/credentials.json")
    print("✅ [Local Agent] Google Auth Successful!")
except Exception as e:
    sys.exit(f"❌ [Local Agent] Google Auth Failed.\nError: {e}")

# 2. Configuration & Secrets Injection
SECRETS = {"GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", "")}

config = AppConfig(
    VERBOSITY_LEVEL=1,
    SECRETS=SECRETS,
    
    GOOGLE_SHEET_KB_INPUT=os.environ.get("KB_SHEET_ID"),
    GOOGLE_SHEET_HINTS_INPUT=os.environ.get("HINTS_SHEET_ID"),
    GOOGLE_SHEET_PROJECT_LIST_INPUT=os.environ.get("PROJECT_LIST_ID"),
    GOOGLE_DRIVE_SHEET_FOLDER_ID=os.environ.get("DRIVE_SHEET_FOLDER_ID"),
    GOOGLE_DRIVE_LOG_FOLDER_ID=os.environ.get("DRIVE_LOG_FOLDER_ID"),
    
    ENABLE_DEEP_RESEARCH=True,
    ENABLE_PREFLIGHT_CRAWLER=True,
    ENABLE_ARBITRATION_PROMPT=True,
    ENABLE_STRICT_TAXONOMY=True,
    ENABLE_MULTI_QUERY_RAG=True,
    ENABLE_VARIANT_MAPPING=True,
    ENABLE_SINGLETON_VERIFICATION=True,
    ENABLE_ORACLE_SEARCH=True
)

os.environ["DEEPCOLLECTOR_SHEET_FOLDER_ID"] = config.GOOGLE_DRIVE_SHEET_FOLDER_ID or ""
os.environ["DEEPCOLLECTOR_LOG_FOLDER_ID"] = config.GOOGLE_DRIVE_LOG_FOLDER_ID or ""

# Enforce the Local Hardware settings mapped by start.sh
os.environ["DEEPCOLLECTOR_LLM_BACKEND"] = config.LLM_BACKEND
os.environ["DEEPCOLLECTOR_USE_VLLM"] = str(config.USE_vLLM)
config._process_sheet_ids()

# 3. Job Execution Parameters (DEDICATED LOCAL BATCH LIST)
MODE = "AGENT"

# 👉 CHANGE YOUR LOCAL VLLM DGX PROJECTS HERE:
PROJECT_NAMES = ["AEON", "M6", "LOTSA"]

# 4. Fire the Executor
print(f"🖥️ Firing LOCAL HARDWARE Agent for {len(PROJECT_NAMES)} projects...")
execute_jobs(
    mode=MODE, 
    project_names=PROJECT_NAMES, 
    base_config=config, 
    gc_client=gc, 
    dry_run=False
)
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
