import os
import sys
import gspread
from google.auth import default

from deepcollector.config.settings import AppConfig
from deepcollector.core.executor import execute_jobs

# =============================================================================
# DEEPCOLLECTOR: CLOUD AGENT DRIVER
# =============================================================================

# 1. Authenticate with Google
try:
    print("🔑 [Cloud Agent] Authenticating with Google Cloud...")
    gc = gspread.service_account(filename="/home/geoffrey/Desktop/DeepKG/credentials.json")
    print("✅ [Cloud Agent] Google Auth Successful!")
except Exception as e:
    sys.exit(f"❌ [Cloud Agent] Google Auth Failed.\nError: {e}")

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

# Respect system environment variables set by start_cloud.sh
os.environ["DEEPCOLLECTOR_LLM_BACKEND"] = config.LLM_BACKEND
os.environ["DEEPCOLLECTOR_USE_VLLM"] = str(config.USE_vLLM)
config._process_sheet_ids()

# 3. Job Execution Parameters (DEDICATED CLOUD LIST)
MODE = "AGENT"
PROJECT_NAMES = ["Tempo"]

# 4. Fire the Executor
print(f"☁️ Firing CLOUD Agent for {len(PROJECT_NAMES)} projects...")
execute_jobs(
    mode=MODE, 
    project_names=PROJECT_NAMES, 
    base_config=config, 
    gc_client=gc, 
    dry_run=False
)