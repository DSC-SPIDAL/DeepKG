# =============================================================================
# DEEPCOLLECTOR: COLAB EXECUTION CELL
# Instructions: Copy and paste this entire file into a single Google Colab Cell.
# Ensure you have run the Environment Setup (Cell 1) first.
# =============================================================================

import os
import sys
import nest_asyncio

# 1. 🛠️ Re-apply memory patches lost during any runtime restarts
nest_asyncio.apply()

# 2. 📁 Ensure Drive & Python Path are active
from google.colab import drive
try:
    print("📁 Mounting Google Drive...")
    drive.mount('/content/drive')
except:
    pass

if "/content/DeepKG" not in sys.path:
    sys.path.insert(0, "/content/DeepKG")

# 3. 🌐 Cloud Enforcement: Ensure vLLM is OFF and Cloud Gemini is ON
os.environ["DEEPCOLLECTOR_USE_VLLM"] = "False"
os.environ["DEEPCOLLECTOR_LLM_BACKEND"] = "GEMINI"
os.environ["DEEPCOLLECTOR_SEARCH_BACKEND"] = "GEMINI"

# 4. 🔐 Load Secrets & Auth
from google.colab import output, userdata, auth
import gspread
from google.auth import default
output.no_vertical_scroll()

def get_secret(secret_name, fallback_value=""):
    try: return userdata.get(secret_name) or fallback_value
    except: return fallback_value

print("🔑 Authenticating with Google Sheets...")
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
print("✅ Auth Successful!")

SECRETS = {"GEMINI_API_KEY": get_secret("GEMINI_API_KEY")}

# 5. ⚙️ Initialize Config
from deepcollector.config.settings import AppConfig
from deepcollector.core.executor import execute_jobs

config = AppConfig(
    VERBOSITY_LEVEL=1,
    SECRETS=SECRETS,
    GOOGLE_SHEET_KB_INPUT = get_secret("KB_SHEET_ID"),
    GOOGLE_SHEET_HINTS_INPUT = get_secret("HINTS_SHEET_ID"),
    GOOGLE_SHEET_PROJECT_LIST_INPUT = get_secret("PROJECT_LIST_ID"),
    GOOGLE_DRIVE_SHEET_FOLDER_ID = get_secret("DRIVE_SHEET_FOLDER_ID"),
    GOOGLE_DRIVE_LOG_FOLDER_ID = get_secret("DRIVE_LOG_FOLDER_ID"),
    
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
config.recalculate_runtime_parameters()
config._process_sheet_ids()

# =========================================================
# 👉 CHANGE YOUR PROJECTS HERE:
# =========================================================
# Change this list to whatever you want (e.g., ["M6", "LOTSA"])
# They must match the names in your Canonical Projects sheet.
PROJECT_NAMES = ["Tempo"] 

# 6. 🚀 Run the Pipeline!
print(f"\n🚀 Firing DeepCollector Pipeline for {len(PROJECT_NAMES)} projects...")
execute_jobs(
    mode="AGENT",
    project_names=PROJECT_NAMES,
    base_config=config,
    gc_client=gc,
    dry_run=False
)
