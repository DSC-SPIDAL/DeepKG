import os
import gspread
from google.auth import default
from deepcollector.config.settings import AppConfig
from deepcollector.core.executor import execute_jobs

# 1. Satisfy the framework's pre-flight checks with a dummy key
os.environ["GEMINI_API_KEY"] = "dummy_key_for_local_oom_test"

# 2. Force Local VLLM parameters for Gemma at 64K
os.environ["DEEPCOLLECTOR_USE_VLLM"] = "True"
os.environ["DEEPCOLLECTOR_LLM_BACKEND"] = "gemma-4-31b-it"
os.environ["BENCHMARK_MODE"] = "LOCAL"
os.environ["APPLY_AMNESIA"] = "True"
os.environ["MAX_CONTEXT"] = "65536"

# 3. Authenticate
print("🔑 Authenticating...")
creds, _ = default()
gc = gspread.authorize(creds)

# 4. Configure App
config = AppConfig(
    VERBOSITY_LEVEL=1,
    ENABLE_DEEP_RESEARCH=True
)

print("\n🚀 Firing UTSD Gemma-4-31B [64K] OOM Verification Run...")
try:
    execute_jobs(mode="AGENT", project_names=["UTSD"], base_config=config, gc_client=gc, dry_run=False)
except Exception as e:
    print(f"\n❌ CRASH CONFIRMED: {e}")
