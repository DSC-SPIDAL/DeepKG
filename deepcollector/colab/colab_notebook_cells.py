# ==========================================
# CELL 1: ENVIRONMENT SETUP & DEPENDENCIES
# ==========================================
# !pip install -q llama-index vllm openai pydantic
# import os
# os.environ["GEMINI_API_KEY"] = "your_key_here"

# ==========================================
# CELL 2: CORE SYNCHRONOUS RUNNER DEFINITION
# ==========================================
# This cell
# Colab Code in 3 Cells
#
# Cell 1
# Cell 1 must be always restarted so run Ce ll 1 and then restsrt and then run Ce ll 2 and below 
#
# ==========================================
# CELL 1: RAW INSTALLATION & DEPENDENCY FIX
# ==========================================
%cd /content
!rm -rf DeepKG
!git clone https://github.com/DSC-SPIDAL/DeepKG.git

# 🔥 FIX: Uninstall deprecated packages and install the new LlamaIndex integrations + PDF tools
!pip uninstall -y google-generativeai llama-index-llms-gemini llama-index-embeddings-gemini
!pip install -qU "google-genai>=2.0.0" llama-index llama-index-llms-google-genai llama-index-embeddings-google-genai llama-index-retrievers-bm25 gspread bm25s python-dotenv nest_asyncio arxiv pypdf

print("🛑 STOP! Go to Runtime > Restart Session right now! Then run Cell 2.")
#
# Cell 2
#
# ==========================================
# CELL 2: POST-RESTART AUTHENTICATION
# ==========================================
import os
from google.colab import auth, drive, userdata

print("🔑 1. Authenticating with new packages...")
auth.authenticate_user()

print("📂 2. Mounting Drive...")
drive.mount('/content/drive', force_remount=True)

print("🔒 3. Loading Secrets...")
try:
    api_key = userdata.get('GEMINI_API_KEY')
    os.environ["GEMINI_API_KEY"] = api_key
    os.environ["GOOGLE_API_KEY"] = api_key
    print("✅ Secrets loaded!")
except Exception as e:
    print("❌ ERROR: Missing GEMINI_API_KEY in secrets.")

# Move into the folder so Cell 3 can find the code
%cd /content/DeepKG
print("✅ Environment Ready for Deep Research!")
#
# Cell 3
#
# ==========================================
# CELL 3: RUN THE TRUE CLOUD BASELINE (Anti-Recursion V3)
# ==========================================
import os
import glob
import warnings
import nest_asyncio
import re
import io
import time
import pandas as pd
import requests

warnings.filterwarnings("ignore", category=FutureWarning)
nest_asyncio.apply()

from google.colab import output, userdata, auth
import gspread
from google.auth import default
output.no_vertical_scroll()

# ---------------------------------------------------------
# 🔥 HOT-PATCH 1: Rewrite DeepCollector Imports dynamically
# ---------------------------------------------------------
print("🔧 Patching DeepCollector for the new google-genai SDK...")
for filepath in glob.glob("/content/DeepKG/**/*.py", recursive=True):
    with open(filepath, "r") as f: content = f.read()
    new_content = content.replace("llama_index.llms.gemini", "llama_index.llms.google_genai")
    new_content = new_content.replace("from llama_index.llms.google_genai import Gemini", "from llama_index.llms.google_genai import GoogleGenAI as Gemini")
    new_content = new_content.replace("llama_index.embeddings.gemini", "llama_index.embeddings.google_genai")
    new_content = new_content.replace("from llama_index.embeddings.google_genai import GeminiEmbedding", "from llama_index.embeddings.google_genai import GoogleGenAIEmbedding as GeminiEmbedding")
    if new_content != content:
        with open(filepath, "w") as f: f.write(new_content)

# ---------------------------------------------------------
# 🔥 HOT-PATCH 2: THE ARXIV DIRECT INTERCEPTOR
# ---------------------------------------------------------
from deepcollector.tools.research import ResearchTools

# ✨ ANTI-RECURSION GUARD
if hasattr(ResearchTools, 'tool_load_url') and not getattr(ResearchTools, '_arxiv_patched', False):
    _original_tool_load_url = ResearchTools.tool_load_url

    def patched_tool_load_url(self, url: str, *args, **kwargs):
        if isinstance(url, str) and "arxiv.org" in url:
            match = re.search(r'(\d{4}\.\d{4,5}(v\d+)?|[a-z\-]+/\d{7})', url)
            if match:
                paper_id = match.group(1)
                print(f"\n   📥 [ArXiv Interceptor] Identified {paper_id}. Bypassing web scraper to pull binary PDF...")
                try:
                    import arxiv
                    import pypdf
                    client = arxiv.Client()
                    search = arxiv.Search(id_list=[paper_id])
                    paper = next(client.results(search))
                    target_url = paper.pdf_url.replace("http://", "https://")

                    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                    response = requests.get(target_url, headers=headers, timeout=30)

                    if response.status_code == 200:
                        text = ""
                        pdf_reader = pypdf.PdfReader(io.BytesIO(response.content))
                        for page in pdf_reader.pages[:50]:
                            text += (page.extract_text() or "") + "\n"

                        if text and len(text.split()) >= 15:
                            print(f"   ✅ Successfully extracted {len(text)} chars from PDF into memory.")
                            return [{"url": url, "content": text, "title": f"ArXiv Paper {paper_id}", "type": "Direct Load"}]
                        else:
                            print(f"   ❌ Failed length check: only extracted {len(text)} chars.")
                    else:
                        print(f"   ❌ Failed to fetch PDF. HTTP Status: {response.status_code}")
                except Exception as e:
                    print(f"   ⚠️ ArXiv extraction error: {e}")

        return _original_tool_load_url(self, url, *args, **kwargs)

    ResearchTools.tool_load_url = patched_tool_load_url
    ResearchTools._arxiv_patched = True
    print("   ✅ ArXiv PDF Interceptor Online.")

# ---------------------------------------------------------
# EXECUTOR & CONFIGURATION
# ---------------------------------------------------------
os.environ["DEEPCOLLECTOR_USE_VLLM"] = "False"
os.environ["DEEPCOLLECTOR_LLM_BACKEND"] = "GEMINI"
os.environ["DEEPCOLLECTOR_SEARCH_BACKEND"] = "GEMINI"
os.environ["BENCHMARK_MODE"] = "CLOUD"
os.environ["ENABLE_DEEP_RESEARCH_FLAG"] = "True"
os.environ["APPLY_AMNESIA"] = "True"

def get_secret(secret_name, fallback_value=""):
    try: return userdata.get(secret_name) or fallback_value
    except: return fallback_value

print("\n🔑 Authenticating with Google Sheets...")
creds, _ = default()
gc = gspread.authorize(creds)
SECRETS = {"GEMINI_API_KEY": get_secret("GEMINI_API_KEY")}

from deepcollector.config.settings import AppConfig
from deepcollector.core.executor import execute_jobs
from deepcollector.core.agent import CatalogAgent

# ---------------------------------------------------------
# 🔥 AMNESIA & EXPORT PATCHES (With Anti-Recursion Guards)
# ---------------------------------------------------------
if hasattr(CatalogAgent, 'phase_0_bootstrapping') and not getattr(CatalogAgent, '_bootstrap_patched', False):
    _original_bootstrap = CatalogAgent.phase_0_bootstrapping
    def patched_bootstrap(self):
        print("    🛑 [BENCHMARK MODE] Blind Bootstrapping Enabled.")
        old_gspread = getattr(self.config, 'GSPREAD_AVAILABLE', False)
        self.config.GSPREAD_AVAILABLE = False
        res = _original_bootstrap(self)
        self.config.GSPREAD_AVAILABLE = old_gspread
        return res
    CatalogAgent.phase_0_bootstrapping = patched_bootstrap
    CatalogAgent._bootstrap_patched = True

if hasattr(CatalogAgent, 'phase_1a_deep_research') and not getattr(CatalogAgent, '_dr_patched', False):
    _original_phase_1 = CatalogAgent.phase_1a_deep_research
    def patched_phase_1(self):
        _original_phase_1(self)
        print("\n    🛑 [AMNESIA PATCH] Wiping variables to force RAG Extraction...")
        fields_to_wipe = ["Domain", "Detailed Description", "Time interval between points", "Number of Time Points", "Number of Locations/Series", "Variables per Location", "Total Variables", "Comments on Data Preparation"]
        for item in self.state.catalog:
            name = item.get("Dataset Name", {}).get("value")
            if not name or name == "[missing]": continue
            for field in fields_to_wipe:
                if self.state.get_cell_data(name, field).get("value") not in ["[missing]", ""]:
                    self.state.update_cell_data(name, field, {"value": "[missing]", "confidence": 0.0})
    CatalogAgent.phase_1a_deep_research = patched_phase_1
    CatalogAgent._dr_patched = True

if hasattr(CatalogAgent, '_export_run_data') and not getattr(CatalogAgent, '_export_patched', False):
    _original_export = CatalogAgent._export_run_data
    def patched_export(self):
        try: _original_export(self)
        except Exception: pass
        try:
            df = self.get_catalog_report(full_details=True)
            if df is None or df.empty:
                df = pd.DataFrame([{"Dataset Name": "NONE_FOUND", "Completeness (High Conf %)": "0.0%"}])
            proj_name = str(getattr(self.config, 'CURRENT_PROJECT_NAME', 'UNKNOWN'))
            safe_model = "Gemini-Cloud-Cascade"
            df.insert(0, "Project", proj_name)
            df.insert(1, "Benchmark_Model", safe_model)
            safe_proj_name = re.sub(r'[^A-Za-z0-9_\-]', '_', proj_name).strip('_')
            timestamp = time.strftime('%Y%m%d_%H%M')
            out_path = f"/content/drive/MyDrive/Bench_{safe_proj_name}_{safe_model}_{timestamp}.csv"
            df.to_csv(out_path, index=False)
            print(f"🎉 [Export] Benchmark Data saved to Colab Drive: {out_path}")
        except Exception as e:
            print(f"❌ Benchmark CSV Override Failed: {e}")
    CatalogAgent._export_run_data = patched_export
    CatalogAgent._export_patched = True

config = AppConfig(
    VERBOSITY_LEVEL=1, SECRETS=SECRETS,
    GOOGLE_SHEET_KB_INPUT = get_secret("KB_SHEET_ID"),
    GOOGLE_SHEET_PROJECT_LIST_INPUT = get_secret("PROJECT_LIST_ID"),
    GOOGLE_DRIVE_SHEET_FOLDER_ID = get_secret("DRIVE_SHEET_FOLDER_ID"),
    GOOGLE_DRIVE_LOG_FOLDER_ID = get_secret("DRIVE_LOG_FOLDER_ID"),
    ENABLE_DEEP_RESEARCH=True,
    ENABLE_GOLDEN_FASTPATH=False,
    ENABLE_PREFLIGHT_CRAWLER=True, ENABLE_ARBITRATION_PROMPT=True, ENABLE_STRICT_TAXONOMY=True,
    ENABLE_MULTI_QUERY_RAG=True, ENABLE_VARIANT_MAPPING=True, ENABLE_SINGLETON_VERIFICATION=True, ENABLE_ORACLE_SEARCH=True
)

os.environ["DEEPCOLLECTOR_SHEET_FOLDER_ID"] = config.GOOGLE_DRIVE_SHEET_FOLDER_ID or ""
os.environ["DEEPCOLLECTOR_LOG_FOLDER_ID"] = config.GOOGLE_DRIVE_LOG_FOLDER_ID or ""
config.recalculate_runtime_parameters()
config._process_sheet_ids()

# ==========================================
# 🎯 EXECUTE THE RUN
# ==========================================
PROJECT_NAMES = ["UTSD"] # Just starting with UTSD to verify the fix!

print(f"\n🚀 Firing TRUE CLOUD BASELINE for {PROJECT_NAMES}...")
%cd /content/DeepKG
try:
    execute_jobs(mode="AGENT", project_names=PROJECT_NAMES, base_config=config, gc_client=gc, dry_run=False)
except Exception as e:
    print(f"❌ Main Execution Failed: {e}")
#
# END Cell 3
