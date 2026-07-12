import os
import sys
import time
import asyncio
import functools
import concurrent.futures
import re
import pandas as pd
import warnings
import glob
import io
import requests

warnings.simplefilter(action='ignore', category=FutureWarning)

BENCHMARK_MODE = os.environ.get("BENCHMARK_MODE", "LOCAL")
ENABLE_DR = os.environ.get("ENABLE_DEEP_RESEARCH_FLAG", "True") == "True"
APPLY_AMNESIA = os.environ.get("APPLY_AMNESIA", "False") == "True"

if BENCHMARK_MODE == "LOCAL":
    MODEL_ID = os.environ.get("LOCAL_MODEL_ID", "google/gemma-4-31b-it")
    MAX_CONTEXT = int(os.environ.get("LOCAL_MAX_CONTEXT", "32768"))
    os.environ["DEEPCOLLECTOR_LLM_BACKEND"] = "LOCAL_PRO"
    os.environ["DEEPCOLLECTOR_USE_VLLM"] = "True"
    os.environ["ABORT_ON_VLLM_FAILURE"] = "True"
    os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"
    os.environ["OPENAI_API_KEY"] = "sk-vllm-dummy-key"
    os.environ["VLLM_API_BASE"] = "http://localhost:8000/v1"
    PRINT_RAG_MODEL = f"vLLM Local GPU ({MODEL_ID})"
else:
    os.environ["DEEPCOLLECTOR_LLM_BACKEND"] = "GEMINI"
    os.environ["DEEPCOLLECTOR_USE_VLLM"] = "False"
    MAX_CONTEXT = 2000000 
    MODEL_ID = "Gemini-Cloud-Cascade"
    PRINT_RAG_MODEL = "Native Cloud Cascade (Gemini 3.1-Pro / 3.5-Flash)"

os.environ["DEEPCOLLECTOR_SEARCH_BACKEND"] = "GEMINI"

import gspread
import google.auth
from google.oauth2.credentials import Credentials

if BENCHMARK_MODE == "LOCAL":
    try:
        import openai
    except ImportError:
        sys.exit("❌ Missing 'openai' package. Please run: pip3 install openai")

from deepcollector.tools.research import ResearchTools
from deepcollector.core.rag_engine import RAGEngine
from deepcollector.core.agent import CatalogAgent

# =============================================================================
# 🛡️ THE ARXIV DIRECT INTERCEPTOR
# =============================================================================
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

# =============================================================================
# 🐵 UNIVERSAL BENCHMARK PATCHES
# =============================================================================

# 1. Parallel Search Patch
def patched_plan_and_execute_extraction_search(self, gaps):
    def fetch_gap(gap):
        ds = gap['Dataset']
        field = gap['Field'].replace(" (C)", "")
        eff_name = self.state.get_effective_name(ds)
        f_lower = field.lower()
        if "url" in f_lower or "source" in f_lower or "link" in f_lower: query = f"official download url repository github dataset '{eff_name}'"
        elif "variable" in f_lower or "feature" in f_lower or "dimension" in f_lower: query = f"how many variables features dimensions in '{eff_name}' dataset"
        elif "time" in f_lower or "length" in f_lower or "point" in f_lower: query = f"number of time points rows length of '{eff_name}' dataset"
        elif "location" in f_lower or "series" in f_lower: query = f"number of locations unique series in '{eff_name}' dataset"
        else: query = f"'{eff_name}' dataset {field}"
        return self.tools.tool_search_and_fetch(query, num_results=getattr(self.config, 'SEARCH_NUM_RESULTS', 8))

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch_gap, gap) for gap in gaps[:15]]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res: self.state.add_data_and_index(res)
CatalogAgent._plan_and_execute_extraction_search = patched_plan_and_execute_extraction_search

# 2. Blind Bootstrapping
if not getattr(CatalogAgent, '_bootstrap_patched', False):
    _original_bootstrap = CatalogAgent.phase_0_bootstrapping
    def patched_bootstrap(self):
        print("    🛑 [BENCHMARK MODE] Blind Bootstrapping Enabled. The Agent is blocked from reading the KB.")
        old_gspread = getattr(self.config, 'GSPREAD_AVAILABLE', False)
        self.config.GSPREAD_AVAILABLE = False
        res = _original_bootstrap(self)
        self.config.GSPREAD_AVAILABLE = old_gspread
        return res
    CatalogAgent.phase_0_bootstrapping = patched_bootstrap
    CatalogAgent._bootstrap_patched = True

# 3. The Amnesia Patch
if APPLY_AMNESIA:
    target_method = 'phase_1a_deep_research' if hasattr(CatalogAgent, 'phase_1a_deep_research') else 'phase_1_deep_research'
    if hasattr(CatalogAgent, target_method) and not getattr(CatalogAgent, '_amnesia_patched', False):
        _original_phase = getattr(CatalogAgent, target_method)
        def patched_phase(self):
            res = _original_phase(self)
            if getattr(self.config, 'ENABLE_DEEP_RESEARCH', False):
                print("\n    🛑 [AMNESIA PATCH] Wiping variables to force Local RAG...")
                fields_to_wipe = ["Domain", "Detailed Description", "Time interval between points", "Number of Time Points", "Number of Locations/Series", "Variables per Location", "Total Variables", "Comments on Data Preparation"]
                for item in self.state.catalog:
                    name = item.get("Dataset Name", {}).get("value")
                    if not name or name == "[missing]": continue
                    for field in fields_to_wipe:
                        if self.state.get_cell_data(name, field).get("value") not in ["[missing]", ""]:
                            self.state.update_cell_data(name, field, {"value": "[missing]", "confidence": 0.0})
            return res
        setattr(CatalogAgent, target_method, patched_phase)
        CatalogAgent._amnesia_patched = True

# 4. CSV Auto-Exporter
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
            safe_model = MODEL_ID.split('/')[-1]
            df.insert(0, "Project", proj_name)
            df.insert(1, "Benchmark_Model", safe_model)
            safe_proj_name = re.sub(r'[^A-Za-z0-9_\-]', '_', proj_name).strip('_')
            timestamp = time.strftime('%Y%m%d_%H%M')
            prefix = os.environ.get("DC_FILE_PREFIX")
            if prefix:
                filename = f"{prefix}_Data.csv"
            else:
                filename = f"{os.environ.get('DC_FILE_PREFIX')}_Data.csv" if os.environ.get("DC_FILE_PREFIX") else f"Bench_{safe_proj_name}_{safe_model}_{timestamp}.csv"
            
            # Use COLAB_OUTPUT_DIR if available
            out_dir = os.environ.get("COLAB_OUTPUT_DIR", "")
            if out_dir:
                filename = os.path.join(out_dir, filename)
                
            df.to_csv(filename, index=False)
            print(f"🎉 [Export] Benchmark Data saved to CSV: {filename}")
        except Exception as e:
            print(f"❌ Benchmark CSV Override Failed: {e}")
    CatalogAgent._export_run_data = patched_export
    CatalogAgent._export_patched = True

if hasattr(CatalogAgent, 'run_workflow') and not getattr(CatalogAgent, '_workflow_patched', False):
    _original_run_workflow = CatalogAgent.run_workflow
    def patched_run_workflow(self):
        res = _original_run_workflow(self)
        if not self.state.catalog:
            proj_name = str(getattr(self.config, 'CURRENT_PROJECT_NAME', 'UNKNOWN'))
            safe_model = MODEL_ID.split('/')[-1]
            safe_proj_name = re.sub(r'[^A-Za-z0-9_\-]', '_', proj_name).strip('_')
            timestamp = time.strftime('%Y%m%d_%H%M')
            prefix = os.environ.get("DC_FILE_PREFIX")
            if prefix:
                filename = f"{prefix}_Data.csv"
            else:
                filename = f"{os.environ.get('DC_FILE_PREFIX')}_Data.csv" if os.environ.get("DC_FILE_PREFIX") else f"Bench_{safe_proj_name}_{safe_model}_{timestamp}.csv"
            
            out_dir = os.environ.get("COLAB_OUTPUT_DIR", "")
            if out_dir:
                filename = os.path.join(out_dir, filename)
                
            df = pd.DataFrame([{"Project": proj_name, "Benchmark_Model": safe_model, "Dataset Name": "NONE_FOUND", "Completeness (High Conf %)": "0.0%"}])
            df.to_csv(filename, index=False)
            print(f"    ⚠️ Project found 0 datasets. Forcing empty benchmark stub: {filename}")
        return res
    CatalogAgent.run_workflow = patched_run_workflow
    CatalogAgent._workflow_patched = True

# =============================================================================
# 🐵 LOCAL HARDWARE vLLM PATCHES (Skipped entirely if running Pure Cloud)
# =============================================================================
if BENCHMARK_MODE == "LOCAL":
    _original_init = ResearchTools.__init__

    def patched_init(self, config, keys, models):
        _original_init(self, config, keys, models)
        if getattr(self.config, 'LLM_BACKEND', '') in ["LOCAL_PRO", "LOCAL_CLASSROOM"] and self.models is not None:
            setattr(self.models, 'LOCAL_MODEL', "vllm_docker_active")
            setattr(self.models, 'LOCAL_TOKENIZER', "vllm_docker_active")
            limit = int(os.environ.get("VLLM_CONCURRENCY", "8"))
            self.thread_pool.shutdown(wait=False)
            self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=limit)

    ResearchTools.__init__ = patched_init

    def vllm_generate_content_local(self, prompt: str, **kwargs):
        api_start = time.time()
        model_name_label = f"vLLM ({MODEL_ID})"
        class MockResponseWrapper:
            def __init__(self, text): self.text = text

        client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "sk-vllm-dummy-key"),
            base_url=os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1"),
            max_retries=0, timeout=1200.0 
        )
        
        # Read the strict hyperparams passed by the bash script
        dc_temp = float(os.environ.get("DC_TEMP", "0.0"))
        dc_tokens = int(os.environ.get("DC_TOKENS", "4096"))
        
        max_new_tokens = min(kwargs.get("max_new_tokens", dc_tokens), dc_tokens)
        current_prompt = prompt
        
        # 🔥 Dynamic Chat Templating & Anti-Laziness Guard
        sys_msg = (
            "You are a strict data extraction AI. You MUST output ONLY valid JSON format.\n"
            "🚨 CRITICAL CAPABILITY TEST: If you find a massive repository (like LOTSA), DO NOT summarize it. "
            "You MUST extract exactly 15 to 20 representative datasets as individual entries to prove your capabilities. "
            "Ensure the JSON array is perfectly closed."
        )

        if any(x in MODEL_ID.lower() for x in ["gemma-2", "deepseek", "qwen"]): 
            messages = [{"role": "user", "content": sys_msg + "\n\n" + current_prompt}]
        else: 
            messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": current_prompt}]
        
        for attempt in range(3):
            try:
                response = client.chat.completions.create(model=MODEL_ID, messages=messages, max_tokens=max_new_tokens, temperature=dc_temp)
                if hasattr(self, '_record_timing'): self._record_timing(model_name_label, time.time() - api_start, model_name_label)
                
                raw_res = response.choices[0].message.content
                # Strip out DeepSeek `<think>` chains so JSON parser doesn't crash
                clean_res = re.sub(r'<think>.*?</think>', '', raw_res, flags=re.DOTALL)
                clean_res = clean_res.replace("```json", "").replace("```", "").strip()
                return MockResponseWrapper(clean_res)
            except Exception as e:
                err_str = str(e).lower()
                print(f"\n❌ [vLLM Error] Attempt {attempt+1} failed! Reason: {e}")
                sys.stdout.flush()
                
                if "context length" in err_str or "input_tokens" in err_str or str(MAX_CONTEXT) in err_str:
                    chop_len = int(len(current_prompt) * 0.85)
                    current_prompt = current_prompt[:chop_len] + "\n\n...[TRUNCATED TO FIT VRAM]..."
                    if any(x in MODEL_ID.lower() for x in ["gemma-2", "deepseek", "qwen"]): messages = [{"role": "user", "content": sys_msg + "\n\n" + current_prompt}]
                    else: messages[1]["content"] = current_prompt
                    continue 
                
                if attempt == 2:
                    if os.environ.get("ABORT_ON_VLLM_FAILURE", "True") == "True":
                        print(f"\n🛑 [FATAL] vLLM permanently crashed! Exiting cleanly. Reason: {e}")
                        sys.stdout.flush()
                        os._exit(1)
                    else:
                        return self._generate_content_cascade("PRO" if "strategic planner" in prompt else "FLASH", prompt, **kwargs)
                time.sleep(2)
        
        print("\n🛑 [FATAL] vLLM API failed completely. Exiting.")
        os._exit(1)

    ResearchTools._generate_content_local = vllm_generate_content_local

    async def vllm_rag_async(self, prompt, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.thread_pool, functools.partial(self.generate_content_rag, prompt, **kwargs))
    ResearchTools.generate_content_rag_async = vllm_rag_async

    async def vllm_synthesizer_async(self, model_name, prompt, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.thread_pool, functools.partial(self.generate_content_synthesizer, model_name, prompt, **kwargs))
    ResearchTools.generate_content_synthesizer_async = vllm_synthesizer_async

    _original_run_rag_batches = RAGEngine._run_rag_batches
    async def patched_run_rag_batches(self, state, candidate_cells, retriever):
        if not getattr(self.config, 'USE_vLLM', False): return await _original_run_rag_batches(self, state, candidate_cells, retriever)
        results = []
        total_cells = len(candidate_cells)
        batch_size = int(os.environ.get("VLLM_CONCURRENCY", "8")) 

        print(f"\n    ⚙️ [vLLM RAG] Found {total_cells} tasks. Processing steadily in batches of {batch_size}...")
        sys.stdout.flush()

        for i in range(0, total_cells, batch_size):
            batch = candidate_cells[i:i+batch_size]
            print(f"      ⏳ Processing RAG Batch {(i//batch_size) + 1}/{(total_cells + batch_size - 1)//batch_size} ({len(batch)} cells)...")
            sys.stdout.flush()
            
            tasks, valid_batch = [], []
            for cell_info in batch:
                query_template = self.CATALOG_SCHEMA.get(cell_info["field_name"], {}).get("query")
                if not query_template: continue
                verified_url = self._get_cell_value(cell_info['item'], "Link to Data (Actual Source)")
                tasks.append(self._extract_cell_data_rag(
                    cell_info["dataset_name"], effective_name=state.get_effective_name(cell_info['item']),
                    field_name=cell_info["field_name"], query_template=query_template,
                    verified_url=None if verified_url == "[missing]" else verified_url, retriever=retriever
                ))
                valid_batch.append(cell_info)
            batch_results = await self._run_async_tasks(tasks)
            for cell_info, rag_result in zip(valid_batch, batch_results):
                if getattr(self.config, '_CUDA_OOM_ABORT', False): raise RuntimeError("CUDA OOM Abort")
                if isinstance(rag_result, Exception): continue
                if rag_result: results.append({"dataset_name": cell_info["dataset_name"], "field_name": cell_info["field_name"], "rag_result": rag_result})
        return results
    RAGEngine._run_rag_batches = patched_run_rag_batches

from deepcollector.config.settings import AppConfig
from deepcollector.core.executor import execute_jobs

try:
    if "google.colab" in sys.modules:
        from google.auth import default
        creds, _ = default()
        gc = gspread.authorize(creds)
    else:
        creds = Credentials.from_authorized_user_file("/home/geoffrey/Desktop/DeepKG/token.json")
        google.auth.default = lambda *args, **kwargs: (creds, "deepcollector-app")
        gc = gspread.authorize(creds)
except Exception as e:
    sys.exit(f"❌ OAuth Failed.\nError: {e}")

SECRETS = {"GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", "")}

if BENCHMARK_MODE == "LOCAL":
    ESTIMATED_CHARS = int(MAX_CONTEXT * 3.0) 
    DISCOVERY_CHARS = min(ESTIMATED_CHARS, 400000)
    CELLULAR_CHARS = min(int(ESTIMATED_CHARS * 0.8), 350000)
    FALLBACK_CHARS = min(int(ESTIMATED_CHARS * 0.5), 180000)
else:
    DISCOVERY_CHARS = 1000000
    CELLULAR_CHARS = 500000
    FALLBACK_CHARS = 250000

ENABLE_LOCAL_DR = False

config = AppConfig(
    VERBOSITY_LEVEL=1, SECRETS=SECRETS,
    LLM_BACKEND=os.environ.get("DEEPCOLLECTOR_LLM_BACKEND"), USE_vLLM=(BENCHMARK_MODE == "LOCAL"),
    SEARCH_BACKEND=os.environ.get("DEEPCOLLECTOR_SEARCH_BACKEND"),
    GOOGLE_SHEET_KB_INPUT=os.environ.get("KB_SHEET_ID"), GOOGLE_SHEET_HINTS_INPUT=os.environ.get("HINTS_SHEET_ID"),
    GOOGLE_SHEET_PROJECT_LIST_INPUT=os.environ.get("PROJECT_LIST_ID"),
    GOOGLE_DRIVE_SHEET_FOLDER_ID=os.environ.get("DRIVE_SHEET_FOLDER_ID"),
    GOOGLE_DRIVE_LOG_FOLDER_ID=os.environ.get("DRIVE_LOG_FOLDER_ID"),
    RAG_DISCOVERY_MAX_CHARS=DISCOVERY_CHARS, RAG_CELLULAR_MAX_CHARS=CELLULAR_CHARS, RAG_CELLULAR_FALLBACK_CHARS=FALLBACK_CHARS,
    
    ENABLE_DEEP_RESEARCH=ENABLE_DR,           
    ENABLE_LOCAL_DEEP_RESEARCH=ENABLE_LOCAL_DR, 
    ENABLE_GOLDEN_FASTPATH=False,    
    
    ENABLE_PREFLIGHT_CRAWLER=True, ENABLE_ARBITRATION_PROMPT=True, ENABLE_STRICT_TAXONOMY=True,
    ENABLE_MULTI_QUERY_RAG=True, ENABLE_VARIANT_MAPPING=True, ENABLE_SINGLETON_VERIFICATION=True,
    ENABLE_ORACLE_SEARCH=True
)

os.environ["DEEPCOLLECTOR_SHEET_FOLDER_ID"] = config.GOOGLE_DRIVE_SHEET_FOLDER_ID or ""
os.environ["DEEPCOLLECTOR_LOG_FOLDER_ID"] = config.GOOGLE_DRIVE_LOG_FOLDER_ID or ""
config._process_sheet_ids()

PROJECT_NAME_ARG = sys.argv[4] if len(sys.argv) > 4 else os.environ.get("TARGET_PROJECT", "UTSD")
PROJECT_NAMES = [PROJECT_NAME_ARG]

print(f"\n🖥️ ABLATION STUDY INITIALIZED...")
print(f"   - Environment: {BENCHMARK_MODE} (Model: {MODEL_ID})")
if BENCHMARK_MODE == "LOCAL":
    print(f"   - Hardware Opts: Context = {MAX_CONTEXT} tokens | Concurrency = {os.environ.get('VLLM_CONCURRENCY', 'N/A')}")
    print(f"   - LLM Settings: Temp = {os.environ.get('DC_TEMP', '0.0')} | Max Tokens = {os.environ.get('DC_TOKENS', '4096')}")
print(f"   - Deep Research: {'ACTIVE' if ENABLE_DR else 'DISABLED'}")
print(f"   - Amnesia Patch: {'ACTIVE (Wiping DR memory for RAG)' if APPLY_AMNESIA and ENABLE_DR else 'DISABLED'}")
print(f"   - Targets: {PROJECT_NAMES}\n")

execute_jobs(mode="AGENT", project_names=PROJECT_NAMES, base_config=config, gc_client=gc, dry_run=False)
