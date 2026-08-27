# =============================================================================
# V280: Research Tools (Original Colab ArXiv Logic Restored)
# =============================================================================
import os
import requests
import time
import re
import io
import sys
import gc
import traceback
import asyncio
import json
import functools
import concurrent.futures
import threading
import logging
from collections import defaultdict
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

logging.getLogger("google_genai.models").setLevel(logging.ERROR)
logging.getLogger("google_genai._api_client").setLevel(logging.ERROR)

try:
    import torch
except ImportError:
    torch = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    from deepcollector.utils.profiler import profiler
    from deepcollector.utils.initialization import get_network_retry_strategy, get_gemini_retry_strategy, HEADERS
except ImportError:
    class DummyProfiler:
        def track(self, category): return lambda f: f
    profiler = DummyProfiler()
    def get_network_retry_strategy(verbosity): return lambda f: f
    def get_gemini_retry_strategy(verbosity): return lambda f: f
    HEADERS = {}

try:
    from google.genai import types
except ImportError:
    types = None

class CellExtractionSchema(BaseModel):
    value: str = Field(description="The extracted value or '[missing]' if not found in context.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0.")
    rationale: str = Field(description="Concise rationale or citation from the text.")

class DiscoveredDatasetItem(BaseModel):
    dataset_name: str = Field(description="Exact canonical or variant name of the dataset.")
    type: str = Field(description="Entity type: Real-World Dataset, Synthetic Dataset, Collection, or Provider.")
    confidence: float = Field(description="Relevance confidence score between 0.0 and 1.0.")
    rationale: str = Field(description="Brief justification for why this dataset belongs to the project.")

class DiscoveryCatalogSchema(BaseModel):
    discovered_datasets: List[DiscoveredDatasetItem] = Field(description="List of all discovered datasets.")

class SearchQueriesSchema(BaseModel):
    queries: List[str] = Field(description="List of targeted search query strings.")


class ResearchTools:
    MAX_FETCH_LENGTH = 1000000
    MAX_PDF_PAGES = 50

    def __init__(self, config: Any, keys: Any, models: Any):
        self.config = config
        self.keys = keys
        self.models = models
        self.PdfReader = getattr(config, 'PdfReader', None)
        self.verbosity = getattr(config, 'VERBOSITY_LEVEL', 1)
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=30)
        self.search_failure_count = 0
        self.model_usage_stats = defaultdict(lambda: {"count": 0, "time": 0.0, "time_sq": 0.0})
        self.pool_lock = threading.Lock()
        self.local_llm_lock = threading.Lock()
        self.slow_strikes = defaultdict(int)
        self.SLOW_THRESHOLD_SEC = 50.0
        self.MAX_SLOW_STRIKES = 2
        self.SEARCH_ENABLED = bool(models and getattr(models, 'CLIENT', None)) and bool(types)
        self.NETWORK_RETRY_STRATEGY = get_network_retry_strategy(self.verbosity)
        self.GEMINI_API_RETRY_STRATEGY = get_gemini_retry_strategy(self.verbosity)
        self._fetch_page_content_impl = self.NETWORK_RETRY_STRATEGY(self._fetch_page_content_impl)

    def _get_active_pool(self, pool_name: str) -> list:
        with self.pool_lock:
            pool = self.models.PRO_POOL if pool_name == "PRO" else self.models.FLASH_POOL
            if len(pool) > 1:
                leader = str(pool[0])
                if self.slow_strikes[leader] >= self.MAX_SLOW_STRIKES:
                    demoted = pool.pop(0)
                    pool.append(demoted)
                    self.slow_strikes[str(demoted)] = 0
                    if self.verbosity >= 1: print(f"\n    ⚠️ [Health Monitor] '{demoted}' hung >{self.SLOW_THRESHOLD_SEC}s {self.MAX_SLOW_STRIKES} times. Demoted.")
            return list(pool)

    def _record_timing(self, target_model: str, duration: float, tracker_key: str):
        t_model_str = str(target_model)
        t_key_str = str(tracker_key)
        with self.pool_lock:
            self.model_usage_stats[t_key_str]["time"] += duration
            self.model_usage_stats[t_key_str]["time_sq"] += duration ** 2
            self.model_usage_stats[t_key_str]["count"] += 1
            if duration > self.SLOW_THRESHOLD_SEC: self.slow_strikes[t_model_str] += 1
            else: self.slow_strikes[t_model_str] = 0

    @profiler.track("Tool: Web Fetching")
    def _fetch_page_content(self, url: str, timeout=15, minimal_cleaning=False) -> str:
        return self._fetch_page_content_impl(url, timeout, minimal_cleaning)

    def _fetch_page_content_impl(self, url: str, timeout=15, minimal_cleaning=False) -> str:
        def truncate(text):
            if len(text) > self.MAX_FETCH_LENGTH: return text[:self.MAX_FETCH_LENGTH] + "... [TRUNCATED]"
            return text

        if url.startswith('/content/drive/') or url.startswith('~'):
            try:
                if os.environ.get("BENCHMARK_MODE") == "LOCAL" and ("PDFGems" in url or "/content/drive/" in url):
                    filename = os.path.basename(url)
                    url = os.path.expanduser(f"~/Desktop/DeepKG/PDFGems/{filename}")
                if not os.path.exists(url): return ""
                with open(url, 'rb') as f: content = f.read()
                content_type = 'application/pdf' if url.lower().endswith('.pdf') else 'text/plain'
            except Exception: return ""
        else:
            if "github.com" in url and "/blob/" not in url and "/tree/" not in url:
                try:
                    parts = url.rstrip('/').split('/')
                    if len(parts) >= 5:
                        user, repo = parts[-2], parts[-1]
                        for branch in ['main', 'master']:
                            raw_url = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/README.md"
                            try:
                                response = requests.get(raw_url, headers=HEADERS, timeout=5)
                                if response.status_code == 200: return truncate(response.text)
                            except Exception: pass
                except Exception: pass

            if "arxiv.org" in url:
                url = url.replace("/abs/", "/pdf/").replace("http://", "https://")
                if not url.lower().endswith(".pdf") and '/pdf/' in url: url += ".pdf"

            try:
                with requests.get(url, headers=HEADERS, timeout=timeout, stream=True) as response:
                    response.raise_for_status()
                    content = b""
                    start_time = time.time()
                    for chunk in response.iter_content(chunk_size=8192):
                        if time.time() - start_time > timeout: break
                        content += chunk
                        if len(content) > self.MAX_FETCH_LENGTH: break
                    content_type = response.headers.get('Content-Type', '').lower()
            except Exception: return ""

        text = ""
        if 'application/pdf' in content_type or (url.lower().endswith('.pdf') and 'text/html' not in content_type):
            if self.PdfReader:
                try:
                    pdf_reader = self.PdfReader(io.BytesIO(content))
                    for page in pdf_reader.pages[:self.MAX_PDF_PAGES]: text += (page.extract_text() or "") + "\n"
                except Exception: pass
        else:
            if not BeautifulSoup: text = content.decode('utf-8', errors='ignore')
            else:
                try: parser = 'lxml'
                except ImportError: parser = 'html.parser'
                try:
                    soup = BeautifulSoup(content, parser)
                    if minimal_cleaning: text = soup.get_text(separator=' ')
                    else:
                        elements = soup.find_all(['p', 'table', 'li', 'h1', 'h2', 'h3', 'h4', 'div', 'span', 'pre', 'code'])
                        text = ' '.join([elem.get_text(strip=True, separator=' ') for elem in elements])
                        if len(text.split()) < 50: text = soup.get_text(strip=True, separator=' ')
                except Exception: text = content.decode('utf-8', errors='ignore')

        if text is None: text = ""
        if not minimal_cleaning: text = re.sub(r'\s+', ' ', text).strip()
        return truncate(text)

    def tool_pre_flight_crawl(self, text: str, max_links: int = 5) -> List[str]:
        if not text: return []
        prompt = (
            "Extract the most important data-related outbound URLs from the following text. "
            "Focus specifically on links to GitHub repositories, HuggingFace data cards, Zenodo, "
            "Kaggle, or supplementary PDFs that might contain detailed dataset configurations or technical appendices.\n\n"
            f"TEXT:\n{text[:25000]}\n\n"
            'Return ONLY a JSON list of URL strings. E.g., ["https://github.com/...", "https://huggingface.co/..."] Return maximum 5 URLs.'
        )
        try:
            model = self.models.MODEL_PLANNER
            response = self.generate_content_planner(model, prompt)
            urls = self._extract_json_robustly(response.text)
            if isinstance(urls, list):
                urls = [u for u in urls if isinstance(u, str) and u.startswith('http')]
                return urls[:max_links]
        except Exception: pass
        return []

    @profiler.track("Tool: Gemini Search")
    def tool_search_and_fetch(self, query: str, num_results=None) -> List[Dict[str, str]]:
        query = self._clean_query_string(query)
        if self.verbosity >= 1: print(f"🌐 [Tool: Search/Fetch] Query: '{query}'")
        if num_results is None: num_results = getattr(self.config, 'SEARCH_NUM_RESULTS', 10)
        if not self.SEARCH_ENABLED: return []
        try:
            results = self._perform_gemini_search(query, num_results)
            if not results:
                simple_query = self._simplify_query(query)
                if simple_query != query and len(simple_query) > 3:
                    if self.verbosity >= 1: print(f"    ⚠️ 0 Results. Retrying with simplified query: '{simple_query}'")
                    results = self._perform_gemini_search(simple_query, num_results)
            if results:
                if self.verbosity >= 1: print(f"    ✅ Gemini Search returned {len(results)} usable results.")
                return results[:num_results]
            else:
                self.search_failure_count += 1
                if self.verbosity >= 1: print("    ❌ Gemini Search returned 0 usable results.")
        except Exception: self.search_failure_count += 1
        return []

    def _clean_query_string(self, query: str) -> str: return str(query).replace("**", "").replace("__", "").strip()

    def _perform_gemini_search(self, query, num_results):
        prompt = (
            f"Perform a Google Search for: '{query}'. "
            f"Return the top {num_results} most relevant results. "
            "You MUST output raw HTTP links to datasets, Githubs, or Archives. "
            "Provide the Title, URL, and Summary."
        )
        pool = self._get_active_pool("PRO")
        for current_idx, target_model in enumerate(pool):
            target_model_str = str(target_model)
            api_start = time.time()
            tracker_key = f"{target_model_str} (Search)"
            try:
                if types:
                    cfg = types.GenerateContentConfig(temperature=0.0, tools=[types.Tool(google_search=types.GoogleSearch())])
                    if "3.1-pro" in target_model_str or "3-pro" in target_model_str: cfg.thinking_config = types.ThinkingConfig(thinking_budget=4096)
                else: cfg = None

                response = self.models.CLIENT.models.generate_content(model=target_model_str, contents=str(prompt), config=cfg)
                duration = time.time() - api_start
                self._record_timing(target_model_str, duration, tracker_key)

                results = []
                text_content = response.text if response.text else ""
                md_links = re.findall(r'\[(.*?)\]\((https?://[^\)]+)\)', text_content)
                for title, url in md_links: results.append({"url": url.strip(), "title": title.strip(), "content": "Extracted via Markdown", "type": "Gemini Grounding"})
                
                # Hex-encoded \x22 and \x27
                raw_urls = re.findall(r'(https?://[^\s>\x22\x27\)]+)', text_content)
                for url in raw_urls:
                    clean_url = url.rstrip('.,;:')
                    if not any(r['url'] == clean_url for r in results):
                        results.append({"url": clean_url, "title": "Raw Extracted Link", "content": "Extracted via Omni-Regex", "type": "Gemini Grounding"})

                if not results and response.candidates:
                    cand = response.candidates
                    if cand[0].grounding_metadata and cand[0].grounding_metadata.grounding_chunks:
                        for chunk in (cand[0].grounding_metadata.grounding_chunks or []):
                            if getattr(chunk, 'web', None):
                                uri = getattr(chunk.web, 'uri', '')
                                if "vertexaisearch" in uri or "scholar.google" in uri: continue
                                results.append({"url": uri, "title": getattr(chunk.web, 'title', 'Untitled'), "content": "Grounding Source Metadata", "type": "Gemini Metadata"})
                return results
            except Exception as e:
                duration = time.time() - api_start
                self._record_timing(target_model_str, duration, tracker_key)
                error_str = str(e).lower()
                if self.verbosity >= 1: print(f"    ⚠️ [Search Cascade] '{target_model_str}' failed: {type(e).__name__} - {str(e)[:100]}")
                if ("404" in error_str or "not found" in error_str or "429" in error_str or "quota" in error_str or "503" in error_str or "timeout" in error_str or duration > self.SLOW_THRESHOLD_SEC):
                    time.sleep(1.0)
                    if current_idx < len(pool) - 1: continue
                    return []
                else:
                    if current_idx < len(pool) - 1: continue
                    return []
        return []

    def _simplify_query(self, query):
        query = str(query).replace('"', '').replace(" OR ", " ").replace(" AND ", " ")
        query = re.sub(r'site:\S+', '', query)
        query = re.sub(r'^(Look for|Search for|Find|Identify|Locate)\s+', '', query, flags=re.IGNORECASE)
        query = re.sub(r'\s+(with its attributes|with attributes|and provide|details about).*$', '', query, flags=re.IGNORECASE)
        return query.strip()

    def _extract_json_robustly(self, text: str) -> Any:
        if not text or text == "[missing]": return []
        text = str(text).strip()
        text = re.sub(r'^.*?```json\s*', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'^.*?```\s*', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'\s*```.*$', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = text.strip()
        start_obj = text.find('{')
        start_arr = text.find('[')
        try:
            if start_obj != -1 and (start_arr == -1 or start_obj < start_arr):
                end_obj = text.rfind('}')
                if end_obj != -1: return json.loads(text[start_obj:end_obj+1])
            elif start_arr != -1:
                end_arr = text.rfind(']')
                if end_arr != -1: return json.loads(text[start_arr:end_arr+1])
        except Exception: pass
        result = {}
        val_match = re.search(r'"value"\s*:\s*\x22?([^\x22\}]+)\x22?', text, re.IGNORECASE)
        conf_match = re.search(r'"confidence"\s*:\s*([\d\.]+)', text, re.IGNORECASE)
        rat_match = re.search(r'"rationale"\s*:\s*\x22?([^\x22\}]+)\x22?', text, re.IGNORECASE)
        if val_match: result['value'] = val_match.group(1).strip()
        if conf_match: result['confidence'] = float(conf_match.group(1))
        if rat_match: result['rationale'] = rat_match.group(1).strip()
        if result: return result
        return []

    def tool_load_url(self, url: str) -> List[Dict[str, str]]:
        if isinstance(url, str) and "arxiv.org" in url:
            match = re.search(r'(\d{4}\.\d{4,5}(v\d+)?|[a-z\-]+/\d{7})', url)
            if match:
                paper_id = match.group(1)
                if self.verbosity >= 1: print(f"\n   📥 [ArXiv Interceptor] Identified {paper_id}. Bypassing web scraper to pull binary PDF...")
                try:
                    import arxiv
                    import io
                    import pypdf
                    client = arxiv.Client()
                    search = arxiv.Search(id_list=[paper_id])
                    paper = next(client.results(search))
                    
                    # Original working Colab resolution
                    target_url = paper.pdf_url.replace("http://", "https://")

                    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                    response = requests.get(target_url, headers=headers, timeout=30)
                    if response.status_code == 200:
                        text = ""
                        pdf_reader = pypdf.PdfReader(io.BytesIO(response.content))
                        for page in pdf_reader.pages[:50]: text += (page.extract_text() or "") + "\n"
                        if text and len(text.split()) >= 15:
                            return [{"url": url, "content": text, "title": f"ArXiv Paper {paper_id}", "type": "Direct Load"}]
                except Exception as e:
                    if self.verbosity >= 1: print(f"   ⚠️ ArXiv direct pull error: {e}")

        try:
            content = self._fetch_page_content(url)
            if content and len(content.split()) >= 15:
                return [{"url": url, "content": content, "title": f"Direct Load: {url[:50]}", "type": "Direct Load"}]
        except Exception: pass
        return []

    def tool_inspect_data_file(self, url: str, ddi_tool: Any = None) -> Dict:
        if ddi_tool:
            try: return ddi_tool.inspect_remote_file(url)
            except Exception as e: return {"status": "error", "error": str(e)}
        try:
            from deepcollector.tools.ddi import DDITools
            temp_tool = DDITools(self.config)
            return temp_tool.inspect_remote_file(url)
        except Exception as e: return {"status": "skipped", "error": f"DDI Tool missing: {e}"}

    def _generate_content_local(self, prompt: str, **kwargs):
        is_vllm = getattr(self.config, 'USE_vLLM', False) or os.environ.get("DEEPCOLLECTOR_USE_VLLM", "False") == "True"
        sys_msg = "You are a strict data extraction AI. You MUST output ONLY valid JSON format.\nExtract requested attributes accurately without conversational filler."
        if is_vllm:
            import openai
            api_start = time.time()
            model_id = os.environ.get("LOCAL_MODEL_ID", "google/gemma-4-31b-it")
            model_name_label = f"vLLM ({model_id})"
            class MockResponseWrapper:
                def __init__(self, text): self.text = text
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-vllm-dummy-key"), base_url=os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1"), max_retries=0, timeout=1200.0)
            dc_temp = float(os.environ.get("DC_TEMP", "0.0"))
            dc_tokens = int(os.environ.get("DC_TOKENS", "4096"))
            max_new_tokens = min(kwargs.get("max_new_tokens", dc_tokens), dc_tokens)
            current_prompt = prompt
            force_json = kwargs.get("force_json", False)

            for attempt in range(3):
                try:
                    if any(x in model_id.lower() for x in ["gemma-2", "deepseek", "qwen", "llama", "command-r"]): 
                        messages = [{"role": "user", "content": sys_msg + "\n\n" + current_prompt}]
                    else: 
                        messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": current_prompt}]
                    
                    payload = {"model": model_id, "messages": messages, "max_tokens": max_new_tokens, "temperature": dc_temp}
                    if force_json and not any(x in model_id.lower() for x in ["deepseek", "command-r", "qwen"]):
                        payload["response_format"] = {"type": "json_object"}
                    try: response = client.chat.completions.create(**payload)
                    except Exception as api_err:
                        if "format" in str(api_err).lower() or "json" in str(api_err).lower() or "400" in str(api_err).lower():
                            payload.pop("response_format", None)
                            response = client.chat.completions.create(**payload)
                        else: raise api_err

                    if hasattr(self, '_record_timing'): self._record_timing(model_name_label, time.time() - api_start, model_name_label)
                    raw_res = response.choices[0].message.content
                    clean_res = re.sub(r'<think>.*?</think>', '', raw_res, flags=re.DOTALL)
                    clean_res = clean_res.replace("```json", "").replace("```", "").strip()
                    return MockResponseWrapper(clean_res)
                except Exception as e:
                    err_str = str(e).lower()
                    if "context length" in err_str or "input_tokens" in err_str:
                        inst_idx = current_prompt.rfind("Instructions:")
                        if inst_idx == -1: inst_idx = current_prompt.rfind("Format EXACTLY")
                        if inst_idx != -1:
                            instructions = current_prompt[inst_idx:]
                            body = current_prompt[:inst_idx]
                            chop_len = int(len(body) * 0.80)
                            current_prompt = body[:chop_len] + "\n\n...[TRUNCATED TO FIT VRAM]...\n\n" + instructions
                        else:
                            chop_len = int(len(current_prompt) * 0.85)
                            current_prompt = current_prompt[:chop_len] + "\n\n...[TRUNCATED TO FIT VRAM]..."
                        continue 
                    if attempt == 2:
                        if os.environ.get("ABORT_ON_VLLM_FAILURE", "True") == "True": os._exit(1)
                        else: return self._generate_content_cascade("PRO" if "strategic planner" in prompt else "FLASH", prompt, **kwargs)
                    time.sleep(2)
            os._exit(1)

        api_start = time.time()
        model_name_label = f"Gemma ({getattr(self.config, 'LLM_BACKEND', 'LOCAL')})"
        class MockResponseWrapper:
            def __init__(self, text): self.text = text
        with self.local_llm_lock:
            inputs = None; outputs = None; out_tokens_list = []
            model = getattr(self.models, 'LOCAL_MODEL', None)
            tokenizer = getattr(self.models, 'LOCAL_TOKENIZER', None)
            if not model or not tokenizer or isinstance(model, str): return self._generate_content_cascade("PRO" if "strategic planner" in prompt else "FLASH", prompt, **kwargs)
            chat = [{"role": "user", "content": sys_msg + "\n\n" + prompt}]
            try: formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            except Exception: formatted_prompt = f"<bos><start_of_turn>user\n{sys_msg}\n\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            current_max_len = 32000
            req_max_new = min(kwargs.get("max_new_tokens", 1024), 2048)
            while current_max_len >= 2000:
                try:
                    if torch is not None and torch.cuda.is_available(): torch.cuda.empty_cache(); torch.cuda.ipc_collect()
                    gc.collect()
                    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=current_max_len).to(model.device)
                    terminators = [tokenizer.eos_token_id]
                    if hasattr(tokenizer, "get_vocab"):
                        vocab = tokenizer.get_vocab()
                        for t in ["<end_of_turn>", "<|eot_id|>", "<|im_end|>"]:
                            if t in vocab: terminators.append(vocab[t])
                    with torch.inference_mode():
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            outputs = model.generate(
                                **inputs, max_new_tokens=req_max_new, do_sample=False, use_cache=True,
                                pad_token_id=tokenizer.eos_token_id, eos_token_id=terminators,
                                output_attentions=False, output_hidden_states=False, return_dict_in_generate=False
                            )
                    prompt_len = inputs["input_ids"].shape[1]
                    out_tensor = outputs[0][prompt_len:]
                    out_tokens_list = out_tensor.cpu().tolist()
                    break
                except Exception as e:
                    err_str = str(e).lower()
                    if "cuda out of memory" in err_str or "outofmemoryerror" in err_str or "alloc" in err_str:
                        current_max_len = int(current_max_len * 0.75)
                        if 'inputs' in locals() and inputs is not None: del inputs
                        if 'outputs' in locals() and outputs is not None: del outputs
                        if hasattr(sys, 'last_traceback'): sys.last_traceback = None
                        if hasattr(sys, 'last_type'): sys.last_type = None
                        if hasattr(sys, 'last_value'): sys.last_value = None
                        if hasattr(e, "__traceback__") and e.__traceback__: traceback.clear_frames(e.__traceback__)
                        del e
                        continue
                    else: return self._generate_content_cascade("PRO" if "strategic planner" in prompt else "FLASH", prompt, **kwargs)
            if not out_tokens_list: return self._generate_content_cascade("PRO" if "strategic planner" in prompt else "FLASH", prompt, **kwargs)
            response_text = tokenizer.decode(out_tokens_list, skip_special_tokens=True)
            del prompt
            duration = time.time() - api_start
            self._record_timing(model_name_label, duration, model_name_label)
            return MockResponseWrapper(response_text.replace("```json", "").replace("```", "").strip())

    def _generate_content_cascade(self, pool_name: str, prompt: str, **kwargs):
        if not getattr(self.models, 'CLIENT', None): raise ValueError("Gemini Client not initialized.")
        pool = self._get_active_pool(str(pool_name))
        force_json = kwargs.pop("force_json", False)
        max_tokens = kwargs.pop("max_new_tokens", None)
        base_config = kwargs.pop("config", None)
        for k in ["do_sample", "temperature", "top_p", "top_k", "repetition_penalty", "return_dict_in_generate", "output_scores", "stop"]: kwargs.pop(k, None)
        sys_msg = "You are a strict data extraction AI. You MUST output ONLY valid JSON format without markdown code blocks."
        for current_idx, target_model in enumerate(list(pool)):
            api_start = time.time()
            target_model_str = str(target_model)
            current_kwargs = dict(kwargs)
            if types:
                current_config = types.GenerateContentConfig()
                if base_config:
                    for attr in ['candidate_count', 'max_output_tokens', 'stop_sequences', 'response_mime_type', 'response_schema', 'tools']:
                        if hasattr(base_config, attr) and getattr(base_config, attr) is not None: setattr(current_config, attr, getattr(base_config, attr))
                if max_tokens: current_config.max_output_tokens = int(max_tokens)
                if "3.1-pro" in target_model_str or "3-pro" in target_model_str: current_config.thinking_config = types.ThinkingConfig(thinking_budget=4096)
                else: current_config.thinking_config = None
                if force_json or getattr(current_config, "response_schema", None):
                    current_config.temperature = 0.0
                    current_config.response_mime_type = "application/json"
                    current_config.system_instruction = sys_msg
                current_kwargs["config"] = current_config
            try:
                response = self.models.CLIENT.models.generate_content(model=target_model_str, contents=str(prompt), **current_kwargs)
                duration = time.time() - api_start
                self._record_timing(target_model_str, duration, target_model_str)
                return response
            except Exception as e:
                duration = time.time() - api_start
                self._record_timing(target_model_str, duration, target_model_str)
                error_str = str(e).lower()
                if self.verbosity >= 1: print(f"    ⚠️ [Cascade] '{target_model_str}' failed: {type(e).__name__} - {str(e)[:150]}")
                if ("404" in error_str or "not found" in error_str or "429" in error_str or "quota" in error_str or "503" in error_str or "timeout" in error_str or duration > self.SLOW_THRESHOLD_SEC):
                    time.sleep(1.0)
                    if current_idx < len(pool) - 1: 
                        if self.verbosity >= 1: print(f"    ➡️ Cascading seamlessly to next model: {pool[current_idx+1]}")
                        continue
                    else: raise ResourceWarning(f"All models in {pool_name} pool exhausted. Last Error: {e}")
                else: 
                    if current_idx < len(pool) - 1:
                        if self.verbosity >= 1: print(f"    ➡️ Unrecognized error. Safety cascade to next model: {pool[current_idx+1]}")
                        continue
                    raise e
        raise ResourceWarning(f"All models in {pool_name} pool exhausted.")

    @profiler.track("LLM: Planner")
    def generate_content_planner(self, model_name, prompt, **kwargs):
        if getattr(self.config, 'LLM_BACKEND', '') in ["LOCAL_PRO", "LOCAL_CLASSROOM"] and (hasattr(self.models, 'LOCAL_MODEL') or getattr(self.config, 'USE_vLLM', False)): return self._generate_content_local(prompt, **kwargs)
        if types and "config" not in kwargs: kwargs["config"] = types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json", response_schema=SearchQueriesSchema)
        return self._generate_content_cascade("PRO", prompt, force_json=True, **kwargs)

    def generate_content_synthesizer(self, model_name, prompt, **kwargs):
        if getattr(self.config, 'LLM_BACKEND', '') in ["LOCAL_PRO", "LOCAL_CLASSROOM"] and (hasattr(self.models, 'LOCAL_MODEL') or getattr(self.config, 'USE_vLLM', False)): return self._generate_content_local(prompt, **kwargs)
        if types and "config" not in kwargs: kwargs["config"] = types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json", response_schema=DiscoveryCatalogSchema)
        return self._generate_content_cascade("PRO", prompt, force_json=True, **kwargs)

    @profiler.track("LLM: Standard")
    def generate_content_standard(self, model_name, prompt, **kwargs):
        if getattr(self.config, 'LLM_BACKEND', '') in ["LOCAL_PRO", "LOCAL_CLASSROOM"] and (hasattr(self.models, 'LOCAL_MODEL') or getattr(self.config, 'USE_vLLM', False)): return self._generate_content_local(prompt, **kwargs)
        if types and "config" not in kwargs: kwargs["config"] = types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json")
        return self._generate_content_cascade("PRO", prompt, force_json=True, **kwargs)

    @profiler.track("LLM: RAG")
    def generate_content_rag(self, prompt, **kwargs):
        if getattr(self.config, 'LLM_BACKEND', '') in ["LOCAL_PRO", "LOCAL_CLASSROOM"] and (hasattr(self.models, 'LOCAL_MODEL') or getattr(self.config, 'USE_vLLM', False)): return self._generate_content_local(prompt, **kwargs)
        if types and "config" not in kwargs: kwargs["config"] = types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json", response_schema=CellExtractionSchema)
        return self._generate_content_cascade("FLASH", prompt, force_json=True, **kwargs)

    async def generate_content_synthesizer_async(self, model_name, prompt, **kwargs):
        if getattr(self.config, 'LLM_BACKEND', '') in ["LOCAL_PRO", "LOCAL_CLASSROOM"] and (hasattr(self.models, 'LOCAL_MODEL') or getattr(self.config, 'USE_vLLM', False)): return self.generate_content_synthesizer(model_name, prompt, **kwargs)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.thread_pool, functools.partial(self.generate_content_synthesizer, model_name, prompt, **kwargs))

    async def generate_content_rag_async(self, prompt, **kwargs):
        provider = os.environ.get("TARGET_PROVIDER", getattr(self.config, 'TARGET_PROVIDER', 'GEMINI'))
        model = os.environ.get("TARGET_MODEL", getattr(self.config, 'TARGET_MODEL', ''))
        is_local = getattr(self.config, 'LLM_BACKEND', '') in ["LOCAL_PRO", "LOCAL_CLASSROOM"]
        use_vllm = getattr(self.config, 'USE_vLLM', False) or os.environ.get("DEEPCOLLECTOR_USE_VLLM", "False") == "True"
        max_t = kwargs.pop("max_new_tokens", 512)
        force_json = kwargs.pop("force_json", True) 
        for bad_k in ["do_sample", "temperature", "top_p", "top_k", "repetition_penalty", "return_dict_in_generate", "output_scores", "stop", "config", "force_json"]: kwargs.pop(bad_k, None)
        sys_msg = "You are a strict data extraction AI. You MUST output ONLY valid JSON format without markdown code blocks."
        class MockResp:
            def __init__(self, text): self.text = text
        api_start = time.time()

        if is_local and use_vllm:
            import openai
            client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-vllm-dummy-key"), base_url=os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1"), max_retries=0, timeout=1200.0)
            model_id = os.environ.get("LOCAL_MODEL_ID", "google/gemma-4-31b-it")
            dc_temp = float(os.environ.get("DC_TEMP", "0.0"))
            
            if any(x in model_id.lower() for x in ["gemma-2", "deepseek"]): messages = [{"role": "user", "content": sys_msg + "\n\n" + prompt}]
            else: messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}]
            
            for attempt in range(3):
                try:
                    payload = {"model": model_id, "messages": messages, "max_tokens": max_t, "temperature": dc_temp}
                    if force_json and not any(x in model_id.lower() for x in ["deepseek", "command-r", "qwen"]): payload["response_format"] = {"type": "json_object"}
                    try: resp = await client.chat.completions.create(**payload)
                    except Exception as api_err:
                        if "format" in str(api_err).lower() or "json" in str(api_err).lower() or "400" in str(api_err).lower():
                            payload.pop("response_format", None)
                            resp = await client.chat.completions.create(**payload)
                        else: raise api_err
                    self._record_timing(f"vLLM ({model_id})", time.time() - api_start, f"vLLM ({model_id})")
                    clean_res = re.sub(r'<think>.*?</think>', '', resp.choices[0].message.content, flags=re.DOTALL)
                    return MockResp(clean_res.replace("```json", "").replace("```", "").strip())
                except Exception as e:
                    if "context length" in str(e).lower() or "input_tokens" in str(e).lower():
                        inst_idx = prompt.rfind("Instructions:")
                        if inst_idx == -1: inst_idx = prompt.rfind("Format EXACTLY")
                        if inst_idx != -1:
                            instructions = prompt[inst_idx:]
                            body = prompt[:inst_idx]
                            chop_len = int(len(body) * 0.80)
                            prompt = body[:chop_len] + "\n\n...[TRUNCATED]...\n\n" + instructions
                        else: prompt = prompt[:int(len(prompt)*0.85)] + "\n\n...[TRUNCATED]..."
                        if any(x in model_id.lower() for x in ["gemma-2", "deepseek"]): messages[0]["content"] = sys_msg + "\n\n" + prompt
                        else: messages[1]["content"] = prompt
                        continue
                    if attempt == 2: return MockResp("[missing]")
                    await asyncio.sleep(2)

        elif provider == "OPENAI":
            import openai
            client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=1800.0)
            for attempt in range(4):
                try:
                    payload = {"model": model, "messages": [{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": max_t}
                    if force_json: payload["response_format"] = {"type": "json_object"}
                    if "sol" in model.lower() or "-o" in model.lower(): payload = {"model": model, "messages": [{"role": "user", "content": f"{sys_msg}\n\n{prompt}"}]}
                    resp = await client.chat.completions.create(**payload)
                    self._record_timing(model, time.time() - api_start, model)
                    return MockResp(resp.choices[0].message.content.replace("```json", "").replace("```", "").strip())
                except Exception as e:
                    if "429" in str(e).lower() or "quota" in str(e).lower(): await asyncio.sleep((2**attempt)*3 + 2); continue
                    if attempt == 3: raise e

        elif provider == "ANTHROPIC":
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=1800.0)
            for attempt in range(4):
                try:
                    resp = await client.messages.create(model=model, system=sys_msg, messages=[{"role": "user", "content": prompt}], max_tokens=max_t)
                    self._record_timing(model, time.time() - api_start, model)
                    extracted_text = next((block.text for block in resp.content if getattr(block, "type", "") == "text"), "")
                    return MockResp(extracted_text.replace("```json", "").replace("```", "").strip())
                except Exception as e:
                    if "429" in str(e).lower() or "overloaded" in str(e).lower(): await asyncio.sleep((2**attempt)*3 + 2); continue
                    if attempt == 3: raise e

        if types:
            cfg = types.GenerateContentConfig(max_output_tokens=max_t, temperature=0.0, system_instruction=sys_msg)
            if force_json:
                cfg.response_mime_type = "application/json"
                cfg.response_schema = CellExtractionSchema
            kwargs["config"] = cfg

        loop = asyncio.get_running_loop()
        safe_prompt = f"{sys_msg}\n\n{prompt}"
        return await loop.run_in_executor(self.thread_pool, functools.partial(self._generate_content_cascade, "FLASH", safe_prompt, force_json=force_json, **kwargs))

print("✅ deepcollector/tools/research.py LOADED (V280: Original Colab ArXiv Logic Restored)")
