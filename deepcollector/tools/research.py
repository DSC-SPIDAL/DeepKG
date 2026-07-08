# =============================================================================
# V261: Research Tools (Full Untruncated Native PyTorch + 404/429 Cascade & Thinking Payload)
# =============================================================================
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

try:
    import torch
except ImportError:
    torch = None

from collections import defaultdict
from typing import List, Dict, Any, Optional
from tenacity import RetryError
from requests.exceptions import RequestException, HTTPError, ConnectionError, Timeout, SSLError

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
        self._generate_content_cascade = self.GEMINI_API_RETRY_STRATEGY(self._generate_content_cascade)

    def _get_active_pool(self, pool_name: str) -> list:
        with self.pool_lock:
            pool = self.models.PRO_POOL if pool_name == "PRO" else self.models.FLASH_POOL
            if len(pool) > 1:
                leader = str(pool[0])
                if self.slow_strikes[leader] >= self.MAX_SLOW_STRIKES:
                    demoted = pool.pop(0)
                    pool.append(demoted)
                    self.slow_strikes[str(demoted)] = 0
                    if self.verbosity >= 1:
                        print(f"\n    ⚠️ [Health Monitor] '{demoted}' hung >{self.SLOW_THRESHOLD_SEC}s {self.MAX_SLOW_STRIKES} times. Demoted.")
            return list(pool)

    def _record_timing(self, target_model: str, duration: float, tracker_key: str):
        t_model_str = str(target_model)
        t_key_str = str(tracker_key)
        with self.pool_lock:
            self.model_usage_stats[t_key_str]["time"] += duration
            self.model_usage_stats[t_key_str]["time_sq"] += duration ** 2
            self.model_usage_stats[t_key_str]["count"] += 1
            if duration > self.SLOW_THRESHOLD_SEC:
                self.slow_strikes[t_model_str] += 1
            else:
                self.slow_strikes[t_model_str] = 0

    @profiler.track("Tool: Web Fetching")
    def _fetch_page_content(self, url: str, timeout=15, minimal_cleaning=False) -> str:
        return self._fetch_page_content_impl(url, timeout, minimal_cleaning)

    def _fetch_page_content_impl(self, url: str, timeout=15, minimal_cleaning=False) -> str:
        def truncate(text):
             if len(text) > self.MAX_FETCH_LENGTH:
                 return text[:self.MAX_FETCH_LENGTH] + "... [TRUNCATED]"
             return text

        if url.startswith('/content/drive/'):
            try:
                if self.verbosity >= 2: print(f"    📁 [Local Fetch] Reading {url} directly from Drive...")
                with open(url, 'rb') as f:
                    content = f.read()
                content_type = 'application/pdf' if url.lower().endswith('.pdf') else 'text/plain'
            except Exception:
                return ""
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
                                if response.status_code == 200:
                                    return truncate(response.text)
                            except Exception:
                                pass
                 except Exception:
                     pass

            if "arxiv.org" in url:
                url = url.replace("/abs/", "/pdf/").replace("http://", "https://")
                if not url.lower().endswith(".pdf") and '/pdf/' in url:
                    url += ".pdf"

            try:
                with requests.get(url, headers=HEADERS, timeout=timeout, stream=True) as response:
                    response.raise_for_status()
                    content = b""
                    start_time = time.time()
                    for chunk in response.iter_content(chunk_size=8192):
                        if time.time() - start_time > timeout:
                            break
                        content += chunk
                        if len(content) > self.MAX_FETCH_LENGTH:
                            break
                    content_type = response.headers.get('Content-Type', '').lower()
            except Exception:
                return ""

        text = ""
        if 'application/pdf' in content_type or (url.lower().endswith('.pdf') and 'text/html' not in content_type):
            if self.PdfReader:
                try:
                    pdf_reader = self.PdfReader(io.BytesIO(content))
                    for page in pdf_reader.pages[:self.MAX_PDF_PAGES]:
                        text += (page.extract_text() or "") + "\n"
                except Exception:
                    pass
        else:
            if not BeautifulSoup:
                text = content.decode('utf-8', errors='ignore')
            else:
                try:
                    parser = 'lxml'
                except ImportError:
                    parser = 'html.parser'
                try:
                    soup = BeautifulSoup(content, parser)
                    if minimal_cleaning:
                        text = soup.get_text(separator=' ')
                    else:
                        elements = soup.find_all(['p', 'table', 'li', 'h1', 'h2', 'h3', 'h4', 'div', 'span', 'pre', 'code'])
                        text = ' '.join([elem.get_text(strip=True, separator=' ') for elem in elements])
                        if len(text.split()) < 50:
                            text = soup.get_text(strip=True, separator=' ')
                except Exception:
                    text = content.decode('utf-8', errors='ignore')

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
            "Return ONLY a JSON list of URL strings. E.g., [\"https://github.com/...\", \"https://huggingface.co/...\"] Return maximum 5 URLs."
        )
        try:
            model = self.models.MODEL_PLANNER
            response = self.generate_content_planner(model, prompt)
            urls = self._extract_json_robustly(response.text)
            if isinstance(urls, list):
                urls = [u for u in urls if isinstance(u, str) and u.startswith('http')]
                return urls[:max_links]
        except Exception as e:
            if self.verbosity >= 2:
                print(f"    ⚠️ [Pre-Flight Crawler Error] {e}")
        return []

    @profiler.track("Tool: Gemini Search")
    def tool_search_and_fetch(self, query: str, num_results=None) -> List[Dict[str, str]]:
        query = self._clean_query_string(query)
        if self.verbosity >= 1: print(f"🌐 [Tool: Search/Fetch] Query: '{query}'")
        if num_results is None: num_results = getattr(self.config, 'SEARCH_NUM_RESULTS', 10)

        if getattr(self.config, 'SEARCH_BACKEND', 'GEMINI') == "SEARXNG":
            try:
                from deepcollector.utils.initialization import SearXNGClient
                client = SearXNGClient(base_url=getattr(self.config, 'SEARXNG_URL', "http://localhost:8080"))
                results = client.search(query, max_results=num_results)
                if results: return results
                else:
                    simple_query = self._simplify_query(query)
                    if simple_query != query and len(simple_query) > 3:
                        results = client.search(simple_query, max_results=num_results)
                        if results: return results
            except Exception as e:
                if self.verbosity >= 1: print(f"    ⚠️ SearXNG Failed ({e}). Falling back to Gemini Search...")

        if not self.SEARCH_ENABLED:
            return []

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
        except Exception as e:
            self.search_failure_count += 1

        return []

    def _clean_query_string(self, query: str) -> str:
        return str(query).replace("**", "").replace("__", "").strip()

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
                # 🚀 Apply Thinking Config + Search Tool Safely
                if types:
                    cfg = types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())])
                    if "3.1-pro" in target_model_str or "3-pro" in target_model_str:
                        cfg.thinking_config = types.ThinkingConfig(thinking_budget=4096)
                else:
                    cfg = None

                response = self.models.CLIENT.models.generate_content(
                    model=target_model_str,
                    contents=str(prompt),
                    config=cfg
                )

                duration = time.time() - api_start
                self._record_timing(target_model_str, duration, tracker_key)

                results = []
                text_content = response.text if response.text else ""

                md_links = re.findall(r'\[(.*?)\]\((https?://[^\)]+)\)', text_content)
                for title, url in md_links:
                    results.append({"url": url.strip(), "title": title.strip(), "content": "Extracted via Markdown", "type": "Gemini Grounding"})

                raw_urls = re.findall(r'(https?://[^\s>\"\'\)]+)', text_content)
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
                
                if self.verbosity >= 1:
                    print(f"    ⚠️ [Search Cascade] '{target_model_str}' failed: {type(e).__name__} - {str(e)[:100]}")
                
                if ("404" in error_str or "not found" in error_str or 
                    "429" in error_str or "quota" in error_str or 
                    "503" in error_str or "timeout" in error_str or 
                    duration > self.SLOW_THRESHOLD_SEC):
                    
                    time.sleep(1.0)
                    if current_idx < len(pool) - 1: 
                        if self.verbosity >= 1: print(f"    ➡️ Search cascading to next model: {pool[current_idx+1]}")
                        continue
                    else: return []
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
        if not isinstance(text, str): text = str(text)

        text = re.sub(r',\s*([\]}])', r'\1', text)
        text = re.split(r'\n\s*thought\b', text, maxsplit=1, flags=re.IGNORECASE)[0]
        text = re.split(r'\n\s*Wait, I noticed', text, maxsplit=1, flags=re.IGNORECASE)[0]
        text = re.split(r'======', text, maxsplit=1)[0]

        match = re.search(r"`{3}(?:json)?\n(.*?)\n`{3}", text, re.DOTALL | re.IGNORECASE)
        if match:
            try: return json.loads(match.group(1))
            except Exception: pass

        def extract_balanced(s, open_char, close_char):
            start = s.find(open_char)
            if start == -1: return None
            count = 0
            for i in range(start, len(s)):
                if s[i] == open_char: count += 1
                elif s[i] == close_char: count -= 1
                if count == 0:
                    try: return json.loads(s[start:i+1])
                    except: return None
            return None

        res = extract_balanced(text, '[', ']')
        if res is not None: return res

        res = extract_balanced(text, '{', '}')
        if res is not None: return res

        return []

    def tool_load_url(self, url: str) -> List[Dict[str, str]]:
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
        except Exception as e:
            return {"status": "skipped", "error": f"DDI Tool missing: {e}"}

    def _generate_content_local(self, prompt: str, **kwargs):
        """COLAB ARCHIVAL: Pure Native PyTorch 4-Bit Waterfall."""
        api_start = time.time()
        model_name_label = f"Gemma ({getattr(self.config, 'LLM_BACKEND', 'LOCAL')})"

        class MockResponseWrapper:
            def __init__(self, text): self.text = text

        with self.local_llm_lock:
            inputs = None
            outputs = None
            out_tokens_list = []

            model = getattr(self.models, 'LOCAL_MODEL', None)
            tokenizer = getattr(self.models, 'LOCAL_TOKENIZER', None)

            if not model or not tokenizer or isinstance(model, str):
                if getattr(self.config, 'VERBOSITY_LEVEL', 1) >= 2:
                    print(f"    ⚠️ Local PyTorch model not loaded (Found {type(model)}). Falling back to Cloud Gemini...")
                return self._generate_content_cascade("PRO" if "strategic planner" in prompt else "FLASH", prompt, **kwargs)

            sys_prefix = "You are a strict data extraction AI. You MUST output ONLY valid JSON format.\n\n"
            chat = [{"role": "user", "content": sys_prefix + prompt}]

            try:
                formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            except Exception:
                formatted_prompt = f"<bos><start_of_turn>user\n{sys_prefix}{prompt}<end_of_turn>\n<start_of_turn>model\n"

            # 4-bit gives us tons of room. Start with massive limit and back off dynamically.
            current_max_len = 32000
            req_max_new = min(kwargs.get("max_new_tokens", 1024), 2048)

            while current_max_len >= 2000:
                try:
                    if torch is not None and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    gc.collect()

                    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=current_max_len).to(model.device)

                    terminators = [tokenizer.eos_token_id]
                    if hasattr(tokenizer, "get_vocab"):
                        vocab = tokenizer.get_vocab()
                        for t in ["<end_of_turn>", "<|eot_id|>", "<|im_end|>"]:
                            if t in vocab:
                                terminators.append(vocab[t])

                    with torch.inference_mode():
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=req_max_new,
                                do_sample=False, # Greedy decoding (NaN immune)
                                use_cache=True,
                                pad_token_id=tokenizer.eos_token_id,
                                eos_token_id=terminators,
                                output_attentions=False,
                                output_hidden_states=False,
                                return_dict_in_generate=False
                            )

                    # SAFE PYTORCH SLICING
                    prompt_len = inputs["input_ids"].shape[1]
                    out_tensor = outputs[0][prompt_len:]
                    out_tokens_list = out_tensor.cpu().tolist()
                    break

                except Exception as e:
                    err_str = str(e).lower()
                    if "cuda out of memory" in err_str or "outofmemoryerror" in err_str or "alloc" in err_str:
                        if getattr(self.config, 'VERBOSITY_LEVEL', 1) >= 1:
                            print(f"    ⚠️ [VRAM Ceiling Hit] OOM at {current_max_len} tokens. Truncating by 25% and retrying...")

                        current_max_len = int(current_max_len * 0.75)

                        if 'inputs' in locals() and inputs is not None: del inputs
                        if 'outputs' in locals() and outputs is not None: del outputs

                        if hasattr(sys, 'last_traceback'): sys.last_traceback = None
                        if hasattr(sys, 'last_type'): sys.last_type = None
                        if hasattr(sys, 'last_value'): sys.last_value = None
                        if hasattr(e, "__traceback__") and e.__traceback__: traceback.clear_frames(e.__traceback__)
                        del e
                        continue
                    else:
                        if getattr(self.config, 'VERBOSITY_LEVEL', 1) >= 1:
                            print(f"    ⚠️ [Local PyTorch Error] {e}. Falling back to Cloud...")
                        return self._generate_content_cascade("PRO" if "strategic planner" in prompt else "FLASH", prompt, **kwargs)

            if not out_tokens_list:
                return self._generate_content_cascade("PRO" if "strategic planner" in prompt else "FLASH", prompt, **kwargs)

            response_text = tokenizer.decode(out_tokens_list, skip_special_tokens=True)
            del prompt
            duration = time.time() - api_start
            self._record_timing(model_name_label, duration, model_name_label)

            clean_res = response_text.replace("`" * 3 + "json", "").replace("`" * 3, "").strip()
            return MockResponseWrapper(clean_res)

    def _generate_content_cascade(self, pool_name: str, prompt: str, **kwargs):
        if not getattr(self.models, 'CLIENT', None): raise ValueError("Gemini Client not initialized.")
        pool = self._get_active_pool(str(pool_name))

        max_tokens = kwargs.pop("max_new_tokens", None)
        for k in ["do_sample", "temperature", "top_p", "top_k", "repetition_penalty", "return_dict_in_generate", "output_scores", "stop"]:
            kwargs.pop(k, None)

        base_config = kwargs.get("config", None)

        for current_idx, target_model in enumerate(list(pool)):
            api_start = time.time()
            target_model_str = str(target_model)
            
            current_kwargs = dict(kwargs)
            
            # ✅ Manual deep copy to prevent SDK schema contamination during fallbacks
            if types:
                current_config = types.GenerateContentConfig()
                if base_config:
                    for attr in ['temperature', 'top_p', 'top_k', 'candidate_count', 'max_output_tokens', 'stop_sequences', 'response_mime_type', 'response_schema', 'system_instruction', 'tools']:
                        if hasattr(base_config, attr) and getattr(base_config, attr) is not None:
                            setattr(current_config, attr, getattr(base_config, attr))
                            
                if max_tokens:
                    current_config.max_output_tokens = int(max_tokens)
                
                # 🚀 INJECT THINKING CONFIG EXCLUSIVELY FOR 3.1-PRO
                if "3.1-pro" in target_model_str or "3-pro" in target_model_str:
                    current_config.thinking_config = types.ThinkingConfig(thinking_budget=4096)
                else:
                    current_config.thinking_config = None
                
                current_kwargs["config"] = current_config

            try:
                response = self.models.CLIENT.models.generate_content(
                    model=target_model_str, 
                    contents=str(prompt), 
                    **current_kwargs
                )
                duration = time.time() - api_start
                self._record_timing(target_model_str, duration, target_model_str)
                return response
                
            except Exception as e:
                duration = time.time() - api_start
                self._record_timing(target_model_str, duration, target_model_str)

                error_str = str(e).lower()
                if self.verbosity >= 1:
                    print(f"    ⚠️ [Cascade] '{target_model_str}' failed: {type(e).__name__} - {str(e)[:150]}")
                
                # 🛡️ BULLETPROOF FALLBACK CASCADE
                if ("404" in error_str or "not found" in error_str or 
                    "429" in error_str or "quota" in error_str or 
                    "503" in error_str or "timeout" in error_str or 
                    duration > self.SLOW_THRESHOLD_SEC):
                    
                    time.sleep(1.0)
                    if current_idx < len(pool) - 1: 
                        if self.verbosity >= 1:
                            print(f"    ➡️ Cascading seamlessly to next model: {pool[current_idx+1]}")
                        continue
                    else: 
                        raise ResourceWarning(f"All models in {pool_name} pool exhausted. Last Error: {e}")
                else: 
                    if current_idx < len(pool) - 1:
                        if self.verbosity >= 1:
                            print(f"    ➡️ Unrecognized error. Safety cascade to next model: {pool[current_idx+1]}")
                        continue
                    raise e
                    
        raise ResourceWarning(f"All models in {pool_name} pool exhausted.")

    @profiler.track("LLM: Planner")
    def generate_content_planner(self, model_name, prompt, **kwargs):
        if getattr(self.config, 'LLM_BACKEND', '') in ["LOCAL_PRO", "LOCAL_CLASSROOM"] and hasattr(self.models, 'LOCAL_MODEL'):
            return self._generate_content_local(prompt, **kwargs)
        return self._generate_content_cascade("PRO", prompt, **kwargs)

    def generate_content_synthesizer(self, model_name, prompt, **kwargs):
        if getattr(self.config, 'LLM_BACKEND', '') in ["LOCAL_PRO", "LOCAL_CLASSROOM"] and hasattr(self.models, 'LOCAL_MODEL'):
            return self._generate_content_local(prompt, **kwargs)
        return self._generate_content_cascade("PRO", prompt, **kwargs)

    @profiler.track("LLM: Standard")
    def generate_content_standard(self, model_name, prompt, **kwargs):
        if getattr(self.config, 'LLM_BACKEND', '') in ["LOCAL_PRO", "LOCAL_CLASSROOM"] and hasattr(self.models, 'LOCAL_MODEL'):
            return self._generate_content_local(prompt, **kwargs)
        return self._generate_content_cascade("PRO", prompt, **kwargs)

    @profiler.track("LLM: RAG")
    def generate_content_rag(self, prompt, **kwargs):
        if getattr(self.config, 'LLM_BACKEND', '') in ["LOCAL_PRO", "LOCAL_CLASSROOM"] and hasattr(self.models, 'LOCAL_MODEL'):
            return self._generate_content_local(prompt, **kwargs)
        return self._generate_content_cascade("FLASH", prompt, **kwargs)

    async def generate_content_synthesizer_async(self, model_name, prompt, **kwargs):
        if getattr(self.config, 'LLM_BACKEND', '') in ["LOCAL_PRO", "LOCAL_CLASSROOM"]:
            return self.generate_content_synthesizer(model_name, prompt, **kwargs)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.thread_pool, functools.partial(self.generate_content_synthesizer, model_name, prompt, **kwargs))

    async def generate_content_rag_async(self, prompt, **kwargs):
        if getattr(self.config, 'LLM_BACKEND', '') in ["LOCAL_PRO", "LOCAL_CLASSROOM"]:
            return self.generate_content_rag(prompt, **kwargs)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.thread_pool, functools.partial(self.generate_content_rag, prompt, **kwargs))