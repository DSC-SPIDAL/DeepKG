# =============================================================================
# V186: Initialization (DGX / Gemma-Compatible & 404 Deprecation Safe)
# =============================================================================
import os
import sys
import time
import warnings
import logging
import requests
import gc
from typing import Tuple, Dict, Any, List, Optional

try:
    import torch
except ImportError:
    torch = None

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:32"

# --- SILENCE NOISY THIRD-PARTY WARNINGS ---
warnings.filterwarnings("ignore", category=UserWarning, module="google.generativeai")
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", module="llama_index.retrievers.bm25.base")
warnings.filterwarnings("ignore", module="pypdf._reader")
warnings.filterwarnings("ignore", module="google_auth_httplib2")
warnings.filterwarnings("ignore", message=".*Interactions usage is experimental.*")

logging.getLogger('bm25s').setLevel(logging.ERROR)
logging.getLogger('pypdf').setLevel(logging.ERROR)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# =============================================================================
# --- SEARXNG CLIENT ---
# =============================================================================
class SearXNGClient:
    """Robust client for interacting with SearXNG proxy arrays."""
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url.rstrip("/")
        self.fallback_urls = [
            "https://searx.be",
            "https://searx.tiekoetter.com",
            "https://search.ononoki.org",
            "https://searx.work"
        ]

    def search(self, query: str, max_results: int = 7) -> List[Dict[str, str]]:
        print(f"    🌐 [Tool: SearXNG] Executing Query: '{query}'")

        urls_to_try = [self.base_url] + self.fallback_urls

        for url in urls_to_try:
            try:
                params = {
                    "q": query,
                    "format": "json",
                    "engines": "google,bing,duckduckgo,wikipedia,github",
                    "language": "en"
                }

                response = requests.get(f"{url}/", params=params, headers=HEADERS, timeout=15)

                if response.status_code == 404:
                    response = requests.get(f"{url}/search", params=params, headers=HEADERS, timeout=15)

                response.raise_for_status()
                data = response.json()

                results = []
                for item in data.get("results", [])[:max_results]:
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "content": item.get("content", "")
                    })

                if results:
                    if url != self.base_url:
                        print(f"       ⚠️ Localhost 404/Refused. Automatically failed over to public instance: {url}")
                    print(f"       ✅ SearXNG returned {len(results)} usable results.")
                    return results

            except Exception as e:
                continue

        print(f"       ❌ SearXNG Search failed across all attempted instances (Local & Public).")
        return []

# =============================================================================
# --- RETRY STRATEGIES ---
# =============================================================================
def get_network_retry_strategy(verbosity=1):
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    from requests.exceptions import RequestException, ConnectionError, Timeout, HTTPError

    def log_retry(retry_state):
        if verbosity >= 2:
            print(f"    ⚠️ [Network] Retry {retry_state.attempt_number}...")

    return retry(
        retry=retry_if_exception_type((ConnectionError, Timeout, HTTPError, RequestException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=log_retry,
        reraise=True
    )

def get_gemini_retry_strategy(verbosity=1):
    from tenacity import retry, stop_after_attempt, wait_exponential

    def log_retry(retry_state):
        if verbosity >= 1:
            ex = retry_state.outcome.exception()
            if ex:
                msg = str(ex)[:100]
            else:
                msg = "Unknown"
            print(f"    ⚠️ [LLM API] Retry {retry_state.attempt_number}: {msg}...")

    return retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        before_sleep=log_retry,
        reraise=True
    )

class SystemMonitor:
    @staticmethod
    def print_vram_status(label=""):
        try:
            if torch is not None and torch.cuda.is_available():
                reserved = torch.cuda.memory_reserved(0) / (1024**3)
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                free = total - reserved
                print(f"       📊 VRAM Status [{label}]: {allocated:.1f}GB Allocated | {reserved:.1f}GB Reserved | {free:.1f}GB Free (out of {total:.1f}GB)")
        except Exception:
            pass

# =============================================================================
# --- CORE INITIALIZATION ---
# =============================================================================
def initialize_apis(config: Any) -> Tuple[Dict[str, str], Any]:
    keys = {}
    backend = getattr(config, 'LLM_BACKEND', 'GEMINI')
    use_vllm = getattr(config, 'USE_vLLM', False)

    # 1. ALWAYS Initialize Gemini (Required for Agentic Planning/Routing)
    api_key = config.SECRETS.get("GEMINI_API_KEY")
    if not api_key:
        try:
            from google.colab import userdata
            api_key = userdata.get("GEMINI_API_KEY")
        except ImportError:
            pass
        except Exception:
            pass

    if not api_key:
        print("❌ GEMINI_API_KEY not found in Secrets or Config.")
        sys.exit("🛑 ABORTING: Missing Gemini API Key.")

    keys["GEMINI"] = api_key
    os.environ["GOOGLE_API_KEY"] = api_key

    try:
        from google import genai
        client = genai.Client(api_key=api_key)

        candidates_pro = [
            "gemini-3.1-pro-preview",
            "gemini-3-pro-preview",
            "gemini-2.5-pro",
            "gemini-pro-latest"
        ]

        candidates_flash = [
            "gemini-3.5-flash",
            "gemini-3-flash-preview",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite"
        ]

        def verify_pool(model_list, pool_name):
            verified = []
            print(f"    🔍 Verifying {pool_name} Pool...")

            for model in model_list:
                try:
                    # Quick ping to check access permissions
                    client.models.count_tokens(model=model, contents="Test")
                    verified.append(model)
                except Exception as e:
                    err_str = str(e).lower()
                    if "404" in err_str or "not found" in err_str:
                        print(f"       ⚠️ Skipping {model}: Model deprecated or not found (404).")
                    else:
                        print(f"       ⚠️ Skipping {model}: API access error -> {e}")

            if verified:
                print(f"       ✅ Verified {len(verified)} models: {verified}")
            else:
                print(f"       ❌ No models verified in {pool_name} pool.")

            return verified

        verified_pro = verify_pool(candidates_pro, "Pro (Reasoning)")
        verified_flash = verify_pool(candidates_flash, "Flash (Extraction)")

        if not verified_pro and not verified_flash:
            raise RuntimeError("CRITICAL: ALL models failed verification. Check API Key validity.")

        if not verified_pro:
            print("    ⚠️ WARNING: Pro pool empty. Promoting Flash models to Pro role.")
            verified_pro = verified_flash

        if not verified_flash:
            print("    ⚠️ WARNING: Flash pool empty. Using Pro models for extraction.")
            verified_flash = verified_pro

        class Models:
            CLIENT = client
            # Use only the verified lists
            PRO_POOL = verified_pro
            FLASH_POOL = verified_flash

            CURRENT_PRO_INDEX = 0
            CURRENT_FLASH_INDEX = 0

            # Roles - Defaults to the leader of the verified pool
            MODEL_PLANNER = verified_pro[0]
            MODEL_SYNTHESIZER = verified_pro[0]
            MODEL_STANDARD = verified_flash[0]
            MODEL_RAG_PRIMARY = verified_flash[0]

        print(f"    ✅ Gemini Client initialized.")
        print(f"    🎭 Active Roles:")
        print(f"       - Planner/Search: {Models.MODEL_PLANNER}")
        print(f"       - RAG Extraction: {Models.MODEL_RAG_PRIMARY}")

        return keys, Models

    except ImportError:
        sys.exit("🛑 ABORTING: google-genai SDK not installed.")
    except Exception as e:
        sys.exit(f"🛑 ABORTING: Failed to initialize Gemini Client: {e}")

# -----------------------------------------------------------------------------
# 5. Native Embedding Class
# -----------------------------------------------------------------------------
try:
    from llama_index.core.base.embeddings.base import BaseEmbedding
    from llama_index.core.bridge.pydantic import PrivateAttr
    from google import genai

    class NativeGeminiEmbedding(BaseEmbedding):
        """
        Direct wrapper for google-genai SDK embeddings to bypass
        dependency issues with official llama-index-embeddings-gemini.
        """
        _client: Any = PrivateAttr()
        _model_name: str = PrivateAttr()

        def __init__(self, api_key: str, model_name: str = "models/text-embedding-004", **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._model_name = model_name
            self._client = genai.Client(api_key=api_key)

        @classmethod
        def class_name(cls) -> str:
            return "NativeGeminiEmbedding"

        def _get_query_embedding(self, query: str) -> List[float]:
            return self._embed(query)

        async def _aget_query_embedding(self, query: str) -> List[float]:
            return self._embed(query)

        def _get_text_embedding(self, text: str) -> List[float]:
            return self._embed(text)

        async def _aget_text_embedding(self, text: str) -> List[float]:
            return self._embed(text)

        def _embed(self, text: str) -> List[float]:
            try:
                response = self._client.models.embed_content(model=self._model_name, contents=text)
                return response.embeddings[0].values
            except Exception as e:
                raise ValueError(f"Embedding failed: {e}")

except ImportError:
    NativeGeminiEmbedding = None

# -----------------------------------------------------------------------------
# 6. LlamaIndex Configuration
# -----------------------------------------------------------------------------
def configure_llama_index(config, models, keys) -> bool:
    if not getattr(config, 'LLAMA_INDEX_AVAILABLE', False):
        print("    ℹ️  LlamaIndex not available in environment.")
        return False

    print("🔧 Configuring LlamaIndex (Native Interface)...")

    try:
        from llama_index.core import Settings
        from llama_index.llms.gemini import Gemini

        if models and models.FLASH_POOL:
            default_model = models.FLASH_POOL[0]
            # Safety wrapper for LlamaIndex which strictly expects the "models/" prefix
            if not default_model.startswith("models/"):
                default_model = f"models/{default_model}"
        else:
            default_model = "models/gemini-2.5-flash"

        llm = Gemini(model_name=default_model, api_key=keys.get("GEMINI", ""), temperature=0.1)
        Settings.llm = llm

        active_embed = None

        if NativeGeminiEmbedding:
            embedding_candidates = [
                "models/text-embedding-004",
                "models/gemini-embedding-001",
                "models/embedding-001"
            ]

            print("    🔍 Initializing Native Embedding Model (Cascade)...")

            for model_name in embedding_candidates:
                try:
                    candidate = NativeGeminiEmbedding(api_key=keys.get("GEMINI", ""), model_name=model_name)

                    # Verify it actually works by attempting to embed a short string
                    vec = candidate.get_text_embedding("Startup Verification")

                    if vec and len(vec) > 0:
                        active_embed = candidate
                        print(f"      ✅ Selected: {model_name} (via google.genai)")
                        break
                except Exception:
                    pass

        if active_embed:
            Settings.embed_model = active_embed
            print("    ✅ LlamaIndex Configured Successfully.")
            return True
        else:
            print("    ❌ CRITICAL: All Gemini embedding models failed to connect.")
            return False

    except ImportError as e:
        print(f"\n🛑 CRITICAL ERROR: LlamaIndex Dependency Missing: {e}")
        return False
    except Exception as e:
        print(f"    ❌ LlamaIndex Global Config Error: {e}")
        return False