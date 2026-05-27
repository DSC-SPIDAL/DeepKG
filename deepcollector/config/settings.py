# =============================================================================
# V92: Settings (Dual Folder ID Configuration - Fully Expanded & Uncompressed)
# =============================================================================
import os
import re
import importlib
import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

# Force reload of schema to pick up changes immediately
import deepcollector.config.schema as schema_module
importlib.reload(schema_module)

from deepcollector.config.schema import (
    KB_SCHEMA,
    CATALOG_SCHEMA,
    GROUNDING_FIELDS,
    EXTRACTED_FIELDS,
    ASSIGNMENT_FIELDS,
    DDI_INSPECTABLE_FIELDS,
    MISSING_DATA_PLACEHOLDERS,
    PLAUSIBILITY_THRESHOLDS,
    KB_SCHEMA_VERSION
)

def check_import(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

PROFILES = {
    "BALANCED_STABILITY": {
        "ARCHITECTURE": "ROBUST",
        "PARALLEL_CONCURRENCY_LIMIT": 4,
        "CELLULAR_RAG_BATCH_SIZE": 50,
        "INDEXING_BATCH_SIZE": 8,
        "INDEXING_THROTTLE_DELAY": 1.5,
        "CELLULAR_RAG_THROTTLE_DELAY": 0.75,
    },
    "LOW_CONCURRENCY_ROBUST": {
        "ARCHITECTURE": "ROBUST",
        "PARALLEL_CONCURRENCY_LIMIT": 2,
        "CELLULAR_RAG_BATCH_SIZE": 30,
        "INDEXING_BATCH_SIZE": 5,
        "INDEXING_THROTTLE_DELAY": 3.0,
        "CELLULAR_RAG_THROTTLE_DELAY": 1.5,
    },
     "HIGH_THROUGHPUT_ROBUST": {
        "ARCHITECTURE": "ROBUST",
        "PARALLEL_CONCURRENCY_LIMIT": 10,
        "CELLULAR_RAG_BATCH_SIZE": 100,
        "INDEXING_BATCH_SIZE": 15,
        "INDEXING_THROTTLE_DELAY": 0.5,
        "CELLULAR_RAG_THROTTLE_DELAY": 0.25,
    },
    "STREAMLINED_BALANCED": {
        "ARCHITECTURE": "STREAMLINED",
        "PARALLEL_CONCURRENCY_LIMIT": 6,
        "CELLULAR_RAG_BATCH_SIZE": 60,
        "INDEXING_BATCH_SIZE": 10,
        "INDEXING_THROTTLE_DELAY": 1.0,
        "CELLULAR_RAG_THROTTLE_DELAY": 0.5,
    },
    "VLLM_HIGH_THROUGHPUT": {
        "ARCHITECTURE": "ROBUST",
        "PARALLEL_CONCURRENCY_LIMIT": 20,
        "CELLULAR_RAG_BATCH_SIZE": 100,
        "INDEXING_BATCH_SIZE": 15,
        "INDEXING_THROTTLE_DELAY": 0.1,
        "CELLULAR_RAG_THROTTLE_DELAY": 0.1,
    }
}

@dataclass
class AppConfig:
    CURRENT_PROJECT_ID: Optional[str] = None
    CURRENT_PROJECT_NAME: Optional[str] = None
    INITIAL_URLS: List[str] = field(default_factory=list)
    PROJECT_CONTEXT: Optional[str] = None
    PROJECT_METHOD: int = 1
    HARVESTER_OVERRIDE_URL: Optional[str] = None
    USER_CATALOG_URL: Optional[str] = None

    VERBOSITY_LEVEL: int = 1
    PROFILE_NAME: str = "BALANCED_STABILITY"
    DDI_STRESS_TEST_MODE: bool = False
    EXPORT_TO_NEW_SHEET: bool = True

    # --- HYBRID BACKEND & EXPORT TOGGLES ---
    LLM_BACKEND: str = os.environ.get("DEEPCOLLECTOR_LLM_BACKEND", "GEMINI")
    OUTPUT_FORMAT: str = os.environ.get("DEEPCOLLECTOR_OUTPUT_FORMAT", "SHEET")
    USE_vLLM: bool = os.environ.get("DEEPCOLLECTOR_USE_VLLM", "False") == "True"

    # --- SEARCH TOOLS ---
    SEARCH_BACKEND: str = os.environ.get("DEEPCOLLECTOR_SEARCH_BACKEND", "GEMINI")
    SEARXNG_URL: str = os.environ.get("DEEPCOLLECTOR_SEARXNG_URL", "http://localhost:8080")
    SEARCH_NUM_RESULTS: int = 7

    # --- CONFIGURABLE DRIVE LOCATIONS ---
    # 1. Folder for Spreadsheet/Catalog CSV Exports
    GOOGLE_DRIVE_SHEET_FOLDER_ID: str = "15nvgAVcvoigcQUyla9T9A66KDdj7ZPoK"

    # 2. Folder for System/Console Logs
    GOOGLE_DRIVE_LOG_FOLDER_ID: str = "1mw42keL9BNmaNgrW_ssDFGxe8_lcXBQ2"

    # 3. Knowledge Injection Registry Doc
    KNOWLEDGE_MASTER_DOC_URL: str = "https://docs.google.com/document/d/16oN5NyOC2lFBQvLgept4y9yiTUbgKqiz-mLRXnokefw/export?format=html"

    # 4. Deep Research Model Target
    DEEP_RESEARCH_AGENT_MODEL: str = "deep-research-pro-preview-12-2025"

    # --- RAG TUNING PARAMETERS (Context Window Management) ---
    RAG_DISCOVERY_TOP_K: int = 15
    RAG_DISCOVERY_MAX_CHARS: int = 45000
    RAG_CELLULAR_TOP_K: int = 10
    RAG_CELLULAR_MAX_CHARS: int = 35000
    RAG_CELLULAR_FALLBACK_CHARS: int = 15000

    # --- Deep Research Settings ---
    ENABLE_DEEP_RESEARCH: bool = True
    DEEP_RESEARCH_TIMEOUT_MINUTES: int = 45
    DEEP_RESEARCH_MAX_RETRIES: int = 6
    ABORT_ON_DEEP_RESEARCH_FAILURE: bool = True
    FORCE_ASYNC_POLLING: bool = False
    DEEP_RESEARCH_POLLING_INTERVAL_SECONDS: int = 15
    ENABLE_LOCAL_DEEP_RESEARCH: bool = True

    # --- Advanced RAG Engine Features ---
    ENABLE_PREFLIGHT_CRAWLER: bool = True
    ENABLE_ARBITRATION_PROMPT: bool = True
    ENABLE_STRICT_TAXONOMY: bool = True
    ENABLE_VARIANT_MAPPING: bool = True
    ENABLE_MULTI_QUERY_RAG: bool = True
    ENABLE_GOLDEN_FASTPATH: bool = True

    # --- MISSING KWARGS RESTORED HERE ---
    ENABLE_SINGLETON_VERIFICATION: bool = True
    ENABLE_ORACLE_SEARCH: bool = True
    SCRAPING_METHOD: int = 1
    WIPE_CURRENT_PROJECT_ONLY: bool = False
    _CUDA_OOM_ABORT: bool = False
    LLM_ARBITRATION_LIMIT: int = 2500

    MIN_ASSIGNMENT_CONFIDENCE_GATE: float = 0.40
    CONFIDENCE_THRESHOLD: float = 0.80
    CONFIDENCE_LOCK_THRESHOLD: float = 0.95
    GROUNDING_CONFIDENCE_THRESHOLD: float = 0.90

    # --- Iteration limits (Now configurable via __init__) ---
    MAX_DISCOVERY_ITERATIONS: int = 2
    MAX_GROUNDING_ITERATIONS: int = 3
    MAX_EXTRACTION_ITERATIONS: int = 8

    GOOGLE_SHEET_KB_INPUT: Optional[str] = None
    GOOGLE_SHEET_HINTS_INPUT: Optional[str] = None
    GOOGLE_SHEET_PROJECT_LIST_INPUT: Optional[str] = None
    KNOWLEDGE_INJECTION_DATA: List[Dict[str, str]] = field(default_factory=list)
    SECRETS: Dict[str, Optional[str]] = field(default_factory=dict)

    DATA_INSPECTION_ENABLED: bool = True
    MAX_DOWNLOAD_PREVIEW_BYTES: int = 1024 * 1024 * 1
    MAX_DOWNLOAD_ARCHIVE_BYTES: int = 1024 * 1024 * 50
    INCLUDE_RAG_TELEMETRY_IN_REPORT: bool = False

    # --- Runtime Flags ---
    IN_COLAB: bool = False
    GSPREAD_AVAILABLE: bool = False
    LLAMA_INDEX_AVAILABLE: bool = False
    PdfReader: Optional[Any] = None
    BM25Retriever: Optional[Any] = None
    UCIMLREPO_AVAILABLE: bool = False

    EXECUTION_ARCHITECTURE: str = field(init=False, default="UNKNOWN")
    GOOGLE_SHEET_KB_ID: Optional[str] = field(init=False, default=None)
    GOOGLE_SHEET_HINTS_ID: Optional[str] = field(init=False, default=None)
    GOOGLE_SHEET_PROJECT_LIST_ID: Optional[str] = field(init=False, default=None)

    PARALLEL_CONCURRENCY_LIMIT: int = field(init=False, default=4)
    CELLULAR_RAG_BATCH_SIZE: int = field(init=False, default=50)
    INDEXING_BATCH_SIZE: int = field(init=False, default=8)
    INDEXING_THROTTLE_DELAY: float = field(init=False, default=1.5)
    CELLULAR_RAG_THROTTLE_DELAY: float = field(init=False, default=0.75)

    @property
    def KB_SCHEMA(self):
        return KB_SCHEMA

    @property
    def KB_SCHEMA_VERSION(self):
        return KB_SCHEMA_VERSION

    @property
    def CATALOG_SCHEMA(self):
        return CATALOG_SCHEMA

    @property
    def GROUNDING_FIELDS(self):
        return GROUNDING_FIELDS

    @property
    def EXTRACTED_FIELDS(self):
        return EXTRACTED_FIELDS

    @property
    def ASSIGNMENT_FIELDS(self):
        return ASSIGNMENT_FIELDS

    @property
    def DDI_INSPECTABLE_FIELDS(self):
        return DDI_INSPECTABLE_FIELDS

    @property
    def MISSING_DATA_PLACEHOLDERS(self):
        return MISSING_DATA_PLACEHOLDERS

    @property
    def PLAUSIBILITY_THRESHOLDS(self):
        return PLAUSIBILITY_THRESHOLDS

    def __post_init__(self):
        if self.CURRENT_PROJECT_ID:
            self.CURRENT_PROJECT_ID = self.CURRENT_PROJECT_ID.upper()

        self._apply_profile()
        self._calculate_iterations()
        self._process_sheet_ids()

        self.IN_COLAB = check_import("google.colab")
        self.GSPREAD_AVAILABLE = check_import("gspread")
        self.LLAMA_INDEX_AVAILABLE = check_import("llama_index.core")
        self.UCIMLREPO_AVAILABLE = check_import("ucimlrepo")

        try:
            from pypdf import PdfReader
            self.PdfReader = PdfReader
        except ImportError:
            self.PdfReader = None

        try:
            from llama_index.retrievers.bm25 import BM25Retriever
            self.BM25Retriever = BM25Retriever
        except ImportError:
            self.BM25Retriever = None

    def get_operational_report(self) -> Dict[str, Any]:
        report = {}
        for f in dataclasses.fields(self):
            k = f.name
            v = getattr(self, k)
            if k == 'SECRETS':
                report[k] = "[REDACTED FOR SECURITY]"
            elif isinstance(v, type):
                # Type Serialization fix to prevent JSON crash when writing log
                report[k] = str(v)
            elif isinstance(v, list) and k in ['KNOWLEDGE_INJECTION_DATA', 'INITIAL_URLS']:
                report[k] = f"[{len(v)} items]"
            else:
                report[k] = v
        return report

    def is_complete(self) -> bool:
        if not self.CURRENT_PROJECT_ID or not self.PROJECT_CONTEXT or not self.CURRENT_PROJECT_NAME:
            return False
        return True

    def validate_configuration(self) -> bool:
        if self.is_complete():
            if self.INITIAL_URLS is None:
                self.INITIAL_URLS = []
            return True

        if self.VERBOSITY_LEVEL >= 0:
             print(f"❌ CRITICAL: Missing essential project configuration.")
        return False

    def recalculate_runtime_parameters(self):
        self._calculate_iterations()
        self._process_harvester_override()

    def _process_harvester_override(self):
        self.HARVESTER_OVERRIDE_URL = None
        initial_urls = self.INITIAL_URLS or []

        if self.PROJECT_METHOD > 1 and initial_urls and len(initial_urls) == 1:
            self.HARVESTER_OVERRIDE_URL = initial_urls[0]

    def _apply_profile(self):
        if self.LLM_BACKEND in ["LOCAL_PRO", "LOCAL_CLASSROOM"]:
            self.PROFILE_NAME = "VLLM_HIGH_THROUGHPUT" if getattr(self, 'USE_vLLM', False) else "LOCAL_GPU_SAFE"
        elif self.PROFILE_NAME not in PROFILES:
             self.PROFILE_NAME = "BALANCED_STABILITY"

        profile = PROFILES.get(self.PROFILE_NAME)
        self.EXECUTION_ARCHITECTURE = profile["ARCHITECTURE"]
        self.PARALLEL_CONCURRENCY_LIMIT = profile["PARALLEL_CONCURRENCY_LIMIT"]
        self.CELLULAR_RAG_BATCH_SIZE = profile["CELLULAR_RAG_BATCH_SIZE"]
        self.INDEXING_BATCH_SIZE = profile["INDEXING_BATCH_SIZE"]
        self.INDEXING_THROTTLE_DELAY = profile["INDEXING_THROTTLE_DELAY"]
        self.CELLULAR_RAG_THROTTLE_DELAY = profile.get("CELLULAR_RAG_THROTTLE_DELAY", 0.5)

    def _calculate_iterations(self):
        if self.EXECUTION_ARCHITECTURE == "STREAMLINED":
            self.MAX_DISCOVERY_ITERATIONS = 1
            self.MAX_GROUNDING_ITERATIONS = 0
            self.MAX_EXTRACTION_ITERATIONS = 6
        else:
            self.MAX_DISCOVERY_ITERATIONS = 2
            self.MAX_GROUNDING_ITERATIONS = 3
            self.MAX_EXTRACTION_ITERATIONS = 8

        if self.PROJECT_METHOD > 1:
             self.MAX_DISCOVERY_ITERATIONS = 0
             if self.EXECUTION_ARCHITECTURE == "STREAMLINED":
                  self.MAX_EXTRACTION_ITERATIONS = 5

    def _process_sheet_ids(self):
        self.GOOGLE_SHEET_KB_ID = self._extract_sheet_id(self.GOOGLE_SHEET_KB_INPUT)
        self.GOOGLE_SHEET_HINTS_ID = self._extract_sheet_id(self.GOOGLE_SHEET_HINTS_INPUT)
        self.GOOGLE_SHEET_PROJECT_LIST_ID = self._extract_sheet_id(self.GOOGLE_SHEET_PROJECT_LIST_INPUT)

    def _extract_sheet_id(self, input_str):
        if not input_str or "YOUR_GOOGLE_SHEET_ID_HERE" in input_str:
            return None

        input_str = str(input_str).strip()
        match = re.search(r'/spreadsheets/d/([a-zA-Z0-9-_]+)', input_str)

        if match:
            return match.group(1)

        if re.match(r'^[a-zA-Z0-9-_]{30,}$', input_str):
            return input_str

        return None

print("✅ deepcollector/config/settings.py written (100% Fully Expanded & Uncompressed).")