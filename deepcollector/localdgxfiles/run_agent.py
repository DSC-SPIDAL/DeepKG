#!/usr/bin/env python3
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

# 1. LOAD .ENV (Does not override Bash exports)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

warnings.simplefilter(action='ignore', category=FutureWarning)

# 2. READ SCALING & ENVIRONMENT VARIABLES
BENCHMARK_MODE = os.environ.get("BENCHMARK_MODE", "LOCAL")
ENABLE_DR = os.environ.get("ENABLE_DEEP_RESEARCH_FLAG", "True") == "True"

TARGET_PROVIDER = os.environ.get("TARGET_PROVIDER", "OPENAI")
TARGET_MODEL = os.environ.get("TARGET_MODEL", "gpt-5.6-sol")

# 3. CONFIGURE HARDWARE / CLOUD BACKENDS
if BENCHMARK_MODE == "LOCAL":
    MODEL_ID = os.environ.get("LOCAL_MODEL_ID", "google/gemma-4-31b-it")
    os.environ["DEEPCOLLECTOR_LLM_BACKEND"] = "LOCAL_PRO"
    os.environ["DEEPCOLLECTOR_USE_VLLM"] = "True"
    os.environ["OPENAI_API_BASE"] = os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")
    os.environ["OPENAI_API_KEY"] = "sk-vllm-dummy-key"
else:
    os.environ["DEEPCOLLECTOR_LLM_BACKEND"] = "GEMINI"
    os.environ["DEEPCOLLECTOR_USE_VLLM"] = "False"
    MODEL_ID = TARGET_MODEL

os.environ["DEEPCOLLECTOR_SEARCH_BACKEND"] = "GEMINI"

# 4. AUTHENTICATION & EXECUTION
import gspread
import google.auth
from google.oauth2.credentials import Credentials

if BENCHMARK_MODE == "LOCAL":
    try:
        import openai
    except ImportError:
        sys.exit("❌ Missing 'openai' package. Please run: pip3 install openai")

print("\n🔑 Acquiring Credentials for Job...")
try:
    token_path = os.path.expanduser("~/Desktop/DeepKG/token.json")
    creds = Credentials.from_authorized_user_file(token_path)
    google.auth.default = lambda *args, **kwargs: (creds, "deepcollector-app")
    gc = gspread.authorize(creds)
except Exception as e:
    sys.exit(f"❌ OAuth Failed. Missing token.json?\nError: {e}")

from deepcollector.config.settings import AppConfig
from deepcollector.core.executor import execute_jobs

config = AppConfig(
    VERBOSITY_LEVEL=1, SECRETS={"GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", "")},
    GOOGLE_SHEET_KB_INPUT = os.environ.get("KB_SHEET_ID"),
    GOOGLE_SHEET_HINTS_INPUT = os.environ.get("HINTS_SHEET_ID"),
    GOOGLE_SHEET_PROJECT_LIST_INPUT = os.environ.get("PROJECT_LIST_ID"),
    GOOGLE_DRIVE_SHEET_FOLDER_ID = os.environ.get("DRIVE_SHEET_FOLDER_ID"),
    GOOGLE_DRIVE_LOG_FOLDER_ID = os.environ.get("DRIVE_LOG_FOLDER_ID"),
    ENABLE_DEEP_RESEARCH=ENABLE_DR, ENABLE_GOLDEN_FASTPATH=False,
    ENABLE_PREFLIGHT_CRAWLER=True, ENABLE_ARBITRATION_PROMPT=True, ENABLE_STRICT_TAXONOMY=True,
    ENABLE_MULTI_QUERY_RAG=True, ENABLE_VARIANT_MAPPING=True, ENABLE_SINGLETON_VERIFICATION=True, ENABLE_ORACLE_SEARCH=True
)

if TARGET_PROVIDER in ["ANTHROPIC", "OPENAI", "XAI"]:
    config.CELLULAR_RAG_BATCH_SIZE = 5      
    config.PARALLEL_CONCURRENCY_LIMIT = 5   
    config.CELLULAR_RAG_THROTTLE_DELAY = 1.0  
else: 
    config.CELLULAR_RAG_BATCH_SIZE = 10     

os.environ["DEEPCOLLECTOR_LOG_FOLDER_ID"] = config.GOOGLE_DRIVE_LOG_FOLDER_ID or ""
config.recalculate_runtime_parameters()
config._process_sheet_ids()

try:
    proj_idx = sys.argv.index("--project") + 1
    PROJECT_NAME_ARG = sys.argv[proj_idx]
except (ValueError, IndexError):
    PROJECT_NAME_ARG = os.environ.get("TARGET_PROJECT", "UTSD")
    
PROJECT_NAMES = [PROJECT_NAME_ARG]

print(f"\n🖥️ DEEPCOLLECTOR INITIALIZED...")
print(f"   - Environment: {BENCHMARK_MODE} (Provider: {TARGET_PROVIDER} | Model: {MODEL_ID})")
print(f"   - Target Project: {PROJECT_NAMES[0]}\n")

try: execute_jobs(mode="AGENT", project_names=PROJECT_NAMES, base_config=config, gc_client=gc, dry_run=False)
except Exception as e: print(f"\n❌ Main Execution Failed or Aborted: {e}")
