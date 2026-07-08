#!/bin/bash

# =============================================================================
# DEEPCOLLECTOR: CLOUD PARITY SCRIPT
# Runs DGX using Google Cloud Gemini + Cloud Search (Exactly like Colab)
# =============================================================================

echo "🔄 Checking Git Repository Status..."
if [ -d "/home/geoffrey/Desktop/DeepKG/.git" ]; then
    echo "    ✅ Valid Git repository detected. Pulling latest changes..."
    cd /home/geoffrey/Desktop/DeepKG
    git pull origin main
else
    echo "    ⚠️ Not a git repository. Backing up and cloning fresh..."
    mv /home/geoffrey/Desktop/DeepKG /home/geoffrey/Desktop/DeepKG_backup_$(date +%s)
    cd /home/geoffrey/Desktop
    git clone https://github.com/DSC-SPIDAL/DeepKG.git
    cd DeepKG
fi

# --- 1. SECRETS & CREDENTIALS (SECURE LOAD) ---
export GOOGLE_APPLICATION_CREDENTIALS="/home/geoffrey/Desktop/DeepKG/credentials.json"

if [ -f "/home/geoffrey/Desktop/DeepKG/.env" ]; then
    source /home/geoffrey/Desktop/DeepKG/.env
    echo "    ✅ Secure API Key loaded locally."
else
    echo "    ❌ ERROR: .env file not found! Please create it with your GEMINI_API_KEY."
    exit 1
fi

export KB_SHEET_ID="1-PuWrHO30E4WPM-rOed03n42gfo5AlEtscKqqtjznA0"
export HINTS_SHEET_ID="1mpv0V5dEQOAv1R1H0ggim2c9TrWoq2wxfVKSUHNDVHo"
export PROJECT_LIST_ID="1gJ6oHZj0NzCHNOeFNyJTBTtlmS0b7gBSHF3iOqJrFwE"
export DRIVE_SHEET_FOLDER_ID="1F-zkctvC0R0pjDYGtQyB883LcIddpPA1"
export DRIVE_LOG_FOLDER_ID="1t4i4EaMghcKgssoDcfi-NDwkIZH5hJpS"

# --- 2. CLOUD ENVIRONMENT ENFORCEMENT ---
echo "☁️ Enforcing Cloud Architecture (Bypassing Local vLLM/SearXNG)..."

# These tell run_cloudagent.py to behave EXACTLY like Colab
export DEEPCOLLECTOR_USE_VLLM="False"
export DEEPCOLLECTOR_LLM_BACKEND="GEMINI"
export DEEPCOLLECTOR_SEARCH_BACKEND="GEMINI"
export DEEPCOLLECTOR_SEARXNG_URL=""

# --- 3. PIPELINE EXECUTION ---
echo "🚀 Starting DeepCollector on DGX (Cloud Mode)..."
python3 run_cloudagent.py

echo "✅ Cloud Run complete."