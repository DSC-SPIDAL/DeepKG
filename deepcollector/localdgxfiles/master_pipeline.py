import os
import shutil
import pandas as pd
import numpy as np
import io
import re
import warnings

warnings.simplefilter(action='ignore')

print("🚀 INITIATING DEEPCOLLECTOR MASTER PIPELINE...\n")

ROOT_DIR = os.path.expanduser("~/Desktop/DeepKG")
CLEAN_DIR = os.path.join(ROOT_DIR, "Cleaned_Paper_Data")
TRASH_DIR = os.path.join(ROOT_DIR, "Trash_Bin")

RUN1_DIR = os.path.join(CLEAN_DIR, "run1")
RUN2_DIR = os.path.join(CLEAN_DIR, "run2")
COLAB_DIR = os.path.join(CLEAN_DIR, "Colab")
VRAM_DIR = os.path.join(CLEAN_DIR, "VRAM_Logs")

for d in [RUN1_DIR, RUN2_DIR, COLAB_DIR, VRAM_DIR, TRASH_DIR]:
    os.makedirs(d, exist_ok=True)

# =====================================================================
# PHASE 1: ABSOLUTE FILE SORT & VRAM RESCUE
# =====================================================================
print("🧹 PHASE 1: Sorting Files & Rescuing VRAM...")

SAFE_EXTS = ('.py', '.sh', '.json', '.ipynb', '.md', '.zip')
SAFE_FILES = ['goldenkb_platinum.csv', 'oracle_full_dataset_audit.csv', 'makefile', 'deepcollector_context.txt']
TRASH_WORDS = ['checkpoint', 'ablation', 'master', 'leaderboard', 'actionable', 'plot.txt', 'gatherout.txt']
PROJECTS = ['utsd', 'lotsa', 'timebench', 'tempo', 'tsfm', 'kagglets', 'kag', 'm2', 'm6']

def move_file(src, dest_folder):
    dest_path = os.path.join(dest_folder, os.path.basename(src))
    try:
        shutil.move(src, dest_path)
    except shutil.Error:
        os.replace(src, dest_path)

for dirpath, _, files in os.walk(ROOT_DIR):
    # Prevent traversal into active clean directories or codebase
    if any(skip in dirpath for skip in ["Cleaned_Paper_Data/run1", "Cleaned_Paper_Data/run2", "Cleaned_Paper_Data/Colab", "Cleaned_Paper_Data/VRAM_Logs", "Trash_Bin", "deepcollector"]):
        continue
        
    for file in files:
        f_low = file.lower()
        filepath = os.path.join(dirpath, file)
        
        if f_low.endswith(SAFE_EXTS) or f_low in SAFE_FILES or 'goldenkb' in f_low: continue
            
        if 'vram' in f_low and f_low.endswith('.csv'):
            move_file(filepath, VRAM_DIR)
            continue
            
        if any(w in f_low for w in TRASH_WORDS) or not f_low.endswith(('.csv', '.log', '.txt')):
            move_file(filepath, TRASH_DIR)
            continue
            
        if 'cloud' in f_low or 'colab' in f_low or '20260702' in f_low:
            move_file(filepath, COLAB_DIR)
            continue
            
        if 'run2' in f_low or '20260707' in f_low or '20260708' in f_low:
            move_file(filepath, RUN2_DIR)
            continue
            
        is_run1 = False
        for p in PROJECTS:
            if p in ['m2', 'm6']:
                if f"_{p}_" in f_low or f"bench_{p}_" in f_low or f_low.startswith(f"{p}_"): is_run1 = True
            elif p in f_low: is_run1 = True
                
        if is_run1: move_file(filepath, RUN1_DIR)
        else: move_file(filepath, TRASH_DIR)

print("  ✅ Workspace fully sanitized and partitioned.\n")

# =====================================================================
# PHASE 2: OMNI-PARSER
# =====================================================================
print("🔍 PHASE 2: Parsing CSVs and Markdown Logs...")

def get_model_name(filename):
    text = filename.lower()
    if 'pro' in text or 'monolithic' in text: return 'Gemini-3.1-Pro (Cloud)'
    if 'flash' in text or 'cloud' in text: return 'Gemini-3.5-Flash (Cloud)'
    if 'qwen' in text and 'deepseek' not in text: return 'Qwen2.5-32B'
    if 'gemma' in text: return 'Gemma-4-31B'
    if 'deepseek' in text: return 'DeepSeek-R1'
    return 'Unknown'

def get_context(filename):
    text = filename.lower()
    if '131072' in text or '128k' in text or '131k' in text: return '131K'
    if '65536' in text or '64k' in text: return '64K'
    if '32768' in text or '32k' in text: return '32K'
    return '64K'

def clean_str(val):
    if pd.isna(val): return "not specified"
    s = str(val).strip()
    return "not specified" if s.lower() in ['nan', '', 'none', '[missing]', '[skipped]'] else s

all_records = []

for target_folder in ['run1', 'run2', 'Colab']:
    folder_path = os.path.join(CLEAN_DIR, target_folder)
    if not os.path.exists(folder_path): continue
    
    for file in os.listdir(folder_path):
        f_lower = file.lower()
        filepath = os.path.join(folder_path, file)
        
        base_model = get_model_name(file)
        if base_model == 'Unknown': continue
        
        ctx = 'Cloud' if 'gemini' in base_model.lower() else get_context(file)
        model_config = f"{base_model} [{ctx}]" if ctx != 'Cloud' else base_model
        
        # Parse CSVs
        if f_lower.endswith('.csv'):
            try:
                df = pd.read_csv(filepath, low_memory=False)
                ds_col = next((c for c in df.columns if 'dataset' in c.lower()), None)
                val_col = next((c for c in df.columns if 'value' in c.lower() or 'extracted' in c.lower()), None)
                
                if ds_col and val_col:
                    meta = ['model', 'project', 'timestamp', 'job_id', 'runtime', 'elapsed_seconds', 'context', 'benchmark_model']
                    science_cols = [c for c in df.columns if c.lower() not in meta and not str(c).endswith('(c)')]
                    
                    for _, row in df.iterrows():
                        ds = clean_str(row[ds_col])
                        if ds == "not specified": continue
                        
                        if 'Schema_Column' in df.columns or 'Schema_Column_Out' in df.columns:
                            col_name = row.get('Schema_Column', row.get('Schema_Column_Out', ''))
                            all_records.append({'Run_ID': target_folder, 'Model': model_config, 'Dataset': ds, 'Attribute': col_name, 'Extracted': clean_str(row[val_col])})
                        else:
                            for col in science_cols:
                                all_records.append({'Run_ID': target_folder, 'Model': model_config, 'Dataset': ds, 'Attribute': col, 'Extracted': clean_str(row[col])})
            except: pass

        # Parse Text/Logs (Markdown Tables)
        elif f_lower.endswith('.log') or f_lower.endswith('.txt'):
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    headers = []
                    for line in f:
                        line_stripped = line.strip()
                        if line_stripped.startswith('|') and line_stripped.endswith('|'):
                            parts = [p.strip() for p in line_stripped.strip('|').split('|')]
                            if 'Dataset Name' in parts or 'Dataset' in parts:
                                headers = parts
                            elif headers and len(parts) >= 3 and not parts[0].startswith('---'):
                                ds = parts[0]
                                if ds.lower() in ['nan', 'unknown', '']: continue
                                for i, col_name in enumerate(headers):
                                    if i == 0 or col_name in ['Project', 'Unnamed: 0']: continue
                                    val = parts[i] if i < len(headers) - 1 else " | ".join(parts[i:])
                                    all_records.append({'Run_ID': target_folder, 'Model': model_config, 'Dataset': ds, 'Attribute': col_name, 'Extracted': clean_str(val)})
            except: pass

master_df = pd.DataFrame(all_records).drop_duplicates()
print(f"  ✅ Extracted {len(master_df)} unique facts.\n")

# =====================================================================
# PHASE 3: ADJUDICATOR & KB ALIGNMENT
# =====================================================================
print("⚖️ PHASE 3: Adjudicating against Platinum KB...")

SCHEMA_MAPPING = {
    "time interval between points": "frequency", "number of time points": "num time points",
    "number of locations/series": "num locations/series", "variables per location": "variables per location",
    "total variables": "total variables", "primary url": "primary url",
    "link to data (actual source)": "link to data (actual source)", "other urls": "other url",
    "other url": "other url", "detailed description": "description",
    "primary source repository": "primary creator", "domain": "domain",
    "canonical name": "canonical name", "type": "type"
}

def super_clean_key(text):
    t = str(text).lower().replace('.csv', '').replace('.txt', '').replace(' (telemetry)', '').replace(' (c)', '')
    return re.sub(r'[^a-z0-9]', '', t)

def norm_key(ds, col):
    mapped_col = SCHEMA_MAPPING.get(str(col).lower().strip(), str(col).lower().strip())
    return (super_clean_key(ds), super_clean_key(mapped_col))

kb_df = pd.read_csv(os.path.join(ROOT_DIR, "deepcollector/localdgxfiles/GOLDENKB_Platinum.csv"))
oracle_df = pd.read_csv(os.path.join(ROOT_DIR, "deepcollector/localdgxfiles/Oracle_Full_Dataset_Audit.csv"))

rosetta = {}
for _, row in oracle_df[oracle_df['Oracle_Verdict'].str.contains('Semantic Match', na=False)].iterrows():
    k = norm_key(row.get('Dataset_Name_Out', ''), row.get('Schema_Column_Out', ''))
    rosetta[(k[0], k[1], clean_str(row.get('Extracted_Value', '')).lower())] = True

kb_dict = {}
kb_ds_col = next((c for c in kb_df.columns if 'dataset' in c.lower()), kb_df.columns[0])
for _, row in kb_df.iterrows():
    for col in kb_df.columns:
        if col != kb_ds_col: kb_dict[norm_key(row[kb_ds_col], col)] = clean_str(row[col])

scored_data = []
for _, row in master_df.iterrows():
    model, run, ds, col, extracted = row['Model'], row['Run_ID'], row['Dataset'], row['Attribute'], row['Extracted']
    k = norm_key(ds, col)
    golden_val = kb_dict.get(k, "not specified")
    
    if extracted.lower() == golden_val.lower() or (k[0], k[1], extracted.lower()) in rosetta:
        status = "True Negative" if golden_val == "not specified" else "True Positive"
    elif extracted == "not specified" and golden_val != "not specified":
        status = "False Negative"
    else:
        status = "False Positive"
        
    scored_data.append({'Model': model, 'Run_ID': run, 'Status': status})

print("  ✅ All facts scored.\n")

# =====================================================================
# PHASE 4: LEADERBOARD GENERATION
# =====================================================================
print("📈 PHASE 4: Generating F1 Variance Matrix...")

metrics_df = pd.DataFrame(scored_data)
run_metrics = []

for (model, run_id), group in metrics_df.groupby(['Model', 'Run_ID']):
    tp = len(group[group['Status'] == 'True Positive'])
    fp = len(group[group['Status'] == 'False Positive'])
    fn = len(group[group['Status'] == 'False Negative'])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    run_metrics.append({'Model': model, 'Run_ID': run_id, 'F1': f1, 'Precision': precision, 'Recall': recall})

leaderboard = pd.DataFrame(run_metrics).groupby('Model').agg(
    F1_Mean=('F1', 'mean'), F1_Std=('F1', 'std'), Precision_Mean=('Precision', 'mean'), Runs_Completed=('Run_ID', 'nunique')
).reset_index().fillna(0).sort_values(by='F1_Mean', ascending=False)

leaderboard['F1_Score'] = leaderboard.apply(lambda x: f"{x['F1_Mean']:.4f} ± {x['F1_Std']:.4f}", axis=1)
leaderboard['Precision'] = leaderboard['Precision_Mean'].apply(lambda x: f"{x:.4f}")

out_path = os.path.join(ROOT_DIR, "deepcollector/localdgxfiles/Omni_Leaderboard_Master.csv")
leaderboard.to_csv(out_path, index=False)

print("=====================================================================")
print("🏆 FINAL MODEL PERFORMANCE LEADERBOARD (OMNI PIPELINE)")
print("=====================================================================")
print(leaderboard[['Model', 'F1_Score', 'Precision', 'Runs_Completed']].to_string(index=False))
print("=====================================================================")
print(f"💾 Saved to: {out_path}")
