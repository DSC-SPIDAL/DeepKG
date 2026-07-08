import pandas as pd
import numpy as np
import os
import re
import warnings

warnings.simplefilter(action='ignore')

print("🏆 INIT: DeepCollector Academic Adjudicator (Bulletproof Edition)")

ROOT_DIR = os.path.expanduser("~/Desktop/DeepKG")
CLEANED_DIR = os.path.join(ROOT_DIR, "Cleaned_Paper_Data")

# --- 1. BULLETPROOF KEY MATCHING ---
# Translates long-form prompt questions to Golden KB headers
SCHEMA_MAPPING = {
    "time interval between points": "frequency",
    "number of time points": "num time points",
    "number of locations/series": "num locations/series",
    "variables per location": "variables per location",
    "total variables": "total variables",
    "primary url": "primary url",
    "link to data (actual source)": "link to data (actual source)",
    "other urls": "other url",
    "other url": "other url",
    "detailed description": "description",
    "primary source repository": "primary creator",
    "domain": "domain",
    "canonical name": "canonical name",
    "type": "type"
}

def clean_str(val):
    if pd.isna(val): return "not specified"
    s = str(val).strip()
    return "not specified" if s.lower() in ['nan', '', 'none', '[missing]', '[skipped]'] else s

def super_clean_key(text):
    """Strips file extensions, spaces, dashes, and underscores for absolute matching."""
    t = str(text).lower().replace('.csv', '').replace('.txt', '').replace(' (telemetry)', '').replace(' (c)', '')
    return re.sub(r'[^a-z0-9]', '', t)

def norm_key(ds, col):
    mapped_col = SCHEMA_MAPPING.get(str(col).lower().strip(), str(col).lower().strip())
    return (super_clean_key(ds), super_clean_key(mapped_col))

def get_model_name(filepath):
    text = filepath.lower()
    if 'pro' in text or 'monolithic' in text: return 'Gemini-3.1-Pro (Cloud)'
    if 'flash' in text or 'cloud' in text: return 'Gemini-3.5-Flash (Cloud)'
    if 'qwen' in text and 'deepseek' not in text: return 'Qwen2.5-32B'
    if 'gemma' in text: return 'Gemma-4-31B'
    if 'deepseek' in text: return 'DeepSeek-R1'
    return 'Unknown'

def get_context(filepath):
    text = filepath.lower()
    if '131072' in text or '128k' in text or '131k' in text: return '131K'
    if '65536' in text or '64k' in text: return '64K'
    if '32768' in text or '32k' in text: return '32K'
    return '64K'

# --- 2. HARVEST DATA ---
print(f"\n🔍 Sweeping {CLEANED_DIR} for CSVs and Logs...")
all_records = []
exclude = ['goldenkb', 'oracle', 'master', 'leaderboard', 'checkpoint']

for target_folder in ['run1', 'run2', 'Colab']:
    folder_path = os.path.join(CLEANED_DIR, target_folder)
    if not os.path.exists(folder_path): continue
    
    for file in os.listdir(folder_path):
        f_lower = file.lower()
        if any(x in f_lower for x in exclude): continue
        
        filepath = os.path.join(folder_path, file)
        base_model = get_model_name(file)
        if base_model == 'Unknown': continue
        
        ctx = 'Cloud' if 'gemini' in base_model.lower() else get_context(file)
        model_config = f"{base_model} [{ctx}]" if ctx != 'Cloud' else base_model
        
        # A. Parse CSV files
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
                        
                        # Row-oriented
                        if 'Schema_Column' in df.columns or 'Schema_Column_Out' in df.columns:
                            col_name = row.get('Schema_Column', row.get('Schema_Column_Out', ''))
                            all_records.append({'Run_ID': target_folder, 'Model': model_config, 'Dataset': ds, 'Attribute': col_name, 'Extracted': clean_str(row[val_col])})
                        # Column-oriented
                        else:
                            for col in science_cols:
                                all_records.append({'Run_ID': target_folder, 'Model': model_config, 'Dataset': ds, 'Attribute': col, 'Extracted': clean_str(row[col])})
            except: pass

        # B. Parse Log and Txt files (Handles Markdown Tables perfectly)
        elif f_lower.endswith('.log') or f_lower.endswith('.txt'):
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    headers = []
                    for line in f:
                        line_stripped = line.strip()
                        if line_stripped.startswith('|') and line_stripped.endswith('|'):
                            # Strip outer pipes and split by inner pipes
                            parts = [p.strip() for p in line_stripped.strip('|').split('|')]
                            
                            # Identify header row
                            if 'Dataset Name' in parts or 'Dataset' in parts:
                                headers = parts
                            # Identify data rows
                            elif headers and len(parts) >= 3 and not parts[0].startswith('---'):
                                ds = parts[0]
                                if ds.lower() in ['nan', 'unknown', '']: continue
                                
                                for i, col_name in enumerate(headers):
                                    if i == 0 or col_name in ['Project', 'Unnamed: 0']: continue
                                    # Scoop remaining string arrays to bypass internal pipe errors in descriptions
                                    val = parts[i] if i < len(headers) - 1 else " | ".join(parts[i:])
                                    all_records.append({'Run_ID': target_folder, 'Model': model_config, 'Dataset': ds, 'Attribute': col_name, 'Extracted': clean_str(val)})
            except: pass

master_df = pd.DataFrame(all_records).drop_duplicates()
print(f"  ✅ Compiled {len(master_df)} unique facts from CSVs and Logs.")

# --- 3. LOAD KB & ORACLE ---
print("\n🧠 Loading Platinum KB and Oracle Rosetta Stone...")
kb_df = pd.read_csv(os.path.join(ROOT_DIR, "deepcollector/localdgxfiles/GOLDENKB.csv"))
oracle_df = pd.read_csv(os.path.join(ROOT_DIR, "deepcollector/localdgxfiles/Oracle_Full_Dataset_Audit.csv"))

# Index Oracle (Using super_clean_key)
rosetta = {}
for _, row in oracle_df[oracle_df['Oracle_Verdict'].str.contains('Semantic Match', na=False)].iterrows():
    k = norm_key(row.get('Dataset_Name_Out', ''), row.get('Schema_Column_Out', ''))
    rosetta[(k[0], k[1], clean_str(row.get('Extracted_Value', '')).lower())] = True

# Index Platinum KB (Using super_clean_key)
kb_dict = {}
kb_ds_col = next((c for c in kb_df.columns if 'dataset' in c.lower()), kb_df.columns[0])
for _, row in kb_df.iterrows():
    for col in kb_df.columns:
        if col != kb_ds_col:
            kb_dict[norm_key(row[kb_ds_col], col)] = clean_str(row[col])

# --- 4. SCORE EXTRACTIONS ---
print("⚖️ Adjudicating against Platinum KB...")
scored_data = []
for _, row in master_df.iterrows():
    model, run, ds, col, extracted = row['Model'], row['Run_ID'], row['Dataset'], row['Attribute'], row['Extracted']
    
    k = norm_key(ds, col)
    golden_val = kb_dict.get(k, "not specified")
    
    if extracted.lower() == golden_val.lower():
        status = "True Negative" if golden_val == "not specified" else "True Positive"
    elif (k[0], k[1], extracted.lower()) in rosetta:
        status = "True Positive"
    elif extracted == "not specified" and golden_val != "not specified":
        status = "False Negative"
    else:
        status = "False Positive"
        
    scored_data.append({'Model': model, 'Run_ID': run, 'Status': status})

# --- 5. GENERATE VARIANCE MATRIX ---
print("\n📈 Calculating F1 Variance Metrics...")
metrics_df = pd.DataFrame(scored_data)
run_metrics = []

for (model, run_id), group in metrics_df.groupby(['Model', 'Run_ID']):
    tp = len(group[group['Status'] == 'True Positive'])
    fp = len(group[group['Status'] == 'False Positive'])
    fn = len(group[group['Status'] == 'False Negative'])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    run_metrics.append({'Model': model, 'Run_ID': run_id, 'F1': f1, 'Precision': precision})

leaderboard = pd.DataFrame(run_metrics).groupby('Model').agg(
    F1_Mean=('F1', 'mean'), F1_Std=('F1', 'std'), Precision_Mean=('Precision', 'mean'), Runs_Completed=('Run_ID', 'nunique')
).reset_index().fillna(0).sort_values(by='F1_Mean', ascending=False)

leaderboard['F1_Score'] = leaderboard.apply(lambda x: f"{x['F1_Mean']:.4f} ± {x['F1_Std']:.4f}", axis=1)
leaderboard['Precision'] = leaderboard['Precision_Mean'].apply(lambda x: f"{x:.4f}")

print("\n=====================================================================")
print("🏆 FINAL MODEL PERFORMANCE LEADERBOARD (ACADEMIC VARIANCE MATRIX)")
print("=====================================================================")
print(leaderboard[['Model', 'F1_Score', 'Precision', 'Runs_Completed']].to_string(index=False))
