import pandas as pd
import numpy as np
import os
import io
import re
import warnings

warnings.simplefilter(action='ignore')

print("🚀 INIT: DeepCollector Omni-Parser & Adjudicator")

ROOT_DIR = os.path.expanduser("~/Desktop/DeepKG")
CLEANED_DIR = os.path.join(ROOT_DIR, "Cleaned_Paper_Data")
if not os.path.exists(CLEANED_DIR):
    CLEANED_DIR = ROOT_DIR # Fallback if not organized yet

# --- UTILS FROM YOUR CODEBASE ---
def get_model_name(text):
    text = text.lower()
    if 'pro' in text or 'monolithic' in text: return 'Gemini-3.1-Pro (Cloud)'
    if 'cloud' in text or 'flash' in text or 'cascade' in text: return 'Gemini-3.5-Flash (Cloud)'
    if 'qwen' in text and 'deepseek' not in text: return 'Qwen2.5-32B'
    if 'gemma' in text: return 'Gemma-4-31B'
    if 'deepseek' in text: return 'DeepSeek-R1'
    return 'Unknown'

def get_context_from_filename(filepath):
    text = filepath.lower()
    if '131072' in text or '128k' in text or '131k' in text or 'titan' in text: return '131K'
    if '65536' in text or '64k' in text: return '64K'
    if '32768' in text or '32k' in text or 'ablation' in text: return '32K'
    return '64K' # Default

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

def norm_key(ds, col):
    return (str(ds).strip().lower(), str(col).strip().lower())

# --- 1. HARVEST ALL FILES ---
print(f"🔍 Sweeping {CLEANED_DIR} for CSVs, LOGs, and TXTs...")
all_records = []

# Exclude files that aren't raw data
exclude_files = ['goldenkb.csv', 'oracle_full_dataset_audit.csv', 'master_all_runs_gathered.csv', 'model_evaluation_leaderboard']

for root_path, dirs, files in os.walk(CLEANED_DIR):
    if "Trash" in root_path: continue
    
    for file in files:
        f_lower = file.lower()
        filepath = os.path.join(root_path, file)
        
        if any(ex in f_lower for ex in exclude_files) or 'checkpoint' in f_lower:
            continue
            
        model_base = get_model_name(file)
        if model_base == 'Unknown': continue
        
        ctx = 'Cloud' if 'gemini' in model_base.lower() else get_context_from_filename(file)
        config_name = f"{model_base} [{ctx}]" if ctx != 'Cloud' else model_base
        
        run_id = "run2" if "run2" in root_path.lower() or "run2" in f_lower else "run1"

        # A. Parse CSVs
        if f_lower.endswith('.csv'):
            try:
                df = pd.read_csv(filepath, low_memory=False)
                ds_col = next((c for c in df.columns if c.lower() in ['dataset_name', 'dataset']), None)
                val_col = next((c for c in df.columns if c.lower() in ['extracted_value', 'value']), None)
                
                # Determine project from df or filename
                proj = str(df['Project'].iloc[0]) if 'Project' in df.columns else file.split('_')[1] if len(file.split('_')) > 1 else 'UNKNOWN'
                
                if ds_col and val_col:
                    meta_cols = ['model', 'project', 'timestamp', 'job_id', 'runtime', 'elapsed_seconds', 'context', 'context_length', 'benchmark_model']
                    science_cols = [c for c in df.columns if c.lower() not in meta_cols and not str(c).endswith('(c)') and not str(c).endswith('(telemetry)')]
                    
                    for _, row in df.iterrows():
                        ds = clean_str(row[ds_col])
                        if ds == "not specified": continue
                        
                        # Row-oriented extractions (if Schema_Column exists)
                        if 'Schema_Column' in df.columns or 'Schema_Column_Out' in df.columns:
                            col_name = row.get('Schema_Column', row.get('Schema_Column_Out', ''))
                            extracted = clean_str(row[val_col])
                            raw_col = clean_str(col_name).lower().replace(' (telemetry)', '').replace(' (c)', '').strip()
                            all_records.append({
                                'Run_ID': run_id, 'Model': config_name, 'Project': proj.upper(),
                                'Dataset': ds, 'Attribute': SCHEMA_MAPPING.get(raw_col, raw_col), 'Extracted': extracted
                            })
                        # Column-oriented extractions (Standard Dataframe)
                        else:
                            for col in science_cols:
                                extracted = clean_str(row[col])
                                raw_col = clean_str(col).lower().replace(' (telemetry)', '').replace(' (c)', '').strip()
                                all_records.append({
                                    'Run_ID': run_id, 'Model': config_name, 'Project': proj.upper(),
                                    'Dataset': ds, 'Attribute': SCHEMA_MAPPING.get(raw_col, raw_col), 'Extracted': extracted
                                })
            except Exception as e:
                pass

        # B. Parse Markdown Tables from Logs/Txts
        elif f_lower.endswith('.log') or f_lower.endswith('.txt'):
            current_project = "UNKNOWN"
            table_buffer = []
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    headers = []
                    for line in f:
                        if "STARTING PROJECT:" in line or "Processing Project:" in line or "Project:" in line:
                            match = re.search(r'(?:Project:|PROJECT:|STARTING PROJECT:|Processing Project:)\s*([A-Za-z0-9_]+)', line)
                            if match: current_project = match.group(1).upper()
                        
                        line_stripped = line.strip()
                        if line_stripped.startswith('|') and line_stripped.endswith('|'):
                            table_buffer.append(line_stripped)
                        else:
                            if len(table_buffer) > 2:
                                clean_lines = [r for r in table_buffer if not re.match(r'^\|[-:\s\|]+\|$', r)]
                                table_str = "\n".join(clean_lines)
                                try:
                                    df_parsed = pd.read_csv(io.StringIO(table_str), sep='|', skipinitialspace=True)
                                    df_parsed = df_parsed.dropna(axis=1, how='all')
                                    df_parsed.columns = df_parsed.columns.str.strip()
                                    
                                    for _, row in df_parsed.iterrows():
                                        ds = str(row.get('Dataset Name', row.get('Dataset', 'Unknown'))).strip()
                                        if ds.lower() in ['nan', 'unknown', '']: continue
                                        
                                        for col in df_parsed.columns:
                                            if col.strip() not in ['Dataset Name', 'Dataset', 'Project', 'Unnamed: 0']:
                                                extracted = clean_str(row[col])
                                                raw_col = clean_str(col).lower().replace(' (telemetry)', '').replace(' (c)', '').strip()
                                                all_records.append({
                                                    'Run_ID': run_id, 'Model': config_name, 'Project': current_project,
                                                    'Dataset': ds, 'Attribute': SCHEMA_MAPPING.get(raw_col, raw_col), 'Extracted': extracted
                                                })
                                except: pass
                            table_buffer = []
            except: pass

master_df = pd.DataFrame(all_records)
master_df.drop_duplicates(inplace=True)
print(f"  ✅ Compiled {len(master_df)} unique extractions from across all CSVs and Logs.")

# --- 2. LOAD KB & ORACLE ---
print("\n🧠 Loading Platinum KB and Oracle Rosetta Stone...")
try:
    kb_df = pd.read_csv("GOLDENKB.csv")
    oracle_df = pd.read_csv("Oracle_Full_Dataset_Audit.csv")
except FileNotFoundError as e:
    print(f"❌ Missing GOLDENKB or Oracle file: {e}")
    exit(1)

# Build Oracle Dictionary
semantic_matches = oracle_df[oracle_df['Oracle_Verdict'].str.contains('Semantic Match', na=False, case=False)]
rosetta = {(norm_key(r.get('Dataset_Name_Out',''), r.get('Schema_Column_Out',''))[0], 
            norm_key(r.get('Dataset_Name_Out',''), r.get('Schema_Column_Out',''))[1], 
            clean_str(r.get('Extracted_Value', '')).lower()): True for _, r in semantic_matches.iterrows()}

# Build KB Dictionary
kb_dict = {}
kb_ds_col = next((c for c in kb_df.columns if 'dataset' in c.lower()), kb_df.columns[0])
for _, row in kb_df.iterrows():
    ds = clean_str(row[kb_ds_col])
    for col in kb_df.columns:
        if col != kb_ds_col:
            kb_dict[norm_key(ds, col)] = clean_str(row[col])

# --- 3. SCORING ---
print("⚖️ Scoring extractions...")
scored_data = []
for _, row in master_df.iterrows():
    model, run, ds, col, extracted = row['Model'], row['Run_ID'], row['Dataset'], row['Attribute'], row['Extracted']
    golden_val = kb_dict.get(norm_key(ds, col), "not specified")
    
    if extracted.lower() == golden_val.lower():
        status = "True Negative" if golden_val == "not specified" else "True Positive"
    elif (norm_key(ds, col)[0], norm_key(ds, col)[1], extracted.lower()) in rosetta:
        status = "True Positive"
    elif extracted == "not specified" and golden_val != "not specified":
        status = "False Negative"
    else:
        status = "False Positive"
        
    scored_data.append({'Model': model, 'Run_ID': run, 'Status': status})

scored_df = pd.DataFrame(scored_data)

# --- 4. CALCULATE VARIANCE METRICS ---
print("\n📈 Calculating F1 Matrix...")
run_metrics = []
for (model, run_id), group in scored_df.groupby(['Model', 'Run_ID']):
    tp = len(group[group['Status'] == 'True Positive'])
    fp = len(group[group['Status'] == 'False Positive'])
    fn = len(group[group['Status'] == 'False Negative'])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    run_metrics.append({'Model': model, 'Run_ID': run_id, 'F1': f1, 'Precision': precision})

metrics_df = pd.DataFrame(run_metrics)

leaderboard = metrics_df.groupby('Model').agg(
    F1_Mean=('F1', 'mean'),
    F1_Std=('F1', 'std'),
    Precision_Mean=('Precision', 'mean'),
    Runs_Completed=('Run_ID', 'nunique')
).reset_index().fillna(0)

leaderboard = leaderboard.sort_values(by='F1_Mean', ascending=False)
leaderboard['F1_Score'] = leaderboard.apply(lambda x: f"{x['F1_Mean']:.4f} ± {x['F1_Std']:.4f}", axis=1)
leaderboard['Precision'] = leaderboard['Precision_Mean'].apply(lambda x: f"{x:.4f}")

leaderboard.to_csv("Omni_Evaluation_Leaderboard_Final.csv", index=False)

print("\n=====================================================================")
print("🏆 FINAL MODEL PERFORMANCE LEADERBOARD (ACADEMIC VARIANCE MATRIX)")
print("=====================================================================")
print(leaderboard[['Model', 'F1_Score', 'Precision', 'Runs_Completed']].to_string(index=False))
