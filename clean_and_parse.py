import os
import glob
import shutil
import re
import pandas as pd
from tabulate import tabulate
import warnings

warnings.simplefilter(action='ignore')

def get_model(text):
    text = text.lower()
    if 'pro' in text or 'monolithic' in text: return 'Gemini-3.1-Pro (Cloud)'
    if 'flash' in text or 'cascade' in text or 'cloud' in text: return 'Gemini-3.5-Flash (Cloud)'
    # DeepSeek MUST be checked before Qwen to prevent string matching collision
    if 'deepseek' in text: return 'DeepSeek-R1' 
    if 'qwen' in text: return 'Qwen2.5-32B'
    if 'gemma' in text: return 'Gemma-4-31B'
    return 'Unknown'

def get_project(text):
    for p in ['UTSD', 'TIMEBENCH', 'LOTSA', 'TEMPO', 'TSFM', 'KAGGLETS', 'M2', 'M6']:
        if p.lower() in text.lower(): return p
    return 'Unknown'

def main():
    print("🧹 1. Reorganizing files into Cleaned_Paper_Data...")
    
    out_dir = "Cleaned_Paper_Data"
    os.makedirs(out_dir, exist_ok=True)
    
    # Pre-read logs to determine context lengths for CSVs without context markers
    log_files = glob.glob("Final_Paper_Data/**/*.log", recursive=True) + glob.glob("Final_Paper_Data/**/*ConsoleLog.txt", recursive=True)
    log_meta = []
    
    for f in log_files:
        if 'master' in f.lower() or 'agent_log' in f.lower(): continue
        model = get_model(os.path.basename(f))
        if model == 'Unknown': continue
        
        proj = get_project(os.path.basename(f))
        ctx = "64K"
        if '131072' in f.lower() or '131k' in f.lower() or 'titan' in f.lower(): ctx = "131K"
        elif '32768' in f.lower() or '32k' in f.lower() or 'ablation' in f.lower(): ctx = "32K"
        
        log_meta.append({
            'file': f, 'model': model, 'project': proj, 'context': ctx, 
            'mtime': os.path.getmtime(f)
        })

    # Process all files
    all_files = glob.glob("Final_Paper_Data/**/*", recursive=True)
    valid_exts = ['.csv', '.log', '.txt']
    
    for f in all_files:
        if not os.path.isfile(f) or not any(f.endswith(ext) for ext in valid_exts): continue
        fname = os.path.basename(f).lower()
        if 'vram_' in fname or 'checkpoint_' in fname or 'master_' in fname or 'agent_log' in fname: continue
        
        model = get_model(fname)
        
        # If model is in CSV but not filename, check inside CSV
        if model == 'Unknown' and f.endswith('.csv'):
            try:
                df = pd.read_csv(f)
                bmodel = str(df['Benchmark_Model'].iloc[0]) if 'Benchmark_Model' in df.columns else ""
                model = get_model(bmodel)
            except: pass
            
        if model == 'Unknown' and 'job_' in fname: model = 'Gemini-3.5-Flash (Cloud)'
        if model == 'Unknown': continue

        # Determine Context
        ctx = "64K"
        if 'Gemini' in model:
            ctx = "Cloud"
        else:
            if '131072' in fname or '131k' in fname or 'titan' in fname: ctx = "131K"
            elif '32768' in fname or '32k' in fname or 'ablation' in fname: ctx = "32K"
            elif f.endswith('.csv'):
                # Resolve missing CSV context using log timestamps
                proj = get_project(fname)
                matching_logs = [l for l in log_meta if l['model'] == model and l['project'] == proj]
                if matching_logs:
                    best_log = min(matching_logs, key=lambda x: abs(x['mtime'] - os.path.getmtime(f)))
                    ctx = best_log['context']
        
        # Determine Target Folder
        folder_name = f"{model.split(' ')[0]}_{ctx}" if ctx != "Cloud" else model.replace(' ', '_')
        target_dir = os.path.join(out_dir, folder_name)
        os.makedirs(target_dir, exist_ok=True)
        
        target_path = os.path.join(target_dir, os.path.basename(f))
        shutil.copy2(f, target_path)

    print("🗑️ 2. Moving old Final_Paper_Data to Trash...")
    os.makedirs("Trash", exist_ok=True)
    shutil.move("Final_Paper_Data", os.path.join("Trash", "Final_Paper_Data_Old"))

    print("\n" + "="*140)
    print(" 🏆 DEEPCOLLECTOR: FINAL MATRIX")
    print("="*140)
    
    yield_data = []
    time_data = []
    
    # Parse the correctly sorted Cleaned_Paper_Data
    for root, _, files in os.walk(out_dir):
        for file in files:
            f = os.path.join(root, file)
            model = get_model(file)
            if model == 'Unknown':
                if f.endswith('.csv'):
                    try:
                        df = pd.read_csv(f)
                        model = get_model(str(df['Benchmark_Model'].iloc[0]))
                    except: pass
            
            ctx = os.path.basename(root).split('_')[-1]
            if ctx not in ['32K', '64K', '131K']: ctx = 'Cloud'
            config = f"{model} [{ctx}]" if ctx != 'Cloud' else model
            
            if f.endswith('.csv'):
                try:
                    df = pd.read_csv(f)
                    if df.empty: continue
                    proj = get_project(file)
                    if proj == 'Unknown' and 'Project' in df.columns: proj = str(df['Project'].iloc[0]).upper()
                    if proj == 'Unknown': continue
                    
                    missing_markers = ["[missing]", "[skipped]", "none_found", "unknown", "nan", "none", ""]
                    meta_cols = ['model', 'project', 'timestamp', 'job_id', 'runtime', 'elapsed_seconds', 'context', 'context_length', 'benchmark_model', 'dataset name', 'variant name', 'aliases', 'canonical name', 'type', 'license', 'overall confidence', 'job_created', 'date_created', 'project_created', 'job_updated', 'date_updated', 'project_updated', 'completeness (high conf %)', 'domain']
                    science_cols = [c for c in df.columns if c.lower() not in meta_cols and not str(c).endswith('(c)') and not str(c).endswith('(telemetry)')]
                    
                    def is_valid(x): return str(x).strip().lower() not in missing_markers and pd.notna(x)
                    if hasattr(df[science_cols], 'map'): score = df[science_cols].map(is_valid).sum().sum()
                    else: score = df[science_cols].applymap(is_valid).sum().sum()
                    
                    yield_data.append({'Project': proj, 'Configuration': config, 'Score': int(score)})
                except: pass
                
            elif f.endswith('.txt') or f.endswith('.log'):
                try:
                    with open(f, 'r', encoding='utf-8', errors='ignore') as logfile:
                        content = logfile.read()
                    
                    chunks = re.split(r'(?:STARTING:|STARTING PROJECT:|Project:)\s*([A-Za-z0-9_]+)', content)
                    if len(chunks) == 1:
                        proj = get_project(file)
                        if proj == 'Unknown': continue
                        ym = re.findall(r'(?:Updated|Updates applied:?)\s*(\d+)\s*fields?', content, re.IGNORECASE)
                        if ym: yield_data.append({'Project': proj, 'Configuration': config, 'Score': int(ym[-1])})
                        tm = re.findall(r'Workflow Wall-Clock Time.*?([\d\.]+)s?', content)
                        if tm: time_data.append({'Project': proj, 'Configuration': config, 'Time': float(tm[-1])})
                    else:
                        for i in range(1, len(chunks), 2):
                            proj = chunks[i].upper().strip()
                            if proj not in ['UTSD', 'TIMEBENCH', 'LOTSA', 'TEMPO', 'TSFM', 'KAGGLETS', 'M2', 'M6']: continue
                            chunk_text = chunks[i+1]
                            ym = re.findall(r'(?:Updated|Updates applied:?)\s*(\d+)\s*fields?', chunk_text, re.IGNORECASE)
                            if ym: yield_data.append({'Project': proj, 'Configuration': config, 'Score': int(ym[-1])})
                            tm = re.findall(r'Workflow Wall-Clock Time.*?([\d\.]+)s?', chunk_text)
                            if tm: time_data.append({'Project': proj, 'Configuration': config, 'Time': float(tm[-1])})
                except: pass

    df_yield = pd.DataFrame(yield_data)
    df_time = pd.DataFrame(time_data)
    
    ordered_cols = ['Gemini-3.5-Flash (Cloud)', 'Gemini-3.1-Pro (Cloud)', 
                    'Gemma-4-31B [32K]', 'Gemma-4-31B [64K]', 'Gemma-4-31B [131K]', 
                    'Qwen2.5-32B [32K]', 'Qwen2.5-32B [64K]', 'Qwen2.5-32B [131K]', 
                    'DeepSeek-R1 [32K]', 'DeepSeek-R1 [64K]', 'DeepSeek-R1 [131K]']

    print(" 1. SCIENTIFIC CELL YIELD MATRIX")
    print("-" * 140)
    if not df_yield.empty:
        agg_yield = df_yield.groupby(['Project', 'Configuration'])['Score'].max().reset_index()
        pivot_yield = agg_yield.pivot(index='Project', columns='Configuration', values='Score')
        pivot_yield = pivot_yield.reindex(columns=[c for c in ordered_cols if c in pivot_yield.columns])
        if hasattr(pivot_yield, 'map'): formatted_y = pivot_yield.map(lambda x: f"{int(x)}" if pd.notna(x) else "-")
        else: formatted_y = pivot_yield.applymap(lambda x: f"{int(x)}" if pd.notna(x) else "-")
        print(formatted_y.to_markdown())

    print("\n 2. EXECUTION TIME MATRIX (SECONDS)")
    print("-" * 140)
    if not df_time.empty:
        agg_time = df_time.groupby(['Project', 'Configuration'])['Time'].mean().reset_index()
        pivot_time = agg_time.pivot(index='Project', columns='Configuration', values='Time')
        pivot_time = pivot_time.reindex(columns=[c for c in ordered_cols if c in pivot_time.columns])
        print(tabulate(pivot_time.fillna(0).round(1), headers="keys", tablefmt="pipe"))
        
    print("="*140 + "\n")

if __name__ == "__main__":
    main()
