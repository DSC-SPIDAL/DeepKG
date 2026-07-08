import os
import glob
import pandas as pd
import re
from tabulate import tabulate
import warnings

warnings.simplefilter(action='ignore')

def get_model_and_context(chunk_text, pre_text, fname):
    model = "Unknown"
    ctx = "64K"

    # 1. Detect Model from Log Text
    rag_match = re.search(r'RAG Extraction:\s*([A-Za-z0-9\-\.]+)', chunk_text)
    if not rag_match: rag_match = re.search(r'RAG Extraction:\s*([A-Za-z0-9\-\.]+)', pre_text)
    
    if rag_match:
        raw_m = rag_match.group(1).lower()
        if 'flash' in raw_m: model = 'Gemini-3.5-Flash (Cloud)'
        elif 'pro' in raw_m: model = 'Gemini-3.1-Pro (Cloud)'
        elif 'qwen' in raw_m: model = 'Qwen2.5-32B'
        elif 'gemma' in raw_m: model = 'Gemma-4-31B'
        elif 'deepseek' in raw_m: model = 'DeepSeek-R1'
    
    # Fallback to filename
    if model == "Unknown":
        lower_f = fname.lower()
        if 'pro' in lower_f or 'monolithic' in lower_f: model = 'Gemini-3.1-Pro (Cloud)'
        elif 'flash' in lower_f or 'cascade' in lower_f: model = 'Gemini-3.5-Flash (Cloud)'
        elif 'qwen' in lower_f: model = 'Qwen2.5-32B'
        elif 'gemma' in lower_f: model = 'Gemma-4-31B'
        elif 'deepseek' in lower_f: model = 'DeepSeek-R1'

    # 2. Detect Context
    if 'Gemini' in model:
        ctx = 'Cloud'
    else:
        ctx_match = re.search(r'Context.*?(\d+)', chunk_text, re.IGNORECASE)
        if not ctx_match: ctx_match = re.search(r'Context.*?(\d+)', pre_text, re.IGNORECASE)
        
        if ctx_match:
            val = int(ctx_match.group(1))
            if val >= 128000: ctx = '131K'
            elif val >= 64000: ctx = '64K'
            elif val >= 32000: ctx = '32K'
        else:
            lower_f = fname.lower()
            if '131072' in lower_f or '131k' in lower_f or 'titan' in lower_f: ctx = '131K'
            elif '65536' in lower_f or '64k' in lower_f: ctx = '64K'
            elif '32768' in lower_f or '32k' in lower_f or 'ablation' in lower_f: ctx = '32K'

    config = f"{model} [{ctx}]" if ctx != 'Cloud' else model
    return config

def main():
    print("\n" + "="*140)
    print(" 🏆 DEEPCOLLECTOR: TRUE OMNI-PARSER EVALUATION MATRIX")
    print("="*140)
    
    log_files = glob.glob("**/*.log", recursive=True) + glob.glob("**/*ConsoleLog.txt", recursive=True)
    time_data = []
    yield_data = []
    
    for f in log_files:
        if 'master' in f.lower() or 'agent_log' in f.lower(): continue
        
        try:
            with open(f, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                
            # Handle Multi-Project files by splitting at the start token
            chunks = re.split(r'(?:▶️ STARTING:|STARTING PROJECT:)\s*([A-Za-z0-9_]+)', content)
            pre_text = chunks[0]
            
            for i in range(1, len(chunks), 2):
                proj = chunks[i].upper().strip()
                if proj == 'AEON': continue # Deliberate Omission
                
                chunk_text = chunks[i+1]
                config = get_config_name = get_model_and_context(chunk_text, pre_text, os.path.basename(f))
                if 'Unknown' in config: continue

                # Extract Time
                time_match = re.search(r'Workflow Wall-Clock Time.*?([\d\.]+)s?', chunk_text)
                if time_match:
                    time_data.append({'Project': proj, 'Configuration': config, 'Time': float(time_match.group(1))})

                # Extract Fallback Yield from Logs (Amnesia allows us to do this!)
                yield_match = re.search(r'Updated (\d+) fields', chunk_text)
                if not yield_match: yield_match = re.search(r'Updates applied: (\d+)', chunk_text)
                if yield_match:
                    yield_data.append({'Project': proj, 'Configuration': config, 'Score': int(yield_match.group(1))})
        except: continue

    # Parse CSVs for Primary Yield
    csv_files = glob.glob("**/*.csv", recursive=True)
    missing_markers = ["[missing]", "[skipped]", "none_found", "unknown", "nan", "none", ""]
    
    for f in csv_files:
        if 'vram_' in os.path.basename(f).lower() or 'checkpoint_' in os.path.basename(f).lower(): continue
        try:
            df = pd.read_csv(f)
            if df.empty: continue
            
            if 'Project' in df.columns: proj = str(df['Project'].iloc[0]).upper().strip()
            else:
                parts = os.path.basename(f).split('_')
                if len(parts) > 2 and parts[0].upper() == 'BENCH': proj = parts[1].upper()
                else: proj = parts[0].upper()
            if proj == 'BENCH': proj = parts[2].upper()
            if proj == 'AEON': continue # Deliberate Omission
            
            bmodel = str(df['Benchmark_Model'].iloc[0]) if 'Benchmark_Model' in df.columns else os.path.basename(f)
            config = get_model_and_context("", "", bmodel)
            if 'Unknown' in config: config = get_model_and_context("", "", os.path.basename(f))
            if 'Unknown' in config: continue
            
            meta_cols = ['model', 'project', 'timestamp', 'job_id', 'runtime', 'elapsed_seconds', 'context', 'context_length', 'benchmark_model', 'dataset name', 'variant name', 'aliases', 'canonical name', 'type', 'license', 'overall confidence', 'job_created', 'date_created', 'project_created', 'job_updated', 'date_updated', 'project_updated', 'completeness (high conf %)', 'domain']
            science_cols = [c for c in df.columns if c.lower() not in meta_cols and not str(c).endswith('(c)') and not str(c).endswith('(telemetry)')]
            
            def is_valid(x): return str(x).strip().lower() not in missing_markers and pd.notna(x)
            if hasattr(df[science_cols], 'map'): populated_cells = df[science_cols].map(is_valid).sum().sum()
            else: populated_cells = df[science_cols].applymap(is_valid).sum().sum()
                
            yield_data.append({'Project': proj, 'Configuration': config, 'Score': int(populated_cells)})
        except Exception: pass

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
