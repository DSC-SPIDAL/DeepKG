import os
import glob
import pandas as pd
import re
from tabulate import tabulate
import warnings
from datetime import datetime

warnings.simplefilter(action='ignore')

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
    return None

def main():
    print("\n" + "="*140)
    print(" 🏆 DEEPCOLLECTOR: FINAL OMNI-PARSER EVALUATION MATRIX")
    print("="*140)
    
    # 1. Parse all log files to map timestamps to context tiers
    log_files = glob.glob("**/*.log", recursive=True) + glob.glob("**/*ConsoleLog.txt", recursive=True)
    logs_data = []
    
    for f in log_files:
        if 'master' in f.lower() or 'agent_log' in f.lower(): continue
        model = get_model_name(os.path.basename(f))
        if model == 'Unknown': continue
        
        ctx = get_context_from_filename(f)
        if 'gemini' in model.lower(): ctx = 'Cloud'
        if not ctx: ctx = '64K' # Default if missing
        
        active_proj = "UNKNOWN"
        time_taken = 0.0
        
        try:
            with open(f, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                
                # Double check context from file content if not in filename
                if ctx == '64K' and 'gemini' not in model.lower():
                    if 'Context: 32768' in content: ctx = '32K'
                    elif 'Context: 131072' in content: ctx = '131K'

                # Extract Project and Time
                for line in content.split('\n'):
                    proj_match = re.search(r'(?:Project:|PROJECT:|STARTING PROJECT:)\s*([A-Za-z0-9_]+)', line)
                    if proj_match: active_proj = proj_match.group(1).upper()
                    
                    time_match = re.search(r'Workflow Wall-Clock Time.*?([\d\.]+)s?', line)
                    if time_match and active_proj != "UNKNOWN":
                        time_taken = float(time_match.group(1))
                        logs_data.append({
                            'Project': active_proj,
                            'Model': model,
                            'Context': ctx,
                            'Time': time_taken,
                            'mtime': os.path.getmtime(f)
                        })
                        active_proj = "UNKNOWN"
        except: continue

    # 2. Parse all CSVs and match them to the discovered logs
    csv_files = glob.glob("**/*.csv", recursive=True)
    csv_files = [f for f in csv_files if 'vram_' not in os.path.basename(f).lower() and 'checkpoint_' not in os.path.basename(f).lower()]
    
    yields = []
    missing_markers = ["[missing]", "[skipped]", "none_found", "unknown", "nan", "none", ""]
    
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if df.empty: continue
            
            if 'Project' in df.columns: proj = str(df['Project'].iloc[0]).upper().strip()
            else:
                parts = os.path.basename(f).split('_')
                if len(parts) > 2 and parts[0].upper() == 'BENCH': proj = parts[1].upper()
                else: proj = parts[0].upper()
            if proj == 'BENCH': proj = parts[2].upper()
            
            bmodel = str(df['Benchmark_Model'].iloc[0]) if 'Benchmark_Model' in df.columns else os.path.basename(f)
            model = get_model_name(bmodel)
            if model == 'Unknown': model = get_model_name(f)
            if model == 'Unknown': continue
            
            ctx = '64K'
            if 'gemini' in model.lower():
                ctx = 'Cloud'
            else:
                csv_mtime = os.path.getmtime(f)
                ctx_fallback = get_context_from_filename(f)
                if ctx_fallback: ctx = ctx_fallback

                matching_logs = [l for l in logs_data if l['Project'] == proj and l['Model'] == model]
                if matching_logs:
                    closest_log = min(matching_logs, key=lambda x: abs(x['mtime'] - csv_mtime))
                    ctx = closest_log['Context']
                    
            config_name = f"{model} [{ctx}]" if ctx != 'Cloud' else model
            
            meta_cols = ['model', 'project', 'timestamp', 'job_id', 'runtime', 'elapsed_seconds', 'context', 'context_length', 'benchmark_model']
            science_cols = [c for c in df.columns if c.lower() not in meta_cols and not str(c).endswith('(C)') and not str(c).endswith('(Telemetry)')]
            
            def is_valid(x): return str(x).strip().lower() not in missing_markers and pd.notna(x)
            if hasattr(df[science_cols], 'map'): populated_cells = df[science_cols].map(is_valid).sum().sum()
            else: populated_cells = df[science_cols].applymap(is_valid).sum().sum()
                
            yields.append({'Project': proj, 'Configuration': config_name, 'Score': int(populated_cells)})
        except Exception: pass

    df_yield = pd.DataFrame(yields)
    df_time = pd.DataFrame(logs_data)
    
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
    else: print("No yield data found.")

    print("\n 2. EXECUTION TIME MATRIX (SECONDS)")
    print("-" * 140)
    if not df_time.empty:
        df_time['Configuration'] = df_time.apply(lambda x: x['Model'] if x['Context'] == 'Cloud' else f"{x['Model']} [{x['Context']}]", axis=1)
        agg_time = df_time.groupby(['Project', 'Configuration'])['Time'].mean().reset_index()
        pivot_time = agg_time.pivot(index='Project', columns='Configuration', values='Time')
        pivot_time = pivot_time.reindex(columns=[c for c in ordered_cols if c in pivot_time.columns])
        print(tabulate(pivot_time.fillna(0).round(1), headers="keys", tablefmt="pipe"))
    else: print("No timing data found.")
    
    print("="*140 + "\n")

if __name__ == "__main__":
    main()
