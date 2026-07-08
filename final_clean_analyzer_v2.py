import os
import glob
import pandas as pd
import numpy as np
import re
from tabulate import tabulate
import warnings

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    PLOTTING_ENABLED = True
except ImportError:
    PLOTTING_ENABLED = False

warnings.simplefilter(action='ignore')

def get_project_name(filename):
    fname = filename.upper()
    for p in ['UTSD', 'TIMEBENCH', 'LOTSA', 'TEMPO', 'TSFM', 'KAGGLETS', 'M2', 'M6']:
        if p in fname: return p
    return 'UNKNOWN'

def get_model_and_config(filename, root):
    lf = filename.lower()
    lr = root.lower()
    
    if 'pro' in lf or 'pro' in lr or 'monolithic' in lf: return 'Gemini-3.1-Pro', 'Gemini-3.1-Pro (Cloud)'
    if 'flash' in lf or 'flash' in lr or 'cascade' in lf: return 'Gemini-3.5-Flash', 'Gemini-3.5-Flash (Cloud)'
    
    if 'deepseek' in lf or 'deepseek' in lr: model = 'DeepSeek-R1'
    elif 'qwen' in lf or 'qwen' in lr: model = 'Qwen2.5-32B'
    elif 'gemma' in lf or 'gemma' in lr: model = 'Gemma-4-31B'
    else: return 'Unknown', 'Unknown'
    
    ctx = '64K'
    if '131k' in lr or '131072' in lf or '131k' in lf or 'titan' in lf: ctx = '131K'
    elif '32k' in lr or '32768' in lf or '32k' in lf or 'ablation' in lf: ctx = '32K'
    
    return model, f"{model} [{ctx}]"

def extract_time(text):
    # Targets both standard log format and table piped formats accurately
    matches = list(re.finditer(r'Workflow Wall-Clock Time.*?\|\s+[\d\.]+\s+\|\s+([\d\.]+)|Workflow Wall-Clock Time:\s*([\d\.]+)s?', text))
    if matches:
        last = matches[-1]
        return float(last.group(1) if last.group(1) else last.group(2))
    return None

def main():
    print("\n" + "="*120)
    print(" 📊 DEEPCOLLECTOR: FINAL ACADEMIC ANALYZER (TIMING CORRECTED)")
    print("="*120)
    
    base_dir = "Cleaned_Paper_Data"
    if not os.path.exists(base_dir):
        print(f"❌ Error: Directory '{base_dir}' not found.")
        return

    yield_data = []
    time_data = []
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            filepath = os.path.join(root, file)
            fname_lower = file.lower()
            proj = get_project_name(file)
            if proj == 'UNKNOWN': continue
            
            model_base, config_name = get_model_and_config(file, root)
            if model_base == 'Unknown': continue
            
            # --- Parse CSVs for Yield ---
            if fname_lower.endswith('.csv'):
                try:
                    df = pd.read_csv(filepath)
                    if df.empty: continue
                    
                    meta_cols = ['model', 'project', 'timestamp', 'job_id', 'runtime', 'elapsed_seconds', 'context', 'context_length', 'benchmark_model', 'dataset name', 'variant name', 'aliases', 'canonical name', 'type', 'license', 'overall confidence', 'job_created', 'date_created', 'project_created', 'job_updated', 'date_updated', 'project_updated', 'completeness (high conf %)', 'domain']
                    science_cols = [c for c in df.columns if str(c).lower() not in meta_cols and not str(c).endswith('(c)') and not str(c).endswith('(telemetry)')]
                    
                    missing_markers = ["[missing]", "[skipped]", "none_found", "unknown", "nan", "none", ""]
                    def is_valid(x): return str(x).strip().lower() not in missing_markers and pd.notna(x)
                    
                    if hasattr(df[science_cols], 'map'): score = df[science_cols].map(is_valid).sum().sum()
                    else: score = df[science_cols].applymap(is_valid).sum().sum()
                    
                    yield_data.append({'Project': proj, 'Configuration': config_name, 'Model_Base': model_base, 'Score': int(score)})
                except: pass

            # --- Parse Logs for Time ---
            elif fname_lower.endswith('.log') or fname_lower.endswith('.txt'):
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                    chunks = re.split(r'(?:STARTING:|STARTING PROJECT:|Project:)\s*([A-Za-z0-9_]+)', content)
                    if len(chunks) == 1:
                        t_val = extract_time(content)
                        if t_val is not None:
                            time_data.append({'Project': proj, 'Configuration': config_name, 'Model_Base': model_base, 'Time': t_val})
                    else:
                        for i in range(1, len(chunks), 2):
                            chunk_proj = chunks[i].upper().strip()
                            if chunk_proj not in ['UTSD', 'TIMEBENCH', 'LOTSA', 'TEMPO', 'TSFM', 'KAGGLETS', 'M2', 'M6']: continue
                            
                            chunk_text = chunks[i+1]
                            t_val = extract_time(chunk_text)
                            if t_val is not None:
                                time_data.append({'Project': chunk_proj, 'Configuration': config_name, 'Model_Base': model_base, 'Time': t_val})
                except: pass

    df_yield = pd.DataFrame(yield_data)
    df_time = pd.DataFrame(time_data)

    ordered_cols = ['Gemini-3.5-Flash (Cloud)', 'Gemini-3.1-Pro (Cloud)', 
                    'Gemma-4-31B [32K]', 'Gemma-4-31B [64K]', 'Gemma-4-31B [131K]', 
                    'Qwen2.5-32B [32K]', 'Qwen2.5-32B [64K]', 'Qwen2.5-32B [131K]', 
                    'DeepSeek-R1 [32K]', 'DeepSeek-R1 [64K]', 'DeepSeek-R1 [131K]']

    print("\n 1. SCIENTIFIC CELL YIELD MATRIX (PEAK)")
    print("-" * 120)
    if not df_yield.empty:
        agg_yield = df_yield.groupby(['Project', 'Configuration'])['Score'].max().reset_index()
        pivot_yield = agg_yield.pivot(index='Project', columns='Configuration', values='Score')
        pivot_yield = pivot_yield.reindex(columns=[c for c in ordered_cols if c in pivot_yield.columns])
        formatted_y = pivot_yield.fillna("-").astype(str).replace(r'\.0$', '', regex=True)
        print(formatted_y.to_markdown())
    else: print("❌ No yield data found.")

    print("\n 2. EXECUTION TIME MATRIX (MEAN SECONDS \u00B1 STD DEV)")
    print("-" * 120)
    if not df_time.empty:
        agg_time_mean = df_time.groupby(['Project', 'Configuration'])['Time'].mean()
        agg_time_std = df_time.groupby(['Project', 'Configuration'])['Time'].std().fillna(0)
        
        combined_time = agg_time_mean.round(1).astype(str) + " \u00B1 " + agg_time_std.round(1).astype(str)
        combined_time.name = 'Time_Str' 
        
        pivot_time = combined_time.reset_index().pivot(index='Project', columns='Configuration', values='Time_Str')
        pivot_time = pivot_time.reindex(columns=[c for c in ordered_cols if c in pivot_time.columns])
        print(tabulate(pivot_time.fillna("-"), headers="keys", tablefmt="pipe"))
    else: print("❌ No timing data found.")
    
    print("="*120 + "\n")

    if PLOTTING_ENABLED and not df_yield.empty and not df_time.empty:
        os.makedirs("Paper_Figures", exist_ok=True)
        
        plt.figure(figsize=(14, 8))
        sns.barplot(data=df_yield, x='Project', y='Score', hue='Model_Base', errorbar='sd', palette='viridis', capsize=0.1)
        plt.title('Scientific Metadata Extraction Yield by Architecture', fontsize=16, pad=15)
        plt.ylabel('Valid Cells Populated', fontsize=14)
        plt.xlabel('Evaluation Dataset', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Model Family')
        plt.tight_layout()
        plt.savefig('Paper_Figures/Figure_1_Cell_Yield.pdf', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(14, 8))
        sns.barplot(data=df_time, x='Project', y='Time', hue='Configuration', errorbar='sd', palette='magma', capsize=0.1)
        plt.title('Execution Wall-Clock Time by Configuration', fontsize=16, pad=15)
        plt.ylabel('Seconds (Log Scale)', fontsize=14)
        plt.xlabel('Evaluation Dataset', fontsize=14)
        plt.yscale('log')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Configuration')
        plt.tight_layout()
        plt.savefig('Paper_Figures/Figure_2_Execution_Time.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Paper figures successfully generated in ~/Desktop/DeepKG/Paper_Figures/")

if __name__ == "__main__":
    main()
