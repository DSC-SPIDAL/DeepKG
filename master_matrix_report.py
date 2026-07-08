import pandas as pd
import glob
import re
import warnings
import os

warnings.simplefilter(action='ignore', category=FutureWarning)

def get_model_shortname(raw_name):
    name = str(raw_name).lower()
    if 'gemini' in name: 
        return 'Gemini-Pro (Cloud)' if 'pro' in name else 'Gemini-Flash (Cloud)'
    if 'qwen' in name and 'deepseek' not in name: return 'Qwen2.5-32B'
    if 'gemma' in name: return 'Gemma-4-31B'
    if 'deepseek' in name: return 'DeepSeek-R1'
    return raw_name

def extract_context(filename, df, model_name):
    if 'gemini' in str(model_name).lower(): return 'Cloud'
    name = filename.upper()
    if '131072' in name or '128K' in name or '131K' in name or 'TITAN' in name or 'RESULTS_131K' in name: return '131K'
    if '65536' in name or '64K' in name: return '64K'
    if '32768' in name or '32K' in name: return '32K'
    col_names = [c.lower() for c in df.columns]
    for col in ['context', 'context_length', 'context_size']:
        if col in col_names:
            actual_col = df.columns[col_names.index(col)]
            val = str(df[actual_col].iloc[0])
            if '131' in val or '128' in val: return '131K'
            if '64' in val: return '64K'
            if '32' in val: return '32K'
    return '32K'

def main():
    print("========================================================================================================================")
    print("🏆 DEEPCOLLECTOR MASTER EVALUATION MATRIX: SCIENTIFIC CELL YIELD")
    print("========================================================================================================================")
    
    csv_files = glob.glob("**/*.csv", recursive=True)
    target_files = [f for f in csv_files if 'Bench_' in os.path.basename(f) and 'vram_' not in f.lower()]
    
    if not target_files:
        print("❌ No benchmark data files found.")
        return

    all_data = []
    missing_markers = ["[missing]", "[skipped]", "none_found", "unknown", "nan", "none", ""]
    
    for file in target_files:
        try:
            df = pd.read_csv(file)
            if df.empty: continue
                
            project_col = next((c for c in df.columns if 'project' in c.lower() or 'proj' in c.lower()), None)
            model_col = next((c for c in df.columns if 'model' in c.lower() or 'llm' in c.lower()), None)
            
            if project_col and model_col:
                meta_cols = ['model', 'project', 'timestamp', 'job_id', 'runtime', 'elapsed_seconds', 'context', 'context_length', 'benchmark_model']
                science_cols = [c for c in df.columns if c.lower() not in meta_cols]
                
                def is_valid(x):
                    return str(x).strip().lower() not in missing_markers and pd.notna(x)

                if hasattr(df[science_cols], 'map'):
                    populated_cells = df[science_cols].map(is_valid).sum().sum()
                else:
                    populated_cells = df[science_cols].applymap(is_valid).sum().sum()
                
                raw_model = str(df[model_col].iloc[0]).strip()
                project_name = str(df[project_col].iloc[0]).upper().strip()
                
                short_model = get_model_shortname(raw_model)
                context_tier = extract_context(file, df, raw_model)
                
                config_name = short_model if context_tier == 'Cloud' else f"{short_model} [{context_tier}]"
                    
                all_data.append({
                    'Project': project_name,
                    'Configuration': config_name,
                    'Score': int(populated_cells)
                })
        except Exception:
            pass 

    master_df = pd.DataFrame(all_data)
    
    if master_df.empty:
        print("❌ Could not calculate valid cell data. Please verify the CSV structures.")
        return
        
    aggregated = master_df.groupby(['Project', 'Configuration'])['Score'].max().reset_index()
    pivot_table = aggregated.pivot(index='Project', columns='Configuration', values='Score')
    
    expected_columns = [
        'Gemini-Pro (Cloud)', 'Gemini-Flash (Cloud)',
        'Gemma-4-31B [32K]', 'Gemma-4-31B [64K]', 'Gemma-4-31B [131K]',
        'Qwen2.5-32B [32K]', 'Qwen2.5-32B [64K]', 'Qwen2.5-32B [131K]',
        'DeepSeek-R1 [32K]', 'DeepSeek-R1 [64K]', 'DeepSeek-R1 [131K]'
    ]
    
    pivot_table = pivot_table.reindex(columns=[c for c in expected_columns if c in pivot_table.columns])
    
    def format_val(x):
        return f"{int(x)}" if pd.notna(x) else "-"
        
    if hasattr(pivot_table, 'map'):
        formatted_table = pivot_table.map(format_val)
    else:
        formatted_table = pivot_table.applymap(format_val)
    
    print(formatted_table.to_markdown())
    print("========================================================================================================================")

if __name__ == "__main__":
    main()
