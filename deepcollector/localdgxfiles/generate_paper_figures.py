import os
import glob
import pandas as pd
import numpy as np
import re
from tabulate import tabulate
import warnings

# Try to import matplotlib and seaborn for charts (will skip gracefully if not installed)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    PLOTTING_ENABLED = True
except ImportError:
    PLOTTING_ENABLED = False

warnings.simplefilter(action='ignore')

def get_config_name(filepath):
    # Route based on your pristine folder taxonomy!
    filepath = filepath.replace('\\', '/')
    lower_filepath = filepath.lower()
    
    if '1_cloud_flash' in lower_filepath: return 'Gemini-3.5-Flash (Cloud)', 'Cloud'
    if '2_cloud_pro' in lower_filepath: return 'Gemini-3.1-Pro (Cloud)', 'Cloud'
    
    filename = os.path.basename(filepath).lower()
    
    if 'qwen' in filename and 'deepseek' not in filename: model = 'Qwen2.5-32B'
    elif 'gemma' in filename: model = 'Gemma-4-31B'
    elif 'deepseek' in filename: model = 'DeepSeek-R1'
    else: model = 'Unknown'

    if '3_dgx_64k' in lower_filepath: return f"{model} [64K]", '64K'
    if '4_dgx_32k' in lower_filepath: return f"{model} [32K]", '32K'
    if '5_dgx_131k' in lower_filepath: return f"{model} [131K]", '131K'
    
    return model, 'Unknown'

def build_dataframes():
    csv_files = glob.glob("Final_Paper_Data/**/*.csv", recursive=True)
    target_files = [f for f in csv_files if 'vram_' not in os.path.basename(f).lower() and 'checkpoint_' not in os.path.basename(f).lower()]
    
    yield_data = []
    missing_markers = ["[missing]", "[skipped]", "none_found", "unknown", "nan", "none", ""]
    
    for file in target_files:
        try:
            df = pd.read_csv(file)
            if df.empty: continue
            
            # Dynamically handle both Bench_*.csv and older Flash spreadsheet CSVs
            if 'Project' in df.columns:
                project_name = str(df['Project'].iloc[0]).upper().strip()
            else:
                project_name = os.path.basename(file).split('_')[0].upper().strip()
                if project_name == 'BENCH':
                    project_name = os.path.basename(file).split('_')[1].upper().strip()

            config_name, context_tier = get_config_name(file)
            model_base = config_name.split(' [')[0]
            
            meta_cols = ['model', 'project', 'timestamp', 'job_id', 'runtime', 'elapsed_seconds', 'context', 'context_length', 'benchmark_model', 'dataset name', 'variant name', 'aliases', 'canonical name', 'type', 'license', 'overall confidence', 'job_created', 'date_created', 'project_created', 'job_updated', 'date_updated', 'project_updated', 'completeness (high conf %)']
            science_cols = [c for c in df.columns if c.lower() not in meta_cols and not str(c).endswith('(C)') and not str(c).endswith('(Telemetry)')]
            
            def is_valid(x): return str(x).strip().lower() not in missing_markers and pd.notna(x)

            if hasattr(df[science_cols], 'map'): populated_cells = df[science_cols].map(is_valid).sum().sum()
            else: populated_cells = df[science_cols].applymap(is_valid).sum().sum()
                
            yield_data.append({'Project': project_name, 'Configuration': config_name, 'Score': int(populated_cells), 'Model_Base': model_base, 'Context': context_tier})
        except Exception: pass 

    log_files = glob.glob("Final_Paper_Data/**/*.log", recursive=True) + glob.glob("Final_Paper_Data/**/*.txt", recursive=True)
    time_data = []
    
    for f in log_files:
        filename = os.path.basename(f).lower()
        if "master" in filename or "agent_log" in filename: continue
        config_name, context_tier = get_config_name(f)

        try:
            with open(f, 'r', encoding='utf-8', errors='ignore') as file:
                active_proj = "UNKNOWN"
                for line in file:
                    proj_match = re.search(r'(?:Project:|PROJECT:|STARTING PROJECT:)\s*([A-Za-z0-9_]+)', line)
                    if proj_match: active_proj = proj_match.group(1).upper()
                    
                    time_match = re.search(r'Workflow Wall-Clock Time.*?\|\s+[\d\.]+\s+\|\s+([\d\.]+)|Workflow Wall-Clock Time:\s*([\d\.]+)s?', line)
                    if time_match and active_proj != "UNKNOWN":
                        val = time_match.group(1) if time_match.group(1) else time_match.group(2)
                        time_data.append({"Project": active_proj, "Configuration": config_name, "Model_Base": config_name.split(' [')[0], "Context": context_tier, "Time (s)": float(val)})
                        active_proj = "UNKNOWN" 
        except: continue

    df_yield = pd.DataFrame(yield_data)
    df_time = pd.DataFrame(time_data)
    return df_yield, df_time

def print_tables_and_plot():
    df_yield, df_time = build_dataframes()
    
    if df_yield.empty or df_time.empty:
        print("❌ Could not generate dataframes. Check Final_Paper_Data folder contents.")
        return

    # Aggregation
    agg_yield = df_yield.groupby(['Project', 'Configuration'])['Score'].max().reset_index()
    agg_time = df_time.groupby(['Project', 'Configuration'])['Time (s)'].mean().reset_index()
    
    pivot_yield = agg_yield.pivot(index='Project', columns='Configuration', values='Score')
    pivot_time = agg_time.pivot(index='Project', columns='Configuration', values='Time (s)')

    ordered_cols = ['Gemini-3.5-Flash (Cloud)', 'Gemini-3.1-Pro (Cloud)', 
                    'Gemma-4-31B [32K]', 'Gemma-4-31B [64K]', 'Gemma-4-31B [131K]', 
                    'Qwen2.5-32B [32K]', 'Qwen2.5-32B [64K]', 'Qwen2.5-32B [131K]', 
                    'DeepSeek-R1 [32K]', 'DeepSeek-R1 [64K]', 'DeepSeek-R1 [131K]']
    
    pivot_yield = pivot_yield.reindex(columns=[c for c in ordered_cols if c in pivot_yield.columns])
    pivot_time = pivot_time.reindex(columns=[c for c in ordered_cols if c in pivot_time.columns])

    # 1. Print Markdown Tables
    print("\n" + "="*130)
    print(" 🏆 FINAL SCIENTIFIC CELL YIELD MATRIX")
    print("="*130)
    if hasattr(pivot_yield, 'map'): formatted_y = pivot_yield.map(lambda x: f"{int(x)}" if pd.notna(x) else "-")
    else: formatted_y = pivot_yield.applymap(lambda x: f"{int(x)}" if pd.notna(x) else "-")
    print(formatted_y.to_markdown())

    print("\n" + "="*130)
    print(" ⏱️ EXECUTION TIME MATRIX (SECONDS)")
    print("="*130)
    print(tabulate(pivot_time.fillna(0).round(1), headers="keys", tablefmt="pipe"))
    print("="*130 + "\n")

    if not PLOTTING_ENABLED:
        print("⚠️ Matplotlib/Seaborn not installed. Skipping chart generation.")
        return

    os.makedirs("Paper_Figures", exist_ok=True)
    
    # 2. Plotting 1: Scientific Cell Yield
    plt.figure(figsize=(14, 8))
    best_yields = df_yield.groupby(['Project', 'Model_Base'])['Score'].max().reset_index()
    sns.barplot(data=best_yields, x='Project', y='Score', hue='Model_Base', palette='viridis')
    plt.title('Peak Scientific Metadata Extraction Yield by Architecture', fontsize=16, pad=15)
    plt.ylabel('Valid Cells Populated', fontsize=14)
    plt.xlabel('Evaluation Dataset', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Model Family')
    plt.tight_layout()
    plt.savefig('Paper_Figures/Figure_1_Cell_Yield.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Plotting 2: Average Compute Time
    plt.figure(figsize=(14, 8))
    sns.barplot(data=agg_time, x='Project', y='Time (s)', hue='Configuration', palette='magma')
    plt.title('Execution Wall-Clock Time by Architecture', fontsize=16, pad=15)
    plt.ylabel('Seconds (Log Scale)', fontsize=14)
    plt.xlabel('Evaluation Dataset', fontsize=14)
    plt.yscale('log') # Log scale is best for time due to Cloud vs Local differences
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Configuration')
    plt.tight_layout()
    plt.savefig('Paper_Figures/Figure_2_Execution_Time.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Plotting 3: VRAM Scaling at 131K
    vram_files = glob.glob("Final_Paper_Data/5_DGX_131K/VRAM/vram_*.csv") + glob.glob("Final_Paper_Data/5_DGX_131K/vram_*.csv")
    if vram_files:
        plt.figure(figsize=(10, 6))
        for f in vram_files:
            try:
                vdf = pd.read_csv(f)
                vdf = vdf[~vdf['Total_VRAM_GB'].astype(str).str.contains("CONTAINER|LAUNCH", na=False)]
                vdf['Total_VRAM_GB'] = pd.to_numeric(vdf['Total_VRAM_GB'])
                vdf['Elapsed_Minutes'] = vdf['Elapsed_Seconds'] / 60.0
                name = os.path.basename(f).replace("vram_", "").replace(".csv", "").upper()
                plt.plot(vdf['Elapsed_Minutes'], vdf['Total_VRAM_GB'], label=name, linewidth=2)
            except Exception: pass
        plt.title("GPU Memory Scaling during 131K Context Extraction", fontsize=14, pad=15)
        plt.xlabel("Execution Time (Minutes)", fontsize=12)
        plt.ylabel("Total VRAM Consumed (GB)", fontsize=12)
        plt.legend(title="Model & Dataset")
        plt.tight_layout()
        plt.savefig("Paper_Figures/Figure_3_VRAM_Scaling_131K.pdf", dpi=300)
        print("✅ Created Paper_Figures/Figure_3_VRAM_Scaling_131K.pdf")

    print("✅ Paper figures successfully generated in ~/Desktop/DeepKG/Paper_Figures/")

if __name__ == "__main__":
    print("⚙️ Crunching final benchmark data from Final_Paper_Data/...")
    print_tables_and_plot()
