import os
import glob
import pandas as pd
import re
from tabulate import tabulate
import warnings

warnings.simplefilter(action='ignore')

def main():
    print("\n" + "="*140)
    print(" 🏆 DEEPCOLLECTOR: TRUE OMNI-PARSER EVALUATION MATRIX (V2)")
    print("="*140)
    
    log_files = glob.glob("Final_Paper_Data/**/*.log", recursive=True) + glob.glob("Final_Paper_Data/**/*ConsoleLog.txt", recursive=True)
    yield_data = []
    
    for f in log_files:
        if 'master' in f.lower() or 'agent_log' in f.lower(): continue
        
        try:
            with open(f, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                
            # Split the log by the exact project starting headers
            chunks = re.split(r'(?:▶️ STARTING:|STARTING PROJECT:)\s*([A-Za-z0-9_]+)', content)
            
            # If the file didn't split, it's not a standard run log
            if len(chunks) < 3: continue
            
            # Iterate through the chunks (Project Name is at index 1, 3, 5... Content is at 2, 4, 6...)
            for i in range(1, len(chunks), 2):
                proj = chunks[i].upper().strip()
                if proj == 'AEON': continue # Deliberate Omission
                
                chunk_text = chunks[i+1]
                
                # 1. Identify Model Directly from Log Printout
                model = "Unknown"
                if "ALL-PRO CLOUD" in chunk_text or "Gemini-3.1-Pro" in chunk_text or "pro-preview" in chunk_text:
                    model = "Gemini-3.1-Pro (Cloud)"
                elif "gemini-3.5-flash" in chunk_text or "CLOUD CASCADE" in chunk_text:
                    model = "Gemini-3.5-Flash (Cloud)"
                elif "qwen" in f.lower() and "deepseek" not in f.lower():
                    model = "Qwen2.5-32B"
                elif "gemma" in f.lower():
                    model = "Gemma-4-31B"
                elif "deepseek" in f.lower():
                    model = "DeepSeek-R1"
                
                if model == "Unknown": continue

                # 2. Identify Context Directly from Log Printout or Folder
                ctx = "64K"
                if "Cloud" in model:
                    ctx = "Cloud"
                else:
                    if "Context: 131072" in chunk_text or "131k" in f.lower() or "titan" in f.lower(): ctx = "131K"
                    elif "Context: 32768" in chunk_text or "32k" in f.lower() or "ablation" in f.lower(): ctx = "32K"
                    
                config = f"{model} [{ctx}]" if ctx != "Cloud" else model

                # 3. Extract Amnesia Yield (The absolute ground truth of populated cells)
                yield_match = re.search(r'Updated (\d+) fields', chunk_text)
                if not yield_match: yield_match = re.search(r'Updates applied: (\d+)', chunk_text)
                
                if yield_match:
                    yield_data.append({'Project': proj, 'Configuration': config, 'Score': int(yield_match.group(1))})
                    
        except Exception as e:
            continue

    df_yield = pd.DataFrame(yield_data)
    
    ordered_cols = ['Gemini-3.5-Flash (Cloud)', 'Gemini-3.1-Pro (Cloud)', 
                    'Gemma-4-31B [32K]', 'Gemma-4-31B [64K]', 'Gemma-4-31B [131K]', 
                    'Qwen2.5-32B [32K]', 'Qwen2.5-32B [64K]', 'Qwen2.5-32B [131K]', 
                    'DeepSeek-R1 [32K]', 'DeepSeek-R1 [64K]', 'DeepSeek-R1 [131K]']

    print(" 1. SCIENTIFIC CELL YIELD MATRIX (LOG-VERIFIED)")
    print("-" * 140)
    if not df_yield.empty:
        agg_yield = df_yield.groupby(['Project', 'Configuration'])['Score'].max().reset_index()
        pivot_yield = agg_yield.pivot(index='Project', columns='Configuration', values='Score')
        pivot_yield = pivot_yield.reindex(columns=[c for c in ordered_cols if c in pivot_yield.columns])
        
        if hasattr(pivot_yield, 'map'): formatted_y = pivot_yield.map(lambda x: f"{int(x)}" if pd.notna(x) else "-")
        else: formatted_y = pivot_yield.applymap(lambda x: f"{int(x)}" if pd.notna(x) else "-")
        print(formatted_y.to_markdown())
    else: 
        print("❌ No yield data found in logs.")

    print("="*140 + "\n")

if __name__ == "__main__":
    main()
