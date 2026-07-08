import os
import glob
import pandas as pd
import re
from tabulate import tabulate

def analyze_outputs():
    print(f"\n{'='*95}")
    print("🔍 DEEPCOLLECTOR PIPELINE ANOMALY & EFFICIENCY AUDIT")
    print(f"{'='*95}")
    
    csv_files = glob.glob("**/*.csv", recursive=True)
    log_files = glob.glob("**/*.log", recursive=True) + glob.glob("**/*.txt", recursive=True)
    
    csv_stats = []
    missing_markers = ["[missing]", "[skipped]", "none_found", "unknown", "nan", "none", ""]
    
    for f in csv_files:
        if "checkpoint" in f.lower() or "none_found" in f.lower() or "vram_" in f.lower() or "bench_" not in os.path.basename(f).lower(): continue
        try:
            df = pd.read_csv(f)
            if "Project" not in df.columns or "Benchmark_Model" not in df.columns: continue
            
            proj = df["Project"].str.upper().iloc[0]
            raw_model = df["Benchmark_Model"].iloc[0]
            model = "Gemini-Pro (Cloud)" if "pro" in str(raw_model).lower() else ("Gemini-Flash (Cloud)" if "gemini" in str(raw_model).lower() else raw_model)

            total_rows = len(df)
            
            meta_cols = ['project', 'benchmark_model', 'dataset name', 'timestamp', 'job_id', 'runtime', 'elapsed_seconds', 'context', 'context_length']
            science_cols = [c for c in df.columns if c.lower() not in meta_cols]
            if not science_cols: continue
            
            def is_missing(x):
                return str(x).strip().lower() in missing_markers or pd.isna(x)

            if hasattr(df[science_cols], 'map'):
                missing_count = df[science_cols].map(is_missing).sum().sum()
            else:
                missing_count = df[science_cols].applymap(is_missing).sum().sum()
            
            total_cells = df[science_cols].size
            anomaly_rate = (missing_count / total_cells) * 100 if total_cells > 0 else 0
            
            csv_stats.append({
                "Project": proj,
                "Model": model,
                "Rows Extracted": total_rows,
                "Stub/Missing Rate (%)": round(anomaly_rate, 1),
                "Filename": os.path.basename(f)
            })
        except:
            continue

    timing_data = {}
    for f in log_files:
        filename = os.path.basename(f).lower()
        if "master" in filename: continue
        try:
            with open(f, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                match = re.search(r'Workflow Wall-Clock Time.*?\|\s+[\d\.]+\s+\|\s+([\d\.]+)|Workflow Wall-Clock Time:\s*([\d\.]+)s?', content)
                if match:
                    val = match.group(1) if match.group(1) else match.group(2)
                    proj_match = re.search(r'(utsd|timebench|lotsa|tempo|tsfm|kagglets|m2|m6)', filename)
                    model_str = "Qwen" if "qwen" in filename else ("DeepSeek" if "deepseek" in filename else ("Gemma" if "gemma" in filename else ("Gemini-Pro (Cloud)" if "pro" in filename else "Gemini-Flash (Cloud)")))
                    if proj_match:
                        key = (proj_match.group(1).upper(), model_str)
                        timing_data[key] = float(val)
        except:
            continue

    master_audit = []
    for stat in csv_stats:
        proj = stat["Project"]
        m_short = "Qwen" if "Qwen" in stat["Model"] else ("DeepSeek" if "DeepSeek" in stat["Model"] else ("Gemma" if "Gemma" in stat["Model"] else ("Gemini-Pro (Cloud)" if "Pro" in stat["Model"] else "Gemini-Flash (Cloud)")))
        
        runtime = timing_data.get((proj, m_short), 0.0)
        time_per_row = (runtime / stat["Rows Extracted"]) if stat["Rows Extracted"] > 0 and runtime > 0 else 0
        
        master_audit.append([
            proj, m_short, stat["Rows Extracted"],
            f"{round(runtime/60, 1)} min" if runtime > 0 else "N/A",
            f"{round(time_per_row, 1)}s" if time_per_row > 0 else "N/A",
            f"{stat['Stub/Missing Rate (%)']}%",
            "⚠️ HIGH CELL MISSING RATE" if stat["Stub/Missing Rate (%)"] > 40 else "🟢 HEALTHY"
        ])

    headers = ["Project", "Model", "Rows", "Total Time", "Time/Row", "Missing Cell %", "Status"]
    if master_audit:
        print(tabulate(sorted(master_audit), headers=headers, tablefmt="pipe"))
    print(f"{'='*95}\n")

if __name__ == "__main__":
    analyze_outputs()
