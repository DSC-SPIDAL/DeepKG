import pandas as pd
import glob
import re
from tabulate import tabulate
import warnings
import os

warnings.simplefilter(action='ignore', category=FutureWarning)

def clean_percentage(val):
    try:
        if pd.isna(val): return 0.0
        s = str(val)
        match = re.search(r"(\d+\.?\d*)", s)
        return float(match.group(1)) if match else 0.0
    except:
        return 0.0

def generate_report():
    print(f"\n{'='*85}\n🏆 DEEPCOLLECTOR PERFORMANCE LEADERBOARD\n{'='*85}")
    
    files = glob.glob("**/*.csv", recursive=True)
    all_data = []
    
    for f in files:
        if "checkpoint" in f.lower() or "none_found" in f.lower() or "vram_" in f.lower() or "bench_" not in os.path.basename(f).lower(): continue
        try:
            df = pd.read_csv(f)
            
            project_col = next((c for c in df.columns if 'project' in c.lower()), None)
            model_col = next((c for c in df.columns if 'model' in c.lower()), None)
            score_col = next((c for c in df.columns if any(k in c.lower() for k in ['score', 'accuracy', 'completeness'])), None)
            
            if project_col and model_col and score_col:
                df["Clean_Completeness"] = df[score_col].apply(clean_percentage)
                df["Project"] = df[project_col].str.upper()
                df["Benchmark_Model"] = df[model_col]
                all_data.append(df)
        except: continue

    if not all_data:
        print("❌ No valid benchmark CSVs found.")
        return

    df = pd.concat(all_data, ignore_index=True)
    df["Benchmark_Model"] = df["Benchmark_Model"].fillna("Unknown")
    
    summary = df.groupby(["Project", "Benchmark_Model"])["Clean_Completeness"] \
                .mean().unstack().fillna(0).round(1)
    
    print(tabulate(summary, headers="keys", tablefmt="pipe"))
    print(f"{'='*85}\n")

if __name__ == "__main__":
    generate_report()
