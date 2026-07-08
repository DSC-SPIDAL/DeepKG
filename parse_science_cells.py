import pandas as pd
import glob
import re
import os

print("=====================================================================================")
print("📊 EXTRAPOLATING ABSOLUTE SCIENTIFIC CELL COUNTS (ALL RUNS)")
print("=====================================================================================")

bench_files = sorted(glob.glob("**/*.csv", recursive=True))
bench_files = [f for f in bench_files if "Bench_" in os.path.basename(f) and "vram_" not in f.lower()]

if not bench_files:
    print("❌ No Bench_*.csv files found in the current directory.")
    exit()

print(f"| {'Benchmark Dataset / Model Configuration':<55} | {'Rows':<5} | {'Populated Cells':<16} |")
print(f"|{'-'*57}|{'-'*7}|{'-'*18}|")

missing_markers = ["[missing]", "[skipped]", "none_found", "unknown", "nan", "none", ""]

for file in bench_files:
    try:
        df = pd.read_csv(file)
        if df.empty:
            continue
            
        meta_cols = ['model', 'project', 'timestamp', 'job_id', 'runtime', 'elapsed_seconds', 'context', 'benchmark_model']
        science_cols = [c for c in df.columns if c.lower() not in meta_cols]
        
        total_rows = len(df)
        
        def is_valid(x):
            return str(x).strip().lower() not in missing_markers and pd.notna(x)

        if hasattr(df[science_cols], 'map'):
            populated_cells = df[science_cols].map(is_valid).sum().sum()
        else:
            populated_cells = df[science_cols].applymap(is_valid).sum().sum()
        
        display_name = os.path.basename(file).replace("Bench_", "").replace(".csv", "")
        display_name = re.sub(r'_2026\d+_\d+', '', display_name)
        
        print(f"| {display_name:<55} | {total_rows:<5} | {populated_cells:<16} |")
        
    except Exception as e:
        pass

print("=====================================================================================")
