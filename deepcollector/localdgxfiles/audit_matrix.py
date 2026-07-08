import os
import re

print("📊 DeepCollector Matrix Audit (Read-Only)\n")

root = os.path.expanduser("~/Desktop/DeepKG/Cleaned_Paper_Data")
projects = ['UTSD', 'LOTSA', 'TIMEBENCH', 'TEMPO', 'TSFM', 'KAGGLETS', 'M2', 'M6']
models = ['qwen', 'gemma', 'deepseek']
contexts = ['32768', '65536', '131072']

def print_matrix(run_name, folder):
    path = os.path.join(root, folder)
    if not os.path.exists(path): return
    
    print(f"================ {run_name.upper()} MATRIX ================")
    
    # Header
    header = "| Model Config     | " + " | ".join([p[:4].ljust(4) for p in projects]) + " |"
    print(header)
    print("|" + "-"*18 + "|" + "|".join(["-"*6]*8) + "|")
    
    files = os.listdir(path)
    csv_files = [f.lower() for f in files if f.endswith('.csv')]
    
    for m in models:
        for c in contexts:
            row = f"| {m.capitalize():<8} [{c[:2]}K] | "
            for p in projects:
                # Match logic
                p_low = p.lower()
                found = False
                for f in csv_files:
                    if m in f and (c in f or (c=='131072' and '128k' in f)):
                        if p == 'M2' and 'm2' in f and 'm6' not in f:
                            found = True
                        elif p == 'M6' and 'm6' in f:
                            found = True
                        elif p not in ['M2', 'M6'] and p_low in f:
                            found = True
                row += "[x]  | " if found else "[ ]  | "
            print(row)
    print("\n")

print_matrix("Run 1", "run1")
print_matrix("Run 2", "run2")
