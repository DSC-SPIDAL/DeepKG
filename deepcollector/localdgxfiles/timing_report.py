import os
import glob
import re
import pandas as pd
from tabulate import tabulate

def generate_timing_report():
    print(f"\n{'='*85}\n⏱️ DEEPCOLLECTOR EXECUTION TIME LEADERBOARD (Seconds)\n{'='*85}")
    
    files = glob.glob("**/*.log", recursive=True) + glob.glob("**/*.txt", recursive=True)
    data = []
    
    for f in files:
        filename = os.path.basename(f).lower()
        if "master" in filename or "agent_log" in filename: continue
            
        model = "Unknown"
        if "qwen" in filename: model = "Qwen2.5-32B"
        elif "deepseek" in filename: model = "DeepSeek-R1"
        elif "gemma4" in filename or "gemma-4" in filename: model = "Gemma-4-31B"
        elif "pro" in filename: model = "Gemini-Pro (Cloud)"
        elif "cloud" in filename or "gemini" in filename: model = "Gemini-Flash (Cloud)"

        try:
            with open(f, 'r', encoding='utf-8', errors='ignore') as file:
                active_project = "UNKNOWN"
                for line in file:
                    proj_match = re.search(r'(?:Project:|PROJECT:|STARTING PROJECT:)\s*([A-Za-z0-9_]+)', line)
                    if proj_match:
                        active_project = proj_match.group(1).upper()
                        
                    # Catch the Wall-Clock Time regardless of formatting
                    time_match = re.search(r'Workflow Wall-Clock Time.*?\|\s+[\d\.]+\s+\|\s+([\d\.]+)|Workflow Wall-Clock Time:\s*([\d\.]+)s?', line)
                    if time_match and active_project != "UNKNOWN":
                        val = time_match.group(1) if time_match.group(1) else time_match.group(2)
                        time_taken = float(val)
                        if time_taken > 0:
                            data.append({"Project": active_project, "Model": model, "Time (s)": time_taken})
                            active_project = "UNKNOWN" 
        except:
            continue

    if not data:
        print("❌ No profiling data found in the logs.")
        return

    df = pd.DataFrame(data)
    summary = df.groupby(["Project", "Model"])["Time (s)"].mean().unstack().fillna(0).round(1)
    
    print(tabulate(summary, headers="keys", tablefmt="pipe"))
    print(f"{'='*85}\n")

if __name__ == "__main__":
    generate_timing_report()
