import pandas as pd
import glob
import os

# Define all expected files
vram_files = {
    "Qwen2.5-32B (UTSD)": "vram_time_series.csv", # Initial run file
    "Qwen2.5-32B (LOTSA)": "vram_qwen_LOTSA.csv",
    "Gemma-4-31B (UTSD)": "vram_gemma4_UTSD.csv",
    "Gemma-4-31B (LOTSA)": "vram_gemma4_LOTSA.csv",
    "DeepSeek-R1-Dist (UTSD)": "vram_deepseek_UTSD.csv",
    "DeepSeek-R1-Dist (LOTSA)": "vram_deepseek_LOTSA.csv"
}

print(f"\n========================================================")
print(f"📊 TITAN HARDWARE ABLATION SUMMARY MATRIX (131K)")
print(f"========================================================\n")

rows = []
for label, filename in vram_files.items():
    if not os.path.exists(filename):
        # Fallback check for case differences
        filename = filename.lower()
        if not os.path.exists(filename):
            rows.append({"Configuration": label, "Survival Time (Min)": "Missing", "Peak VRAM (GB)": "Missing"})
            continue
            
    try:
        df = pd.read_csv(filename)
        df_clean = df[~df["Total_VRAM_GB"].astype(str).str.contains("CONTAINER_DEAD|LAUNCH_FAILED", na=True)].copy()
        
        df_clean["Total_VRAM_GB"] = df_clean["Total_VRAM_GB"].astype(float)
        df_clean["Elapsed_Seconds"] = df_clean["Elapsed_Seconds"].astype(int)
        
        peak_vram = df_clean["Total_VRAM_GB"].max()
        total_seconds = df_clean["Elapsed_Seconds"].max()
        total_minutes = round(total_seconds / 60, 1)
        
        rows.append({
            "Configuration": label,
            "Survival Time (Min)": f"{total_minutes} min",
            "Peak VRAM (GB)": f"{peak_vram} GB"
        })
    except Exception as e:
        rows.append({"Configuration": label, "Survival Time (Min)": "Error", "Peak VRAM (GB)": f"{str(e)}"})

summary_df = pd.DataFrame(rows)
print(summary_df.to_markdown(index=False))
print(f"\n========================================================")
