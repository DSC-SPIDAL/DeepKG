import os
import shutil
from collections import Counter

print("🔍 Executing DeepCollector Inventory & Verification...\n")

root_dir = os.path.expanduser("~/Desktop/DeepKG")
clean_dir = os.path.join(root_dir, "Cleaned_Paper_Data")
colab_dir = os.path.join(clean_dir, "Colab")

os.makedirs(colab_dir, exist_ok=True)

# --- 1. THE ABSOLUTE COLAB HUNTER ---
print("☁️ Hunting down the missing Colab logs...")
colab_targets = ["TimeBench_LOTSA_TEMPO_TSFM_Kag_20260702_1433_ConsoleLog.txt", "UTSD_20260702_0226_ConsoleLog.txt"]
colab_found = 0

# Check if they are already safe inside the Colab folder
existing = [f for f in os.listdir(colab_dir) if f in colab_targets] if os.path.exists(colab_dir) else []

if len(existing) == 2:
    print("  ✅ Safe! Both Colab logs are already in the Colab folder.")
else:
    for dirpath, _, files in os.walk(root_dir):
        if "Cleaned_Paper_Data/Colab" in dirpath: continue
        for file in files:
            if file in colab_targets:
                source = os.path.join(dirpath, file)
                dest = os.path.join(colab_dir, file)
                try:
                    shutil.move(source, dest)
                    print(f"  -> Rescued from: {dirpath.replace(root_dir, '~')}")
                    colab_found += 1
                except Exception as e:
                    print(f"  -> Error moving {file}: {e}")

    if colab_found + len(existing) < 2:
        print("  ❌ WARNING: One or both Colab logs are completely missing from ~/Desktop/DeepKG.")

# --- 2. THE INVENTORY AUDIT ---
print("\n📊 DIRECTORY INVENTORY AUDIT:")
for target in ['run1', 'run2', 'Colab']:
    folder_path = os.path.join(clean_dir, target)
    if not os.path.exists(folder_path):
        print(f"[{target.upper()}] Directory missing.")
        print("-" * 40)
        continue
        
    files = os.listdir(folder_path)
    
    # Granular Breakdown
    checkpoints = len([f for f in files if 'checkpoint' in f.lower()])
    final_csvs = len([f for f in files if f.endswith('.csv') and 'checkpoint' not in f.lower()])
    logs = len([f for f in files if f.endswith('.log')])
    txts = len([f for f in files if f.endswith('.txt') and 'checkpoint' not in f.lower()])
    others = len(files) - (checkpoints + final_csvs + logs + txts)
    
    print(f"[{target.upper()}] Total Files: {len(files)}")
    print(f"  -> Final CSVs:  {final_csvs}")
    print(f"  -> Checkpoints: {checkpoints}")
    print(f"  -> Logs (.log): {logs}")
    print(f"  -> Txts (.txt): {txts}")
    if others > 0:
        print(f"  -> Other Exts:  {others}")
    print("-" * 40)
