import os
import shutil

print("🧹 Executing Unified Absolute File Sort & VRAM Rescue...")

root_dir = os.path.expanduser("~/Desktop/DeepKG")
clean_dir = os.path.join(root_dir, "Cleaned_Paper_Data")
trash_dir = os.path.join(root_dir, "Trash_Bin")

run1_dir = os.path.join(clean_dir, "run1")
run2_dir = os.path.join(clean_dir, "run2")
colab_dir = os.path.join(clean_dir, "Colab")
vram_dir = os.path.join(clean_dir, "VRAM_Logs")

for d in [run1_dir, run2_dir, colab_dir, vram_dir, trash_dir]:
    os.makedirs(d, exist_ok=True)

# 1. Immutable System Files (Do not touch)
SAFE_EXTS = ('.py', '.sh', '.json', '.ipynb', '.md', '.zip')
SAFE_FILES = ['goldenkb_platinum.csv', 'oracle_full_dataset_audit.csv', 'makefile', 'deepcollector_context.txt']

# 2. Known Garbage (Note: 'vram' is explicitly removed so it can be rescued)
TRASH_WORDS = ['checkpoint', 'ablation', 'master', 'leaderboard', 'actionable', 'plot.txt', 'gatherout.txt']

# 3. Valid Project Identifiers
PROJECTS = ['utsd', 'lotsa', 'timebench', 'tempo', 'tsfm', 'kagglets', 'kag', 'm2', 'm6']

moved_counts = {"run1": 0, "run2": 0, "Colab": 0, "VRAM": 0, "Trash_Bin": 0, "Ignored": 0}

def move_file(src, dest_folder):
    dest_path = os.path.join(dest_folder, os.path.basename(src))
    try:
        shutil.move(src, dest_path)
        return True
    except shutil.Error:
        # If the file already exists in the destination, overwrite it safely
        os.replace(src, dest_path)
        return True
    except Exception:
        return False

for dirpath, _, files in os.walk(root_dir):
    # Prevent loops into our NEW clean directories and codebase
    # We explicitly allow traversal into the old "Trash" to rescue things
    if any(skip in dirpath for skip in ["Cleaned_Paper_Data/run1", "Cleaned_Paper_Data/run2", "Cleaned_Paper_Data/Colab", "Cleaned_Paper_Data/VRAM_Logs", "Trash_Bin", "deepcollector"]):
        continue
        
    for file in files:
        f_low = file.lower()
        filepath = os.path.join(dirpath, file)
        
        # Protect code and configuration
        if f_low.endswith(SAFE_EXTS) or f_low in SAFE_FILES or 'goldenkb' in f_low:
            moved_counts["Ignored"] += 1
            continue
            
        # RESCUE VRAM FILES FIRST
        if 'vram' in f_low and f_low.endswith('.csv'):
            if move_file(filepath, vram_dir): moved_counts["VRAM"] += 1
            continue
            
        # Send known intermediate/ablation files straight to Trash
        if any(w in f_low for w in TRASH_WORDS):
            if move_file(filepath, trash_dir): moved_counts["Trash_Bin"] += 1
            continue
            
        # We only care about CSVs, Execution Logs, and Console TXTs
        if not f_low.endswith(('.csv', '.log', '.txt')):
            if move_file(filepath, trash_dir): moved_counts["Trash_Bin"] += 1
            continue
            
        # BUCKET 1: Colab / Cloud
        if 'cloud' in f_low or 'colab' in f_low or '20260702' in f_low:
            if move_file(filepath, colab_dir): moved_counts["Colab"] += 1
            continue
            
        # BUCKET 2: Run 2
        if 'run2' in f_low or '20260707' in f_low or '20260708' in f_low:
            if move_file(filepath, run2_dir): moved_counts["run2"] += 1
            continue
            
        # BUCKET 3: Run 1
        is_run1 = False
        for p in PROJECTS:
            if p in ['m2', 'm6']:
                if f"_{p}_" in f_low or f"bench_{p}_" in f_low or f_low.startswith(f"{p}_"):
                    is_run1 = True
                    break
            elif p in f_low:
                is_run1 = True
                break
                
        if is_run1:
            if move_file(filepath, run1_dir): moved_counts["run1"] += 1
        else:
            # Random unassociated CSVs/Logs go to Trash
            if move_file(filepath, trash_dir): moved_counts["Trash_Bin"] += 1

print("\n✅ Unified Absolute File Sort & Rescue Complete.")
print(f"  -> Run 1 Files:   {moved_counts['run1']}")
print(f"  -> Run 2 Files:   {moved_counts['run2']}")
print(f"  -> Colab Files:   {moved_counts['Colab']}")
print(f"  -> VRAM Logs:     {moved_counts['VRAM']}")
print(f"  -> Sent to Trash: {moved_counts['Trash_Bin']}")
