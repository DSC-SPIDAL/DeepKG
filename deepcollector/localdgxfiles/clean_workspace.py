import os
import shutil

print("🧹 Executing Workspace Clean-Up...")

root_dir = os.path.expanduser("~/Desktop/DeepKG")
clean_dir = os.path.join(root_dir, "Cleaned_Paper_Data")
trash_dir = os.path.join(root_dir, "Trash")

run1_dir = os.path.join(clean_dir, "run1")
run2_dir = os.path.join(clean_dir, "run2")
colab_dir = os.path.join(clean_dir, "Colab")

# Create structure
for d in [run1_dir, run2_dir, colab_dir, trash_dir]:
    os.makedirs(d, exist_ok=True)

moved_counts = {"run1": 0, "run2": 0, "Colab": 0, "Trash": 0}

for dirpath, _, files in os.walk(root_dir):
    # Skip destination folders to avoid recursive loops
    if "Cleaned_Paper_Data" in dirpath or "Trash" in dirpath: 
        continue
        
    for file in files:
        filepath = os.path.join(dirpath, file)
        
        # Protect Golden Data and Scripts
        if file in ["GOLDENKB.csv", "Oracle_Full_Dataset_Audit.csv"] or file.endswith(".py") or file.endswith(".sh"):
            continue

        # 1. Colab Logs
        if "ConsoleLog" in file and "20260702" in file:
            shutil.move(filepath, os.path.join(colab_dir, file))
            moved_counts["Colab"] += 1
            
        # 2. Run 2 (Files stamped with 20260707 or explicitly "run2")
        elif ("20260707" in file or "run2" in file.lower()) and (file.endswith(".csv") or file.endswith(".log") or file.endswith(".txt")):
            shutil.move(filepath, os.path.join(run2_dir, file))
            moved_counts["run2"] += 1
            
        # 3. Run 1 (Older Bench files and TimeBench recovery)
        elif ("Bench_" in file or "Gemma_TIMEBENCH" in file) and (file.endswith(".csv") or file.endswith(".log")):
            shutil.move(filepath, os.path.join(run1_dir, file))
            moved_counts["run1"] += 1
            
        # 4. Trash (Stray CSVs, old Master files, checkpoints)
        elif file.endswith(".csv") or "checkpoint" in file.lower():
            shutil.move(filepath, os.path.join(trash_dir, file))
            moved_counts["Trash"] += 1

print("\n✅ Clean-up Complete!")
print(f" -> Run 1 Files: {moved_counts['run1']}")
print(f" -> Run 2 Files: {moved_counts['run2']}")
print(f" -> Colab Files: {moved_counts['Colab']}")
print(f" -> Trashed Files: {moved_counts['Trash']}")
