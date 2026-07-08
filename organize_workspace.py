import os
import shutil
import glob

print("🧹 Organizing DGX Workspace...")

root_dir = os.path.expanduser("~/Desktop/DeepKG")
clean_dir = os.path.join(root_dir, "Cleaned_Paper_Data")
trash_dir = os.path.join(root_dir, "Trash_Bin")

run1_dir = os.path.join(clean_dir, "run1")
run2_dir = os.path.join(clean_dir, "run2")
colab_dir = os.path.join(clean_dir, "Colab")

for d in [run1_dir, run2_dir, colab_dir, trash_dir]:
    os.makedirs(d, exist_ok=True)

# 1. Move Colab Logs (Targeting specific execution dates/names)
for root, _, files in os.walk(root_dir):
    if "Trash_Bin" in root or "Cleaned_Paper_Data/Colab" in root: continue
    for file in files:
        if "ConsoleLog" in file and "20260702" in file:
            shutil.move(os.path.join(root, file), os.path.join(colab_dir, file))
            print(f"  -> Moved Colab file: {file}")

# 2. Move Run 2 Files (Targeting today's date 20260707)
for root, _, files in os.walk(root_dir):
    if "Trash_Bin" in root or "Cleaned_Paper_Data/run2" in root: continue
    for file in files:
        if ("Bench_" in file or "M2_" in file or "M6_" in file or "run2" in file.lower()) and "20260707" in file:
            shutil.move(os.path.join(root, file), os.path.join(run2_dir, file))
            print(f"  -> Moved Run 2 file: {file}")

# 3. Move Run 1 Files (Older Bench_ files)
for root, _, files in os.walk(root_dir):
    if "Trash_Bin" in root or "deepcollector" in root or "run1" in root or "run2" in root or "Colab" in root: 
        continue
    for file in files:
        if file.startswith("Bench_") and (file.endswith(".csv") or file.endswith(".log")):
            shutil.move(os.path.join(root, file), os.path.join(run1_dir, file))
            print(f"  -> Moved Run 1 file: {file}")

# 4. Move old intermediate CSVs to Trash
for f in glob.glob(os.path.join(root_dir, "*.csv")):
    fname = os.path.basename(f)
    if fname not in ["GOLDENKB.csv", "Oracle_Full_Dataset_Audit.csv"]:
        shutil.move(f, os.path.join(trash_dir, fname))
        print(f"  -> Trashed: {fname}")

print("\n✅ Workspace successfully organized into Cleaned_Paper_Data subfolders!")
