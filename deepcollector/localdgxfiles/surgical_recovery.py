import os
import shutil

print("🔧 Executing Surgical Recovery & Organization...")

root_dir = os.path.expanduser("~/Desktop/DeepKG")
clean_dir = os.path.join(root_dir, "Cleaned_Paper_Data")
run1_dir = os.path.join(clean_dir, "run1")
colab_dir = os.path.join(clean_dir, "Colab")

# Exact prefixes for DeepCollector files
VALID_PREFIXES = ('bench_', 'utsd_', 'lotsa_', 'timebench_', 'tempo_', 'tsfm_', 'kagglets_', 'kag_', 'm2_', 'm6_')

# 1. Evict Garbage from Run 1
evicted = 0
kept = 0
if os.path.exists(run1_dir):
    for file in os.listdir(run1_dir):
        # If it doesn't strictly start with a project prefix, it's garbage. Kick it back to root.
        if not file.lower().startswith(VALID_PREFIXES):
            shutil.move(os.path.join(run1_dir, file), os.path.join(root_dir, file))
            evicted += 1
        else:
            kept += 1

print(f"🧹 Evicted {evicted} incorrectly matched files from run1.")
print(f"📁 Run 1 now correctly holds {kept} valid benchmark files.")

# 2. Rescue the Colab Logs from the Old Trash Folder
colab_source = os.path.join(root_dir, "Trash/Final_Paper_Data_Old/1_Cloud_Flash")
rescued_colab = 0
if os.path.exists(colab_source):
    for file in os.listdir(colab_source):
        if "20260702" in file and "ConsoleLog" in file:
            shutil.move(os.path.join(colab_source, file), os.path.join(colab_dir, file))
            rescued_colab += 1

print(f"☁️  Rescued {rescued_colab} Colab logs into the Colab folder.")
print("✅ File system is now clean and ready for Run 3.")
