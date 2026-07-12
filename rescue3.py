import os
import shutil

ROOT_DIR = os.path.expanduser("~/Desktop/DeepKG")
ADMIN_DIR = os.path.join(ROOT_DIR, "Admin")
LAUNCH_DIR = os.path.join(ROOT_DIR, "deepcollector", "localdgxfiles")

# The exact list of files that existed in BOTH dgxshellscripts and localdgxfiles
duplicates = [
    "run_ablation_sweep.sh",
    "run_cloud_suite.sh",
    "run_deepseek_optimized.sh",
    "run_deepseek_suite.sh",
    "run_gemma_suite.sh",
    "run_master_gauntlet.sh",
    "run_qwen_optimized.sh",
    "run_qwen_suite.sh",
    "run_titan_suite.sh",
    "setup.sh",
    "start.sh",
    "start_cloud.sh",
    "vram_monitor.sh"
]

print("🚀 Routing ambiguously tracked shell scripts to localdgxfiles...")

for fname in duplicates:
    src = os.path.join(ADMIN_DIR, fname)
    dest = os.path.join(LAUNCH_DIR, fname)
    
    if os.path.exists(src):
        try:
            shutil.move(src, dest)
            print(f"✅ Moved: {fname} -> {LAUNCH_DIR}")
        except Exception as e:
            print(f"❌ Error moving {fname}: {e}")

print("\n🏁 Ambiguity resolved. You can safely delete the dgxshellscripts folder if it still exists.")
