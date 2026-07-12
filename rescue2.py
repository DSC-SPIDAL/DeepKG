import os
import glob
import shutil

ROOT_DIR = os.path.expanduser("~/Desktop/DeepKG")
ADMIN_DIR = os.path.join(ROOT_DIR, "Admin")
PDF_DIR = os.path.join(ROOT_DIR, "PDFGems")
LAUNCH_DIR = os.path.join(ROOT_DIR, "deepcollector", "localdgxfiles")

# Ensure destination directories exist
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(LAUNCH_DIR, exist_ok=True)

print("🚀 Executing targeted file routing...")

for f in glob.glob(os.path.join(ADMIN_DIR, "*")):
    if not os.path.isfile(f):
        continue
        
    fname = os.path.basename(f)
    dest = None

    # 1. JSON files to Root
    if fname.endswith(".json"):
        dest = os.path.join(ROOT_DIR, fname)
        
    # 2. PDF files to PDFGems
    elif fname.lower().endswith(".pdf"):
        dest = os.path.join(PDF_DIR, fname)
        
    # 3. resume_run3.sh to Root
    elif fname == "resume_run3.sh":
        dest = os.path.join(ROOT_DIR, fname)
        
    # 4. Critical launch/agent files to localdgxfiles
    elif fname in ["start.sh", "run_agent.py", "vram_monitor.sh"]:
        dest = os.path.join(LAUNCH_DIR, fname)
        
    # Execute the move if a destination was mapped
    if dest:
        try:
            shutil.move(f, dest)
            print(f"✅ Moved: {fname} -> {dest}")
        except Exception as e:
            print(f"❌ Error moving {fname}: {e}")

print("\n🏁 File routing complete.")
