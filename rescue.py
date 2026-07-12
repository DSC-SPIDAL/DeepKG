import os
import subprocess
import shutil

ROOT_DIR = os.path.expanduser("~/Desktop/DeepKG")
ADMIN_DIR = os.path.join(ROOT_DIR, "Admin")

print("🚀 Mapping original file locations from Git...")

# 1. Ask git for the original file paths to use as a map
try:
    git_output = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT_DIR, text=True
    )
    tracked_files = git_output.strip().split('\n')
except Exception as e:
    print("❌ Error running git ls-files. Are you in the DeepKG directory?", e)
    exit(1)

# Map filename -> list of full relative paths
file_map = {}
for path in tracked_files:
    fname = os.path.basename(path)
    if fname not in file_map:
        file_map[fname] = []
    file_map[fname].append(path)

print("🚚 Moving files out of Admin and back to their original folders...\n")

for f in os.listdir(ADMIN_DIR):
    src_path = os.path.join(ADMIN_DIR, f)
    
    if not os.path.isfile(src_path):
        continue

    # If the file is tracked in Git, we know where it belongs
    if f in file_map:
        # Check for duplicate filenames across different folders
        if len(file_map[f]) == 1:
            rel_path = file_map[f][0]
            dest_path = os.path.join(ROOT_DIR, rel_path)
            
            # Ensure the target directory exists
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            try:
                shutil.move(src_path, dest_path)
                print(f"✅ Moved: {f} -> {rel_path}")
            except Exception as e:
                print(f"❌ Error moving {f}: {e}")
        else:
            # Looking at the GitHub structure, there are some files with identical names 
            # in different folders (like setup.sh or start.sh).
            print(f"⚠️ Ambiguous duplicate found: '{f}'")
            print(f"   Could belong to: {file_map[f]}")
            print(f"   Leaving {f} in Admin for you to move manually.")
    else:
        # If you created a brand new file on July 10 that isn't in Git yet,
        # it won't be in the map. Leave it safe in Admin.
        print(f"ℹ️ Untracked file (leaving in Admin): {f}")

print("\n🏁 Restoration complete. Check your files!")
