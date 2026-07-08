import os

output_file = "deepcollector_context.txt"
exclude_dirs = {".git", "__pycache__", "venv", ".cache", "data"}
allowed_exts = {".py", ".sh", ".md"} # Get code, scripts, and documentation

with open(output_file, "w", encoding="utf-8") as outfile:
    outfile.write("=== DEEPCOLLECTOR CODEBASE ===\n\n")
    for root, dirs, files in os.walk("."):
        # Modify dirs in-place to skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            # Exclude tokens and bundle code
            if any(file.endswith(ext) for ext in allowed_exts) and "token" not in file:
                filepath = os.path.join(root, file)
                outfile.write(f"\n{'='*80}\nFILE: {filepath}\n{'='*80}\n")
                try:
                    with open(filepath, "r", encoding="utf-8") as infile:
                        outfile.write(infile.read() + "\n")
                except Exception as e:
                    outfile.write(f"Error reading file: {e}\n")

print(f"✅ Successfully bundled all source code into {output_file}")
