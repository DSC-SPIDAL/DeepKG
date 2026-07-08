import os

with open("full_codebase.txt", "w", encoding="utf-8") as outfile:
    for root, dirs, files in os.walk("."):
        if ".git" in root or "__pycache__" in root: continue
        for file in files:
            if file.endswith((".py", ".yaml", ".md", ".yml")):
                filepath = os.path.join(root, file)
                outfile.write(f"\n\n{'='*50}\nFILE: {filepath}\n{'='*50}\n")
                with open(filepath, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read())
print("✅ Created full_codebase.txt")
