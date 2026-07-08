import json
import os

print("Unpacking DeepCollector.ipynb...")
try:
    with open("DeepCollector.ipynb", "r", encoding="utf-8") as f:
        notebook = json.load(f)
except FileNotFoundError:
    print("❌ Error: DeepCollector.ipynb not found in this directory!")
    exit(1)

for cell in notebook.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        if not source: continue
        
        first_line = source[0].strip()
        if first_line.startswith("%%writefile"):
            file_path = first_line.replace("%%writefile", "").strip()
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as out:
                out.writelines(source[1:])
            print(f"✅ Created: {file_path}")

print("🎉 Unpacking complete!")
