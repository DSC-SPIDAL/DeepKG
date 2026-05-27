file_path = "deepcollector/kb/manager.py"
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

with open(file_path, "w", encoding="utf-8") as f:
    for line in lines:
        if 'secondary_text = " | ".join([f"Alt:' in line:
            # Safely calculate the exact number of leading spaces
            spaces = len(line) - len(line.lstrip())
            indent = " " * spaces
            fixed_line = indent + 'secondary_text = " | ".join(["Alt: " + str(u).replace(\'"\', \'""\').replace(\'\\n\', \'\').replace(\'\\r\', \'\') for u in urls[1:]])\n'
            f.write(fixed_line)
        else:
            f.write(line)

print("✅ File successfully patched!")
