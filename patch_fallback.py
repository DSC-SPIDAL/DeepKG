import re
with open("run_agent.py", "r") as f: code = f.read()

# Fix the fallback checker to match the uppercase filenames
old_line = "safe_proj = re.sub(r'[^A-Za-z0-9_\\-]', '_', proj).strip('_')"
new_line = "safe_proj = re.sub(r'[^A-Za-z0-9_\\-]', '_', proj.upper()).strip('_')"
code = code.replace(old_line, new_line)

with open("run_agent.py", "w") as f: f.write(code)
print("✅ run_agent.py Fallback Naming Bug Fixed!")
