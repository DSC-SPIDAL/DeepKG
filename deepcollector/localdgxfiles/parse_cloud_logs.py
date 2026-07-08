import pandas as pd
import re
import os

print("☁️ Initializing Cloud Console Log Parser for Gemini 3.5 Flash...")

# 1. Anchor to the DeepKG Root Directory
current_dir = os.getcwd()
if "DeepKG" in current_dir:
    deepkg_root = current_dir.split("DeepKG")[0] + "DeepKG"
else:
    # Fallback to standard DGX desktop path
    deepkg_root = os.path.expanduser("~/Desktop/DeepKG")

print(f"🔍 Searching for Colab logs starting recursively from: {deepkg_root}")

# 2. Verify Master CSV exists in current execution directory
MASTER_CSV = "Master_All_Runs_Gathered.csv"
if not os.path.exists(MASTER_CSV):
    print(f"❌ Cannot find {MASTER_CSV} in the current directory.")
    exit(1)

master_df = pd.read_csv(MASTER_CSV)
initial_len = len(master_df)

# 3. Target the specific Colab logs
target_log_names = [
    "TimeBench_LOTSA_TEMPO_TSFM_Kag_20260702_1433_ConsoleLog.txt", 
    "UTSD_20260702_0226_ConsoleLog.txt"
]

found_files = []
for root, dirs, files in os.walk(deepkg_root):
    for file in files:
        if file in target_log_names:
            found_files.append(os.path.join(root, file))

if not found_files:
    print(f"❌ Could not find the specified ConsoleLog.txt files anywhere in {deepkg_root}.")
    exit(1)

extracted_records = []

# 4. Dual-Regex Engine for Extraction
# Pattern A: Catches standard Python dictionary prints e.g., {'Project': 'UTSD', 'Dataset_Name': '...', ...}
dict_pattern = re.compile(
    r"['\"]Project['\"]\s*:\s*['\"]([^'\"]+)['\"].*?['\"]Dataset_Name['\"]\s*:\s*['\"]([^'\"]+)['\"].*?['\"]Schema_Column['\"]\s*:\s*['\"]([^'\"]+)['\"].*?['\"]Extracted_Value['\"]\s*:\s*['\"]([^'\"]+)['\"]", 
    re.IGNORECASE
)

# Pattern B: Catches pipe-delimited or custom string logs e.g., Project: UTSD | Dataset_Name: ...
pipe_pattern = re.compile(
    r"Project:\s*(.*?)\s*\|\s*Dataset_Name:\s*(.*?)\s*\|\s*Schema_Column:\s*(.*?)\s*\|\s*(?:Extracted_)?Value:\s*(.*?)(?:$|\||\n)", 
    re.IGNORECASE
)

# 5. Sweep the Logs
for file_path in found_files:
    print(f"📄 Scanning found log: {file_path}")
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = dict_pattern.search(line) or pipe_pattern.search(line)
            if match:
                extracted_records.append({
                    "Project": match.group(1).strip(),
                    "Dataset_Name": match.group(2).strip(),
                    "Schema_Column": match.group(3).strip(),
                    "Model_Config": "Gemini-3.5-Flash (Cloud)",
                    "Run_Number": "Run_1",
                    "Extracted_Value": match.group(4).strip(),
                    "Norm_Value": match.group(4).strip().lower(),
                    "Confidence_Level": "High",
                    "Source_File": os.path.basename(file_path)
                })

# 6. Injection and Alignment
if not extracted_records:
    print("⚠️ No extractions found. The log print format might differ from standard dicts/pipes.")
    print("-> Please paste a single line from your log showing how an extraction was printed, and I will adjust the regex instantly.")
    exit(1)

print(f"✅ Extracted {len(extracted_records)} valid records from console logs.")

new_df = pd.DataFrame(extracted_records)

# Ensure schema alignment with the Master CSV
for col in master_df.columns:
    if col not in new_df.columns:
        new_df[col] = "Unknown"
        
final_df = pd.concat([master_df, new_df[master_df.columns]], ignore_index=True)

# Drop duplicates to ensure we don't double-count if the script is run twice
final_df.drop_duplicates(subset=['Project', 'Dataset_Name', 'Schema_Column', 'Model_Config'], inplace=True)

final_df.to_csv(MASTER_CSV, index=False)
added_count = len(final_df) - initial_len

print(f"🎉 Successfully merged {added_count} unique Flash records into {MASTER_CSV}!")
print("🚀 You can now safely re-run: python3 run1_final_adjudicator.py")
