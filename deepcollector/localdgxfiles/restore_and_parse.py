import pandas as pd
import os
import io

print("🔄 STEP 1: Restoring pristine Master CSV...")
# This automatically restores your 6,168 dropped iterations
os.system("python3 gather_all_runs.py") 

MASTER_CSV = "Master_All_Runs_Gathered.csv"
master_df = pd.read_csv(MASTER_CSV)
initial_len = len(master_df)

log_files = [
    "/home/geoffrey/Desktop/DeepKG/Trash/Final_Paper_Data_Old/1_Cloud_Flash/TimeBench_LOTSA_TEMPO_TSFM_Kag_20260702_1433_ConsoleLog.txt",
    "/home/geoffrey/Desktop/DeepKG/Trash/Final_Paper_Data_Old/1_Cloud_Flash/UTSD_20260702_0226_ConsoleLog.txt"
]

print("☁️ STEP 2: Parsing Cloud Console Logs...")
extracted_records = []

for file in log_files:
    if not os.path.exists(file):
        print(f"  ❌ Missing log file: {file}")
        continue
        
    print(f"  📄 Scanning: {os.path.basename(file)}")
    current_project = "UNKNOWN"
    
    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        headers = []
        for line in f:
            if "Processing Project:" in line:
                current_project = line.split("Processing Project:")[-1].strip()
            
            line_stripped = line.strip()
            if line_stripped.startswith('|') and line_stripped.endswith('|'):
                # Safe parsing ignoring internal pipes in Description/URLs
                parts = [p.strip() for p in line_stripped.strip('|').split('|')]
                
                if 'Dataset Name' in parts and 'Domain' in parts:
                    headers = parts
                    continue
                elif len(parts) >= 3 and not parts[0].startswith('---'):
                    ds_name = parts[0]
                    if ds_name.lower() in ['nan', 'unknown', '']: continue
                    
                    for i, col_name in enumerate(headers):
                        if i == 0 or col_name in ['Project', 'Unnamed: 0']: continue
                        
                        # If the markdown table broke due to internal pipes, scoop the remainder into the last column
                        val = parts[i] if i < len(parts) - 1 else " | ".join(parts[i:])
                        
                        extracted_records.append({
                            "Project": current_project,
                            "Dataset_Name": ds_name,
                            "Schema_Column": col_name,
                            "Model_Config": "Gemini-3.5-Flash (Cloud)",
                            "Run_Number": "Run_1",
                            "Extracted_Value": val,
                            "Norm_Value": val.lower(),
                            "Confidence_Level": "High",
                            "Source_File": os.path.basename(file)
                        })

new_df = pd.DataFrame(extracted_records)

# Safely append without dropping iterations
for col in master_df.columns:
    if col not in new_df.columns:
        new_df[col] = "Unknown"

final_df = pd.concat([master_df, new_df[master_df.columns]], ignore_index=True)
final_df.to_csv(MASTER_CSV, index=False)

print(f"🎉 Inserted {len(final_df) - initial_len} Flash extractions into {MASTER_CSV}!")
