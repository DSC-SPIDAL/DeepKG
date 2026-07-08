import pandas as pd
import os
import re
import io

print("☁️ Initializing Markdown Table Parser for Colab Console Logs...")

# Check local directory first, then fallback to Desktop root
MASTER_CSV = "Master_All_Runs_Gathered.csv"
if not os.path.exists(MASTER_CSV):
    MASTER_CSV = "/home/geoffrey/Desktop/DeepKG/Master_All_Runs_Gathered.csv"
    if not os.path.exists(MASTER_CSV):
        print(f"❌ Cannot find Master_All_Runs_Gathered.csv.")
        exit(1)

master_df = pd.read_csv(MASTER_CSV)
initial_len = len(master_df)

# Hardcoded exact paths from DGX console output
log_files = [
    "/home/geoffrey/Desktop/DeepKG/Trash/Final_Paper_Data_Old/1_Cloud_Flash/TimeBench_LOTSA_TEMPO_TSFM_Kag_20260702_1433_ConsoleLog.txt",
    "/home/geoffrey/Desktop/DeepKG/Trash/Final_Paper_Data_Old/1_Cloud_Flash/UTSD_20260702_0226_ConsoleLog.txt"
]

extracted_records = []

for file in log_files:
    if not os.path.exists(file):
        print(f"❌ Error: Cannot find file at {file}")
        continue
        
    print(f"📄 Scanning log for tables: {file}")
    current_project = "UNKNOWN"
    table_buffer = []
    
    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Track the active project from the executor output
            proj_match = re.search(r'Processing Project:\s*([A-Za-z0-9_]+)', line, re.IGNORECASE)
            if proj_match:
                current_project = proj_match.group(1).upper()
            
            # If line is a markdown table row (starts and ends with a pipe)
            line_stripped = line.strip()
            if line_stripped.startswith('|') and line_stripped.endswith('|'):
                table_buffer.append(line_stripped)
            else:
                # Table ended, process the buffered lines
                if len(table_buffer) > 2:
                    # Remove the markdown alignment row containing only dashes, colons, pipes, and spaces
                    clean_lines = [r for r in table_buffer if not re.match(r'^\|[-:\s\|]+\|$', r)]
                    table_str = "\n".join(clean_lines)
                    
                    try:
                        # Read markdown table using pipe separator
                        df_parsed = pd.read_csv(io.StringIO(table_str), sep='|', skipinitialspace=True)
                        
                        # Drop the empty first and last columns caused by leading/trailing pipes
                        df_parsed = df_parsed.dropna(axis=1, how='all')
                        df_parsed.columns = df_parsed.columns.str.strip()
                        
                        for _, row in df_parsed.iterrows():
                            ds_name = str(row.get('Dataset Name', 'Unknown')).strip()
                            if ds_name == 'Unknown' or ds_name.lower() == 'nan':
                                continue
                                
                            for col in df_parsed.columns:
                                col_clean = col.strip()
                                if col_clean not in ['Dataset Name', 'Project', 'Unnamed: 0']:
                                    val = str(row[col]).strip()
                                    extracted_records.append({
                                        "Project": current_project,
                                        "Dataset_Name": ds_name,
                                        "Schema_Column": col_clean,
                                        "Model_Config": "Gemini-3.5-Flash (Cloud)",
                                        "Run_Number": "Run_1",
                                        "Extracted_Value": val,
                                        "Norm_Value": val.lower(),
                                        "Confidence_Level": "High",
                                        "Source_File": os.path.basename(file)
                                    })
                    except Exception as e:
                        print(f"  ⚠️ Error parsing a table in {file}: {e}")
                        
                table_buffer = []

if not extracted_records:
    print("❌ No markdown tables found in the logs.")
    exit(1)

print(f"✅ Successfully extracted {len(extracted_records)} cell facts from the Markdown tables.")

new_df = pd.DataFrame(extracted_records)

# Ensure schema alignment for the Master CSV
for col in master_df.columns:
    if col not in new_df.columns:
        new_df[col] = "Unknown"

final_df = pd.concat([master_df, new_df[master_df.columns]], ignore_index=True)

# Drop duplicates to ensure we don't double-count
final_df.drop_duplicates(subset=['Project', 'Dataset_Name', 'Schema_Column', 'Model_Config'], inplace=True)

final_df.to_csv(MASTER_CSV, index=False)
added = len(final_df) - initial_len

print(f"🎉 Inserted {added} Flash extractions into {MASTER_CSV}!")
print("🚀 You can now run: python3 run1_final_adjudicator.py")
