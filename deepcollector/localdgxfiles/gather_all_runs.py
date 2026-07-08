import os
import glob
import pandas as pd
import re

# --- CONFIGURATION ---
RESULTS_ROOT_DIR = "Cleaned_Paper_Data" 
OUTPUT_FILE = "Master_All_Runs_Gathered.csv"

# Strict Schema Mapping
SCHEMA_MAP = {
    "Dataset Name": "Variant Name",
    "Detailed Description": "Description",
    "Time interval between points": "Frequency",
    "Number of Time Points": "Num Time Points",
    "Number of Locations/Series": "Num Locations/Series",
    "Primary Source Repository": "Primary Creator",
    "Canonical Name": "Canonical Name",
    "Aliases": "Aliases",
    "Type": "Type",
    "Total Variables": "Total Variables",
    "Variables per Location": "Variables per Location",
    "Domain": "Domain",
    "Primary URL": "Primary URL",
    "Link to Data (Actual Source)": "Link to Data (Actual Source)",
    "Other URL": "Other URL"
}

def normalize_text(text):
    if pd.isna(text): return ""
    return re.sub(r'\s+', ' ', str(text).lower().strip())

def gather_all_runs(root_dir):
    print(f"🔍 Crawling directory: {root_dir} for benchmark CSVs...")
    
    search_pattern = os.path.join(root_dir, "**", "Bench_*.csv")
    all_csvs = glob.glob(search_pattern, recursive=True)
    
    if not all_csvs:
        print("❌ No Bench_*.csv files found! Check your RESULTS_ROOT_DIR path.")
        return

    print(f"✅ Found {len(all_csvs)} total CSV files. Processing...")
    
    file_registry = []
    
    for file in all_csvs:
        filename = os.path.basename(file)
        parent_dir = os.path.basename(os.path.dirname(file))
        
        # Extract Project Name & force Lowercase
        project_match = re.search(r'Bench_([A-Za-z0-9\-]+)_', filename, re.IGNORECASE)
        project_name = project_match.group(1).lower() if project_match else "unknown"
        
        if "kagglets" in filename.lower() or "kaggle" in filename.lower():
            project_name = "kagglets"
            
        time_match = re.search(r'(\d{8}_\d{4})', filename)
        timestamp = time_match.group(1) if time_match else "00000000_0000"
        
        file_registry.append({
            "Filepath": file,
            "Model_Config": parent_dir,
            "Project": project_name,
            "Timestamp": timestamp,
            "Filename": filename
        })

    reg_df = pd.DataFrame(file_registry)
    reg_df = reg_df.sort_values(by=["Model_Config", "Project", "Timestamp"])
    reg_df['Run_Number'] = reg_df.groupby(['Model_Config', 'Project']).cumcount() + 1

    master_data = []
    invalid_vals = {'[missing]', '[skipped]', '', 'n/a', 'nan'}

    for _, row in reg_df.iterrows():
        try:
            df = pd.read_csv(row["Filepath"])
            df_mapped = df.rename(columns=SCHEMA_MAP)
            
            pk_col = 'Variant Name' if 'Variant Name' in df_mapped.columns else None
            if not pk_col:
                for col in ['Dataset Name', 'Document_ID', 'Dataset']:
                    if col in df_mapped.columns:
                        pk_col = col
                        break
            
            if not pk_col or df_mapped.empty:
                continue 
                
            cols = list(df_mapped.columns)
            conf_cols = {c.replace(' (C)', '').replace('(C)', '').strip(): c for c in cols if str(c).strip().endswith('(C)')}
            
            ignore_metadata = {'Job_Created', 'Date_Created', 'Project_Created', 'License', 'Overall Confidence', 'Assignment Confidence', 'Assignment Rationale'}
            base_cols = [c for c in cols if c not in conf_cols.values() and c != pk_col and c not in ignore_metadata and not str(c).startswith('Unnamed')]

            for _, data_row in df_mapped.iterrows():
                dataset_identifier = str(data_row[pk_col]).strip()
                if dataset_identifier.lower() in invalid_vals or not dataset_identifier: 
                    continue
                
                # 💡 CAPTURE DISCOVERY SOURCE FOR STEP 2
                discovery_source = str(data_row.get('Assignment Rationale', 'RAG')).strip()
                
                for base_c in base_cols:
                    if base_c not in data_row: continue
                    val = str(data_row[base_c]).strip() if pd.notna(data_row[base_c]) else ""
                    
                    if val.lower() in invalid_vals: continue
                        
                    conf_col_name = conf_cols.get(base_c)
                    confidence = str(data_row[conf_col_name]).strip() if conf_col_name and conf_col_name in data_row and pd.notna(data_row[conf_col_name]) else ""
                    
                    if not confidence and 'Assignment Confidence' in df.columns:
                        confidence = str(data_row['Assignment Confidence']).strip()
                        
                    master_data.append({
                        "Project": row["Project"],
                        "Dataset_Name": dataset_identifier,
                        "Discovery_Source": discovery_source, 
                        "Schema_Column": base_c,
                        "Model_Config": row["Model_Config"],
                        "Run_Number": f"Run_{row['Run_Number']}",
                        "Extracted_Value": val,
                        "Norm_Value": normalize_text(val),
                        "Confidence_Level": confidence,
                        "Source_File": row["Filename"]
                    })
                
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            pass 
        except Exception:
            pass

    master_df = pd.DataFrame(master_data)
    
    print("\n==================================================")
    print(" 📈 GATHERING REPORT")
    print("==================================================")
    if not master_df.empty:
        print(f"✅ Total Individual Facts Extracted: {len(master_df):,}")
        master_df.to_csv(OUTPUT_FILE, index=False)
        print(f"💾 Saved Master Gathering Database to: {OUTPUT_FILE}")
    else:
        print("❌ No valid data could be extracted.")

if __name__ == "__main__":
    gather_all_runs(RESULTS_ROOT_DIR)
