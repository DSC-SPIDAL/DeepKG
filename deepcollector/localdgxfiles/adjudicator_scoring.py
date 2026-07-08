import pandas as pd
import numpy as np
import re
import os

# --- CONFIGURATION ---
MASTER_DB_FILE = "Master_All_Runs_Gathered.csv"
GOLDEN_KB_FILE = "GOLDENKB.csv"  # Ensure you downloaded the Datasets tab as CSV
OUTPUT_LEADERBOARD = "Model_Evaluation_Leaderboard.csv"
OUTPUT_ACTIONS = "Actionable_KB_Updates.csv"

def normalize_text(text):
    if pd.isna(text): return ""
    return re.sub(r'\s+', ' ', str(text).lower().strip())

def main():
    if not os.path.exists(MASTER_DB_FILE) or not os.path.exists(GOLDEN_KB_FILE):
        print(f"❌ Missing required files ({MASTER_DB_FILE} or {GOLDEN_KB_FILE}).")
        return
        
    print(f"📥 Loading Master Database: {MASTER_DB_FILE}...")
    master_df = pd.read_csv(MASTER_DB_FILE, low_memory=False)
    
    print(f"📥 Loading Golden Knowledgebase: {GOLDEN_KB_FILE}...")
    try:
        golden_df = pd.read_csv(GOLDEN_KB_FILE)
    except Exception as e:
        print(f"❌ Error loading GoldenKB: {e}")
        return

    pk_col = 'Variant Name' if 'Variant Name' in golden_df.columns else None
    if not pk_col:
        for c in ['Dataset Name', 'Document_ID', 'Dataset']:
            if c in golden_df.columns:
                pk_col = c
                break

    if not pk_col:
        print(f"❌ Primary Key not found in GOLDENKB. Columns: {list(golden_df.columns)}")
        return

    # 1. FILTER OUT METADATA FROM GOLDENKB
    ignore_cols = {'DatasetID', 'Job_Created', 'Date_Created', 'Project_Created', 'Job_Updated', 'Date_Updated', 'Project_Updated', 'Overall Confidence', 'License'}
    kb_core_cols = [c for c in golden_df.columns if not str(c).endswith('(C)') and c not in ignore_cols and c != pk_col]
    
    golden_melted = golden_df.melt(id_vars=[pk_col], value_vars=kb_core_cols, var_name='Schema_Column', value_name='Value_Golden')
    golden_melted['Norm_Value_Golden'] = golden_melted['Value_Golden'].apply(normalize_text)
    golden_melted = golden_melted[golden_melted['Norm_Value_Golden'] != ""]
    
    golden_melted['Dataset_Name_Norm'] = golden_melted[pk_col].astype(str).str.lower().str.strip()
    golden_melted['Schema_Column_Norm'] = golden_melted['Schema_Column'].astype(str).str.lower().str.strip()
    
    master_df['Dataset_Name_Norm'] = master_df['Dataset_Name'].astype(str).str.lower().str.strip()
    master_df['Schema_Column_Norm'] = master_df['Schema_Column'].astype(str).str.lower().str.strip()

    # 2. IDENTIFY NEW DATASETS AND DROP DEEP RESEARCH ONES
    known_datasets_kb = set(golden_melted['Dataset_Name_Norm'].unique())
    extracted_datasets = set(master_df['Dataset_Name_Norm'].unique())
    new_datasets = extracted_datasets - known_datasets_kb
    
    print(f"\n🕵️ Found {len(new_datasets)} New Datasets not in GoldenKB.")
    
    # Drop rows that are New Datasets AND discovered by Deep Research
    if 'Discovery_Source' in master_df.columns:
        dr_mask = (master_df['Dataset_Name_Norm'].isin(new_datasets)) & (master_df['Discovery_Source'].astype(str).str.contains('Deep Research|Agent', case=False, na=False))
        dr_dropped = master_df[dr_mask]['Dataset_Name_Norm'].nunique()
        
        if dr_dropped > 0:
            print(f"🛡️ Filtering out {dr_dropped} new datasets found via 'Deep Research' to isolate pure RAG performance.")
            master_df = master_df[~dr_mask].copy()
            
    # Recalculate new RAG datasets after dropping DR ones
    extracted_datasets = set(master_df['Dataset_Name_Norm'].unique())
    new_rag_datasets = extracted_datasets - known_datasets_kb
    
    print(f"🕵️ Evaluating {len(new_rag_datasets)} New Datasets discovered purely by RAG...")
    dataset_reality_dict = {}
    total_runs = master_df.groupby(['Model_Config', 'Run_Number']).ngroups
    
    for ds in new_rag_datasets:
        runs_finding_ds = master_df[master_df['Dataset_Name_Norm'] == ds].groupby(['Model_Config', 'Run_Number']).ngroups
        if runs_finding_ds >= max(3, total_runs * 0.15):
            dataset_reality_dict[ds] = "REAL_DISCOVERY"
        else:
            dataset_reality_dict[ds] = "HALLUCINATED_DATASET"
            
    hallucinated_count = sum(1 for v in dataset_reality_dict.values() if v == "HALLUCINATED_DATASET")
    print(f"   -> Verdict: {len(new_rag_datasets) - hallucinated_count} deemed Real Discoveries, {hallucinated_count} deemed Hallucinations.")

    # 3. CONDUCT THE JURY VOTE (CELL-LEVEL CONSENSUS)
    print("\n⚖️ Conducting the 'Jury Vote' on Extracted Cells...")
    vote_counts = master_df.groupby(['Project', 'Dataset_Name_Norm', 'Schema_Column_Norm', 'Norm_Value']).size().reset_index(name='Votes')
    total_votes = master_df.groupby(['Project', 'Dataset_Name_Norm', 'Schema_Column_Norm']).size().reset_index(name='Total_Votes')
    
    consensus_df = pd.merge(vote_counts, total_votes, on=['Project', 'Dataset_Name_Norm', 'Schema_Column_Norm'])
    consensus_df['Vote_Pct'] = consensus_df['Votes'] / consensus_df['Total_Votes']
    
    consensus_df = consensus_df.sort_values(by=['Project', 'Dataset_Name_Norm', 'Schema_Column_Norm', 'Votes'], ascending=[True, True, True, False])
    top_consensus = consensus_df.drop_duplicates(subset=['Project', 'Dataset_Name_Norm', 'Schema_Column_Norm']).copy()
    top_consensus.rename(columns={'Norm_Value': 'Consensus_Norm_Value'}, inplace=True)
    
    raw_vals = master_df[['Dataset_Name_Norm', 'Schema_Column_Norm', 'Norm_Value', 'Extracted_Value', 'Dataset_Name', 'Schema_Column']].drop_duplicates(subset=['Dataset_Name_Norm', 'Schema_Column_Norm', 'Norm_Value'])
    top_consensus = pd.merge(top_consensus, raw_vals, left_on=['Dataset_Name_Norm', 'Schema_Column_Norm', 'Consensus_Norm_Value'], right_on=['Dataset_Name_Norm', 'Schema_Column_Norm', 'Norm_Value'], how='left')

    print("🔍 Adjudicating Consensus vs GoldenKB...")
    adjudicated = pd.merge(
        top_consensus,
        golden_melted[['Dataset_Name_Norm', 'Schema_Column_Norm', 'Norm_Value_Golden', 'Value_Golden']],
        on=['Dataset_Name_Norm', 'Schema_Column_Norm'],
        how='outer'
    )
    
    # 4. ADJUDICATION LOGIC
    def determine_truth(row):
        ds_name = str(row['Dataset_Name_Norm'])
        
        # Penalize if it's part of a hallucinated dataset
        if dataset_reality_dict.get(ds_name) == "HALLUCINATED_DATASET":
            return "", "❌ Hallucinated Dataset"
            
        kb_val = str(row['Norm_Value_Golden']) if pd.notna(row['Norm_Value_Golden']) else ""
        cons_val = str(row['Consensus_Norm_Value']) if pd.notna(row['Consensus_Norm_Value']) else ""
        pct = row['Vote_Pct'] if pd.notna(row['Vote_Pct']) else 0
        
        if kb_val == cons_val and kb_val != "":
            return kb_val, "✅ Match"
        elif kb_val == "":
            if pct >= 0.50: 
                return cons_val, "🌟 Novel Discovery (Merge)"
            else:
                return "", "⚠️ Weak Novel (Ignore)"
        elif cons_val != "":
            if pct >= 0.60: 
                return cons_val, "❌ Conflict (Repair KB)"
            else:
                return kb_val, "🛡️ GoldenKB Maintained (Weak Conflict)"
        else:
            return kb_val, "🛡️ GoldenKB Maintained (No Extraction)"

    adjudicated[['Adjudicated_Truth', 'Action']] = adjudicated.apply(determine_truth, axis=1, result_type='expand')

    # Export Actionable Items
    actions_df = adjudicated[adjudicated['Action'].isin(["🌟 Novel Discovery (Merge)", "❌ Conflict (Repair KB)"])].copy()
    actions_df['Dataset_Name_Out'] = actions_df['Dataset_Name'].combine_first(actions_df['Dataset_Name_Norm'])
    actions_df['Schema_Column_Out'] = actions_df['Schema_Column'].combine_first(actions_df['Schema_Column_Norm'])
    
    if not actions_df.empty:
        actions_df['Vote_%'] = (actions_df['Vote_Pct'] * 100).round(1).astype(str) + '%'
        out_cols = ['Project', 'Action', 'Dataset_Name_Out', 'Schema_Column_Out', 'Extracted_Value', 'Value_Golden', 'Votes', 'Total_Votes', 'Vote_%']
        
        if 'Project' not in actions_df.columns:
            actions_df['Project'] = 'Unknown'
        else:
            actions_df['Project'] = actions_df['Project'].fillna('Unknown')
            
        valid_out_cols = [c for c in out_cols if c in actions_df.columns]
        actions_df[valid_out_cols].sort_values(by=['Action', 'Project']).to_csv(OUTPUT_ACTIONS, index=False)
        print(f"   💾 Saved {len(actions_df)} items needing your review/update to: {OUTPUT_ACTIONS}")
    
    # 5. SCORING MODELS (FIXED)
    print("\n🏆 SCORING MODELS (Precision & Severe RAG Hallucinations)...")
    
    # Map Truth back to Master DB
    adjudicated_map = adjudicated.set_index(['Dataset_Name_Norm', 'Schema_Column_Norm'])['Adjudicated_Truth'].to_dict()
    master_df['Adjudicated_Truth'] = master_df.set_index(['Dataset_Name_Norm', 'Schema_Column_Norm']).index.map(adjudicated_map)
    master_df['Adjudicated_Truth'] = master_df['Adjudicated_Truth'].fillna("")
    
    # Any dataset identified as a Hallucinated Dataset gets an automatic Adjudicated_Truth of ""
    hallucinated_datasets = [k for k, v in dataset_reality_dict.items() if v == "HALLUCINATED_DATASET"]
    master_df.loc[master_df['Dataset_Name_Norm'].isin(hallucinated_datasets), 'Adjudicated_Truth'] = ""

    # Create 'Is_Correct' directly on the master_df so it propagates safely
    master_df['Is_Correct'] = ((master_df['Adjudicated_Truth'] != "") & 
                               (master_df['Norm_Value'] == master_df['Adjudicated_Truth'])).astype(int)
    
    # Track severe hallucinations separately for reporting
    master_df['Is_Severe_Hallucination'] = master_df['Dataset_Name_Norm'].isin(hallucinated_datasets).astype(int)

    leaderboard = []
    
    for config, group in master_df.groupby('Model_Config'):
        runs = group['Run_Number'].unique()
        run_precisions, run_yields, run_correct, run_severe = [], [], [], []
        
        for run in runs:
            run_data = group[group['Run_Number'] == run]
            
            # Total cells extracted by this model on this run
            tyield = len(run_data)
            
            # Filter to only rows where we have an Adjudicated_Truth to score against
            gradable_data = run_data[run_data['Adjudicated_Truth'] != ""]
            tcorrect = gradable_data['Is_Correct'].sum()
            tsevere = run_data['Is_Severe_Hallucination'].sum()
            
            gradable_yield = len(gradable_data)
            precision = (tcorrect / gradable_yield * 100) if gradable_yield > 0 else 0
            
            run_yields.append(tyield)
            run_correct.append(tcorrect)
            run_severe.append(tsevere)
            run_precisions.append(precision)
            
        leaderboard.append({
            "Model Configuration": config,
            "Runs Evaluated": len(runs),
            "Avg Yield (All Cells)": round(np.mean(run_yields), 1),
            "Avg Correct (Matches)": round(np.mean(run_correct), 1),
            "Avg Semantic/Conflict Errors": round(np.mean(run_yields) - np.mean(run_correct) - np.mean(run_severe), 1),
            "Avg Severe Dataset Hallucinations": round(np.mean(run_severe), 1),
            "Precision (%)": round(np.mean(run_precisions), 2)
        })
        
    ldf = pd.DataFrame(leaderboard).sort_values(by='Precision (%)', ascending=False)
    
    print("\n==========================================================================================")
    print(" 🥇 FINAL MODEL LEADERBOARD (PRECISION & HALLUCINATION PENALTY)")
    print("==========================================================================================")
    print(ldf.to_markdown(index=False))
    
    ldf.to_csv(OUTPUT_LEADERBOARD, index=False)
    print(f"\n💾 Saved Leaderboard to {OUTPUT_LEADERBOARD}")

if __name__ == "__main__":
    main()
