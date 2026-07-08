import pandas as pd
import glob
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def extract_context(filename, df):
    col_names = [c.lower() for c in df.columns]
    for col in ['context', 'context_length', 'context_size']:
        if col in col_names:
            actual_col = df.columns[col_names.index(col)]
            return str(df[actual_col].iloc[0])
            
    name = filename.upper()
    if '131072' in name or '128K' in name or '131K' in name: return '131K'
    elif '65536' in name or '64K' in name: return '64K'
    elif '32768' in name or '32K' in name: return '32K'
        
    return 'Unknown Context'

def main():
    print("=====================================================================================")
    print("🔍 ANALYZING DEEPCOLLECTOR EXTRACTS BY CONTEXT LENGTH")
    print("=====================================================================================")
    
    csv_files = glob.glob("**/*.csv", recursive=True)
    csv_files = [f for f in csv_files if "vram_" not in f.lower() and "checkpoint_" not in f.lower() and "bench_" in os.path.basename(f).lower()]
    
    if not csv_files:
        print("❌ No benchmark CSV files found in the current directory.")
        return

    all_data = []
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            
            project_col = next((c for c in df.columns if 'project' in c.lower()), None)
            model_col = next((c for c in df.columns if 'model' in c.lower()), None)
            score_col = next((c for c in df.columns if any(k in c.lower() for k in ['score', 'accuracy', 'completeness'])), None)
            
            if project_col and model_col and score_col:
                context_tier = extract_context(file, df)
                
                for _, row in df.iterrows():
                    score_val = str(row[score_col]).replace('%', '').strip()
                    all_data.append({
                        'Context': context_tier,
                        'Project': row[project_col],
                        'Model': row[model_col],
                        'Score': pd.to_numeric(score_val, errors='coerce')
                    })
        except Exception as e:
            pass

    if not all_data:
        print("❌ Could not extract scoring data. Please check CSV column names.")
        return

    master_df = pd.DataFrame(all_data).dropna(subset=['Score'])
    aggregated = master_df.groupby(['Context', 'Project', 'Model'])['Score'].max().reset_index()
    
    for context_tier in sorted(aggregated['Context'].unique()):
        print(f"\n=====================================================================================")
        print(f"🏆 PERFORMANCE LEADERBOARD: {context_tier} CONTEXT TIER")
        print(f"=====================================================================================")
        
        tier_data = aggregated[aggregated['Context'] == context_tier]
        pivot_table = tier_data.pivot(index='Project', columns='Model', values='Score').fillna(0)
        
        print(pivot_table.round(1).to_markdown())
        print("=====================================================================================")

if __name__ == "__main__":
    main()
