import pandas as pd
import os

print("🏆 Initializing LLM-Calibrated Adjudicator...")

# --- 1. CONFIGURATION ---
MASTER_RUNS_CSV = "Master_All_Runs_Gathered_run2.csv"  # <--- Point to Run 2
PLATINUM_KB_CSV = "deepcollector/localdgxfiles/GOLDENKB.csv"
ORACLE_AUDIT_CSV = "deepcollector/localdgxfiles/Oracle_Full_Dataset_Audit.csv"
OUTPUT_LEADERBOARD = "Model_Evaluation_Leaderboard_run2.csv"

# --- 2. LOAD DATA ---
print(f"Loading {MASTER_RUNS_CSV}, Platinum KB, and Oracle Audit...")
try:
    master_df = pd.read_csv(MASTER_RUNS_CSV)
    kb_df = pd.read_csv(PLATINUM_KB_CSV)
    oracle_df = pd.read_csv(ORACLE_AUDIT_CSV)
except FileNotFoundError as e:
    print(f"❌ Missing required file: {e}")
    exit(1)

# --- 3. BUILD THE SEMANTIC ROSETTA STONE ---
print("Building Semantic Forgiveness Dictionary from Oracle Verdicts...")
semantic_matches = oracle_df[oracle_df['Oracle_Verdict'].str.contains('Semantic Match', na=False)]

rosetta = {}
for _, row in semantic_matches.iterrows():
    ds = str(row['Dataset_Name_Out']).strip()
    col = str(row['Schema_Column_Out']).strip()
    model_val = str(row['Extracted_Value']).strip()
    kb_val = str(row['Value_Golden']).strip()
    rosetta[(ds, col, model_val)] = kb_val

# --- 4. BUILD FAST KB LOOKUP ---
print("Indexing Platinum Knowledgebase...")
kb_dict = {}
for _, row in kb_df.iterrows():
    ds = str(row.get('Dataset_Name', row.get('Dataset', ''))).strip() 
    for col in kb_df.columns:
        if col not in ['Dataset_Name', 'Dataset', 'Project']:
            kb_dict[(ds, col.strip())] = str(row[col]).strip()

# --- 5. SCORE THE RUN ---
print(f"Scoring {len(master_df)} extractions using exact match + semantic forgiveness...")
results = []

for _, row in master_df.iterrows():
    model = row['Model']
    ds = str(row['Dataset_Name_Out']).strip()
    col = str(row['Schema_Column_Out']).strip()
    extracted = str(row['Extracted_Value']).strip()
    
    golden_val = kb_dict.get((ds, col), "Not specified")
    status = "Unknown"
    
    # Logic 1: Exact Match
    if extracted == golden_val:
        if golden_val == "Not specified":
            status = "True Negative" 
        else:
            status = "True Positive (Exact)"
            
    # Logic 2: Semantic Forgiveness (The Oracle approved this variation)
    elif (ds, col, extracted) in rosetta and rosetta[(ds, col, extracted)] == golden_val:
        status = "True Positive (Semantic Match)"
        
    # Logic 3: Missed Fact
    elif extracted == "Not specified" and golden_val != "Not specified":
        status = "False Negative (Missed Fact)"
        
    # Logic 4: Hallucination or Incorrect Data
    else:
        status = "False Positive (Hallucination/Error)"
        
    results.append({
        'Model': model,
        'Dataset': ds,
        'Attribute': col,
        'Status': status
    })

scored_df = pd.DataFrame(results)

# --- 6. GENERATE LEADERBOARD METRICS ---
print("\nCalculating Precision, Recall, and F1 Scores...")
metrics = []
for model in scored_df['Model'].unique():
    m_df = scored_df[scored_df['Model'] == model]
    
    tp = len(m_df[m_df['Status'].str.contains('True Positive')])
    fp = len(m_df[m_df['Status'] == 'False Positive (Hallucination/Error)'])
    fn = len(m_df[m_df['Status'] == 'False Negative (Missed Fact)'])
    tn = len(m_df[m_df['Status'] == 'True Negative'])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    hallucination_rate = fp / len(m_df) if len(m_df) > 0 else 0
    
    metrics.append({
        'Model': model,
        'F1_Score': round(f1, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'Hallucination_Rate': round(hallucination_rate, 4),
        'Total_Extractions': len(m_df),
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn
    })

leaderboard = pd.DataFrame(metrics).sort_values(by='F1_Score', ascending=False)
leaderboard.to_csv(OUTPUT_LEADERBOARD, index=False)

print("\n=====================================================================")
print(f"🏆 FINAL PERFORMANCE LEADERBOARD (RUN 2)")
print("=====================================================================")
print(leaderboard[['Model', 'F1_Score', 'Precision', 'Recall', 'Hallucination_Rate']].to_string(index=False))
