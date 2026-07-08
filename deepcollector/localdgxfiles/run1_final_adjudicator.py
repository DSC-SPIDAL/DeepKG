import pandas as pd
import os

print("🏆 Initializing LLM-Calibrated Adjudicator for Run 1...")

# --- 1. LOAD RUN 1 DATA ---
MASTER_RUNS_CSV = "Master_All_Runs_Gathered.csv"  # Run 1 Master
PLATINUM_KB_CSV = "GOLDENKB.csv"                  # Repaired Golden KB
ORACLE_AUDIT_CSV = "Oracle_Full_Dataset_Audit.csv"
OUTPUT_LEADERBOARD = "Model_Evaluation_Leaderboard_Run1_Final.csv"

try:
    master_df = pd.read_csv(MASTER_RUNS_CSV)
    kb_df = pd.read_csv(PLATINUM_KB_CSV)
    oracle_df = pd.read_csv(ORACLE_AUDIT_CSV)
except FileNotFoundError as e:
    print(f"❌ Missing required file: {e}")
    exit(1)

# --- 2. ORACLE FRACTION ANALYSIS ---
print("\n📊 Analyzing Oracle Verdict Fractions...")
verdict_counts = oracle_df['Oracle_Verdict'].value_counts()
verdict_percentages = oracle_df['Oracle_Verdict'].value_counts(normalize=True) * 100

for verdict, count in verdict_counts.items():
    pct = verdict_percentages[verdict]
    print(f" - {verdict}: {count} ({pct:.1f}%)")

# --- 3. BUILD THE SEMANTIC ROSETTA STONE ---
print("\n🧠 Building Semantic Forgiveness Dictionary...")
# The Oracle Audit CSV uses the _Out suffixes from the conflict generator
semantic_matches = oracle_df[oracle_df['Oracle_Verdict'].str.contains('Semantic Match', na=False)]

rosetta = {}
for _, row in semantic_matches.iterrows():
    ds = str(row['Dataset_Name_Out']).strip()
    col = str(row['Schema_Column_Out']).strip()
    model_val = str(row['Extracted_Value']).strip()
    kb_val = str(row['Value_Golden']).strip()
    rosetta[(ds, col, model_val)] = kb_val

print(f"Loaded {len(rosetta)} semantic aliases for forgiveness.")

# --- 4. BUILD FAST KB LOOKUP ---
print("Indexing Platinum Knowledgebase...")
kb_dict = {}
for _, row in kb_df.iterrows():
    ds = str(row.get('Dataset_Name', row.get('Dataset', ''))).strip() 
    for col in kb_df.columns:
        if col not in ['Dataset_Name', 'Dataset', 'Project']:
            kb_dict[(ds, col.strip())] = str(row[col]).strip()

# --- 5. SCORE ALL RUN 1 EXTRACTIONS ---
print(f"\n⚖️ Scoring {len(master_df)} extractions using exact match + semantic forgiveness...")
results = []

for _, row in master_df.iterrows():
    # Using the exact keys from your gather_all_runs.py snippet
    model = str(row['Model_Config']).strip()
    ds = str(row['Dataset_Name']).strip()
    col = str(row['Schema_Column']).strip()
    extracted = str(row['Extracted_Value']).strip()
    
    golden_val = kb_dict.get((ds, col), "Not specified")
    status = "Unknown"
    
    # Exact Match
    if extracted == golden_val:
        if golden_val == "Not specified":
            status = "True Negative" 
        else:
            status = "True Positive (Exact)"
            
    # Semantic Forgiveness
    elif (ds, col, extracted) in rosetta and rosetta[(ds, col, extracted)] == golden_val:
        status = "True Positive (Semantic Match)"
        
    # Missed Fact
    elif extracted == "Not specified" and golden_val != "Not specified":
        status = "False Negative (Missed Fact)"
        
    # Hallucination or Error
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
print("\n📈 Calculating Precision, Recall, and F1 Scores for Run 1...")
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
print("🏆 FINAL MODEL PERFORMANCE LEADERBOARD (RUN 1 - ORACLE CALIBRATED)")
print("=====================================================================")
print(leaderboard[['Model', 'F1_Score', 'Precision', 'Recall', 'Hallucination_Rate']].to_string(index=False))
print(f"\n✅ Saved to {OUTPUT_LEADERBOARD}")
