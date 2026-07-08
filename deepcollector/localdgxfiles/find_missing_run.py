import pandas as pd

df = pd.read_csv("Master_All_Runs_Gathered.csv")

# Identify the correct column names
model_col = next((c for c in df.columns if c.strip().lower() in ['model_config', 'model', 'benchmark_model']), None)
project_col = next((c for c in df.columns if c.strip().lower() == 'project'), None)

expected_projects = ['UTSD', 'LOTSA', 'TIMEBENCH', 'TEMPO', 'TSFM', 'KAGGLETS', 'M2', 'M6']
expected_models = df[model_col].unique()

print("\n🔍 Scanning Master CSV for missing combinations...")
missing = 0

for proj in expected_projects:
    # Filter data for this specific project
    proj_data = df[df[project_col].str.upper() == proj.upper()]
    found_models = proj_data[model_col].unique()
    
    for mod in expected_models:
        if mod not in found_models:
            print(f"❌ MISSING RUN: Project [{proj}] is missing data for Model [{mod}]")
            missing += 1

if missing == 0:
    print("✅ All combinations are present! No missing files detected in the Master CSV.")
else:
    print(f"\n⚠️ Total Missing Runs: {missing}")
