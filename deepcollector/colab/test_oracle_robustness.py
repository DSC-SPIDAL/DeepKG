import pandas as pd
import google.generativeai as genai
import os
import time

# --- CONFIGURATION ---
ACTIONS_FILE = "Actionable_KB_Updates.csv" 
ROBUSTNESS_REPORT = "Oracle_Robustness_Report.csv"

# Ensure your API key is in your DGX environment: export GEMINI_API_KEY="your_key"
api_key = os.environ.get("GEMINI_API_KEY") 
if api_key:
    genai.configure(api_key=api_key)
else:
    print("⚠️ GEMINI_API_KEY not found in environment. Please set it using: export GEMINI_API_KEY='your_key'")

def test_algorithm_robustness():
    print(f"📥 Loading Actionable Updates from {ACTIONS_FILE}...")
    try:
        df = pd.read_csv(ACTIONS_FILE)
    except FileNotFoundError:
        print("❌ Cannot find the file. Run the Adjudicator first.")
        return
    
    # Filter for just the Conflicts
    conflicts = df[df['Action'].str.contains("Conflict", na=False)]
    if conflicts.empty:
        print("No conflicts found to test!")
        return

    # Sample up to 100 conflicts
    sample_size = min(100, len(conflicts))
    test_sample = conflicts.sample(n=sample_size, random_state=42)
    
    print(f"🤖 Testing {sample_size} algorithmic conflicts against the PRO LLM Oracle...")
    
    # ✅ FIXED: Using a heavyweight PRO model for Oracle reasoning
    try:
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
    except Exception as e:
        print(f"⚠️ Could not load model. Error: {e}")
        return
    
    results, semantic_matches, true_conflicts = [], 0, 0
    
    for idx, row in test_sample.iterrows():
        dataset = row.get('Dataset_Name_Out', 'Unknown')
        column = row.get('Schema_Column_Out', 'Unknown')
        kb_val = row.get('Value_Golden', '')
        cons_val = row.get('Extracted_Value', '')
        
        prompt = f"""
        You are a highly intelligent, strict data adjudication Oracle.
        Dataset: {dataset} | Attribute: {column}
        
        Value 1 (From old Knowledgebase): "{kb_val}"
        Value 2 (Newly extracted by advanced models): "{cons_val}"
        
        Are these two values semantically identical in meaning within a data science context? 
        Respond with ONLY the word "MATCH" if they mean the exact same thing (e.g., '1 hour' and 'hourly', or 'ECMWF' and 'European Centre for Medium-Range Weather Forecasts'). 
        Respond with ONLY the word "CONFLICT" if they are factually different numbers, categories, or meanings.
        """
        
        try:
            response = model.generate_content(prompt)
            answer = response.text.strip().upper()
            
            if "MATCH" in answer:
                semantic_matches += 1
                status = "Semantic Match (Math Algorithm was too strict)"
            else:
                true_conflicts += 1
                status = "True Conflict (The models caught an error in the old KB!)"
                
            results.append({"Dataset": dataset, "Column": column, "Old_KB_Value": kb_val, "New_Model_Consensus": cons_val, "Oracle_Verdict": status})
            print(f"Tested [{column}]: KB '{kb_val}' vs New '{cons_val}' -> {status}")
            time.sleep(2) # Respect Pro rate limits
            
        except Exception as e:
            print(f"⚠️ Oracle failed on row: {e}")
        
    print("\n=======================================================")
    print(" ⚖️ ALGORITHM vs. PRO ORACLE ROBUSTNESS REPORT")
    print("=======================================================")
    print(f"Total Conflicts Tested: {sample_size}")
    print(f"🤝 True Conflicts (The new models caught a likely error in your Flash-generated KB!): {true_conflicts}")
    print(f"🤔 False Conflicts (Algorithm flagged a difference, but Oracle says they mean the same): {semantic_matches}")
    
    strictness_error_rate = (semantic_matches / sample_size) * 100
    print(f"\n📊 Algorithm 'Too Strict' Error Rate: {strictness_error_rate:.1f}%")
    
    pd.DataFrame(results).to_csv(ROBUSTNESS_REPORT, index=False)
    print(f"💾 Saved detailed audit to {ROBUSTNESS_REPORT}")

if __name__ == "__main__":
    test_algorithm_robustness()
