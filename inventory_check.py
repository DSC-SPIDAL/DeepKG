import os
import glob

def main():
    categories = {
        "1. Cloud on DGX": [],
        "2. Colab Cloud Flash": [],
        "3. Colab Cloud Pro": [],
        "4. Gemma [32K]": [],
        "5. Gemma [64K]": [],
        "6. Gemma [131K]": [],
        "7. Qwen [32K]": [],
        "8. Qwen [64K]": [],
        "9. Qwen [131K]": [],
        "10. DeepSeek [32K]": [],
        "11. DeepSeek [64K]": [],
        "12. DeepSeek [131K]": []
    }

    files = glob.glob("**/*", recursive=True)
    valid_exts = ['.csv', '.log', '.txt']
    
    for f in files:
        if not any(f.endswith(ext) for ext in valid_exts): continue
        fname = os.path.basename(f).lower()
        
        # Skip system logs, checkpoints, and code files
        if 'vram_' in fname or 'checkpoint_' in fname or 'master_' in fname or 'agent_log' in fname: continue
        if 'requirements' in fname or 'context' in fname or 'readme' in fname or 'licences' in fname: continue

        cat = None
        
        # --- CLOUD CATEGORIES ---
        if 'pro-monolithic' in fname or 'cloud_pro' in f.lower():
            cat = "3. Colab Cloud Pro"
        elif 'consolelog.txt' in fname or '1_cloud_flash' in f.lower():
            if 'pro' not in fname:
                cat = "2. Colab Cloud Flash"
        elif 'cloud' in fname or 'cascade' in fname:
            cat = "1. Cloud on DGX"
            
        # --- LOCAL MODELS ---
        elif 'gemma' in fname:
            if '32k' in f.lower() or 'ablation' in f.lower() or '32768' in fname: cat = "4. Gemma [32K]"
            elif '131k' in f.lower() or 'titan' in f.lower() or '131072' in fname: cat = "6. Gemma [131K]"
            else: cat = "5. Gemma [64K]"
            
        elif 'qwen' in fname:
            if '32k' in f.lower() or 'ablation' in f.lower() or '32768' in fname: cat = "7. Qwen [32K]"
            elif '131k' in f.lower() or 'titan' in f.lower() or '131072' in fname: cat = "9. Qwen [131K]"
            else: cat = "8. Qwen [64K]"
            
        elif 'deepseek' in fname:
            if '32k' in f.lower() or 'ablation' in f.lower() or '32768' in fname: cat = "10. DeepSeek [32K]"
            elif '131k' in f.lower() or 'titan' in f.lower() or '131072' in fname: cat = "12. DeepSeek [131K]"
            else: cat = "11. DeepSeek [64K]"

        # --- PROJECT MAPPING ---
        if cat:
            projs = []
            for p in ['UTSD', 'TIMEBENCH', 'LOTSA', 'TEMPO', 'TSFM', 'KAGGLETS', 'M2', 'M6', 'AEON']:
                if p.lower() in fname: projs.append(p)
            
            # If the filename doesn't have the project, label it UNKNOWN so we can manually inspect it
            if not projs:
                projs = ["UNKNOWN_PROJ"]
            
            for p in projs:
                categories[cat].append((p, f))

    print("\n" + "="*90)
    print(" 📂 DEEPCOLLECTOR FILE INVENTORY BY 12 CATEGORIES")
    print("="*90)
    
    for cat, items in sorted(categories.items()):
        unique_items = list(set(items))
        unique_items.sort(key=lambda x: (x[0], x[1]))
        
        print(f"\n{cat} ({len(unique_items)} files)")
        print("-" * 90)
        if not unique_items:
            print("  (No files found)")
        for proj, path in unique_items:
            print(f"  [{proj.ljust(9)}] -> {path}")
            
    print("\n" + "="*90 + "\n")

if __name__ == "__main__":
    main()
