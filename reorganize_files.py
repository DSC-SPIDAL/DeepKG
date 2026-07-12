import os
import glob
import re
import pandas as pd
import shutil
from datetime import datetime
from collections import defaultdict
from tabulate import tabulate  # Added the missing import!

ROOT_DIR = os.path.expanduser("~/Desktop/DeepKG")
COLAB_DIR = os.path.join(ROOT_DIR, "ALL_Colab")
DGX_DIR = os.path.join(ROOT_DIR, "ALL_DGX")
ADMIN_DIR = os.path.join(ROOT_DIR, "Admin")

os.makedirs(COLAB_DIR, exist_ok=True)
os.makedirs(DGX_DIR, exist_ok=True)
os.makedirs(ADMIN_DIR, exist_ok=True)

PROJECTS = ['UTSD', 'TIMEBENCH', 'LOTSA', 'TEMPO', 'TSFM', 'KAGGLETS', 'M2', 'M6']
MODELS = ['Gemma', 'Qwen', 'DeepSeek']
CONTEXTS = ['32K', '64K', '131K']

def get_project(s):
    s = str(s).upper()
    if 'KAG' in s: return 'KAGGLETS'
    for p in PROJECTS:
        if p in s:
            if p == 'M2' and 'M6' in s: continue
            return p
    return 'Unknown'

def get_model(s):
    s = str(s).lower()
    if 'pro' in s or 'monolithic' in s: return 'Cloud-Pro'
    if 'flash' in s or 'cascade' in s or 'consolelog' in s: return 'Cloud-Flash'
    if 'deepseek' in s or 'r1' in s: return 'DeepSeek'
    if 'qwen' in s: return 'Qwen'
    if 'gemma' in s: return 'Gemma'
    return 'Unknown'

def get_context(s, model):
    if model in ['Cloud-Pro', 'Cloud-Flash']: return 'Cloud'
    s = str(s).lower()
    s = re.sub(r'(32|31|27|14|7|70|1\.5)b', '', s, flags=re.IGNORECASE)
    if '131' in s or '128' in s or 'titan' in s: return '131K'
    if '65' in s or '64' in s: return '64K'
    if '32' in s: return '32K'
    return 'Unknown'

def get_run(f, dt):
    f = str(f).lower()
    if 'run3' in f or 'recovery' in f: return 3
    if 'run2' in f: return 2
    if 'run1' in f: return 1
    
    fname = os.path.basename(f)
    m = re.search(r'202[6-9](\d{2})(\d{2})_(\d{2})(\d{2})', fname)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        if month == 7 and day >= 9: return 3
        if month == 7 and day in [7, 8]: return 2
    else:
        if dt.year == 2026:
            if dt.month == 7 and dt.day >= 9: return 3
            if dt.month == 7 and dt.day in [7, 8]: return 2
    return 1

# Hardwired Colab files
COLAB_FLASH_FILES = [
    "TimeBench_LOTSA_TEMPO_TSFM_Kag_20260702_1433_ConsoleLog.txt",
    "UTSD_20260702_0226_ConsoleLog.txt",
    "UTSD_TimeBench_LOTSA_TEMPO_TSF_20260625_2056_ConsoleLog.txt",
    "AEON_M6_LOTSA_20260530_2134_ConsoleLog.txt",
    "AEON_M6_LOTSA_20260615_2237_ConsoleLog.txt",
    "M6_Tempo_LOTSA_Gemma4_20260617_1406_ConsoleLog.txt",
    "M6_Tempo_LOTSA_Cloud_20260617_1518_ConsoleLog.txt",
    "M6_Tempo_LOTSA_20260617_1815_ConsoleLog.txt",
    "M6_Tempo_LOTSA_20260617_1852_ConsoleLog.txt",
    "M6_Tempo_LOTSA_20260618_0935_ConsoleLog.txt",
    "M6_Tempo_LOTSA_20260623_1732_ConsoleLog.txt",
    "M6_Tempo_LOTSA_20260623_1848_ConsoleLog.txt",
    "M6_Tempo_LOTSA_20260623_2104_ConsoleLog.txt"
]
COLAB_PRO_FILES = [
    "Bench_UTSD_Gemini-Cloud-PRO-Monolithic_20260702_0431.csv",
    "Bench_TIMEBENCH_Gemini-Cloud-PRO-Monolithic_20260702_1522.csv",
    "Bench_LOTSA_Gemini-Cloud-PRO-Monolithic_20260702_1823.csv",
    "Bench_TEMPO_Gemini-Cloud-PRO-Monolithic_20260702_1951.csv",
    "Bench_TSFM_Gemini-Cloud-PRO-Monolithic_20260702_2121.csv",
    "Bench_KAGGLETS_Gemini-Cloud-PRO-Monolithic_20260702_2246.csv",
    "Bench_M2_Gemini-Cloud-PRO-Monolithic_20260702_2342.csv",
    "Bench_M6_Gemini-Cloud-PRO-Monolithic_20260703_0019.csv"
]
COLAB_JUNK = [
    "UTSD_20260702_0115_ConsoleLog.txt",
    "UTSD_20260702_0011_ConsoleLog.txt"
]

def is_colab(fname):
    if fname in COLAB_FLASH_FILES or fname in COLAB_PRO_FILES: return True
    fl = fname.lower()
    if 'cloud' in fl or 'cascade' in fl or 'monolithic' in fl or 'consolelog' in fl: return True
    return False

def is_admin_file(fname):
    """Heuristic to catch pure administrative/system files and ignore them"""
    if is_colab(fname): return False
    
    # If the file does not have a Project Name in it, and isn't named "Bench_", it's likely admin
    if get_project(fname) == 'Unknown' and not fname.startswith('Bench_'):
        return True
        
    admin_terms = [
        'chronological', 'adjudicated', 'actionable', 'inventory', 'leaderboard', 
        'goldenkb', 'oracle', 'run_missing_jobs', 'pdfgems', 'matrix_auditor', 
        'reorganize_files', 'gatherout', 'plot', 'agent_log', 'full_context', 
        'benchmark_terminal', 'master_suite_progress', 'licences', 'requirements', 
        'token', 'credentials', 'full_codebase', 'deepcollector_context',
        'master_gauntlet', 'master_ablation', 'qwen_master', 'gemma_master',
        'vram_time_series.csv', 'audit_and_group', 'test_drive'
    ]
    
    fl = fname.lower()
    if fl.endswith('.py') or fl.endswith('.sh') or fl.endswith('.json') or fl.endswith('.md'): return True
    if any(term in fl for term in admin_terms): return True
    return False

def parse_file(f):
    fname = os.path.basename(f)
    if fname in COLAB_JUNK: return []
    
    mtime = os.path.getmtime(f)
    m = re.search(r'(202[6-9])(\d{2})(\d{2})_(\d{2})(\d{2})', fname)
    if m:
        try: dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)))
        except: dt = datetime.fromtimestamp(mtime)
    else:
        dt = datetime.fromtimestamp(mtime)

    # 1. Hardcoded Colab Files
    if fname in COLAB_FLASH_FILES or (is_colab(fname) and fname.endswith('.txt')):
        content = ""
        try:
            with open(f, 'r', encoding='utf-8', errors='ignore') as logfile: content = logfile.read(250000)
        except: pass
        
        projs = set()
        for pm in re.finditer(r'(?:Processing Project|STARTING PROJECT|Project|STARTING:.*?\|\s*Project):\s*([A-Za-z0-9_]+)', content, re.IGNORECASE):
            pr = get_project(pm.group(1))
            if pr != 'Unknown': projs.add(pr)
            
        if not projs:
            pr = get_project(fname)
            if pr != 'Unknown': projs.add(pr)
            else:
                if 'TimeBench_LOTSA' in fname or 'M6_Tempo' in fname or 'AEON' in fname:
                    projs = ['TIMEBENCH', 'LOTSA', 'TEMPO', 'TSFM', 'KAGGLETS', 'M2', 'M6']

        run_id = get_run(fname, dt)
        return [{
            "Datetime": dt, "File": fname, "Path": f, "Type": "TXT-Log", "Project": p,
            "Model": "Cloud-Flash", "Context": "Cloud", "Run": run_id, "Method": 1,
            "Job": f"Cloud-Flash [Cloud] on {p} (Run {run_id})"
        } for p in projs]
        
    if fname in COLAB_PRO_FILES:
        proj = get_project(fname)
        run_id = get_run(fname, dt)
        return [{
            "Datetime": dt, "File": fname, "Path": f, "Type": "Data CSV", "Project": proj,
            "Model": "Cloud-Pro", "Context": "Cloud", "Run": run_id, "Method": 1,
            "Job": f"Cloud-Pro [Cloud] on {proj} (Run {run_id})"
        }]

    # 2. General parsing for DGX files
    model = get_model(fname)
    run = get_run(f, dt)
    context = get_context(fname, model)
    project = get_project(fname)
    
    file_type = "Log" if f.endswith(('.log', '.txt')) else "Data CSV"
    if 'vram' in fname.lower(): file_type = "VRAM CSV"
    elif 'checkpoint' in fname.lower(): file_type = "Checkpoint CSV"
    
    method = 1 if 'cloud' in model.lower() else 'Unknown'
    
    try:
        if file_type == "Data CSV" or file_type == "Checkpoint CSV":
            df = pd.read_csv(f, nrows=5)
            cols = [str(c).lower().strip() for c in df.columns]
            if project == 'Unknown' and 'project' in cols: project = get_project(str(df.iloc[0, cols.index('project')]))
            if model == 'Unknown':
                if 'benchmark_model' in cols: model = get_model(str(df.iloc[0, cols.index('benchmark_model')]))
                elif 'model' in cols: model = get_model(str(df.iloc[0, cols.index('model')]))
            if context == 'Unknown':
                for c in cols:
                    if 'context' in c or 'context_length' in c:
                        cval = get_context(str(df.iloc[0, cols.index(c)]), model)
                        if cval != 'Unknown': context = cval; break
                        
        elif file_type == "Log":
            with open(f, 'r', encoding='utf-8', errors='ignore') as logfile:
                content = logfile.read(50000)
            if project == 'Unknown':
                pm = re.search(r'(?:Processing Project|STARTING PROJECT|Project|STARTING:.*?\|\s*Project):\s*([A-Za-z0-9_]+)', content, re.IGNORECASE)
                if pm: project = get_project(pm.group(1))
            if model == 'Unknown':
                mm = re.search(r'(?:STARTING:\s*|Model:\s*|Starting vLLM \()([A-Za-z0-9\-\.]+)', content, re.IGNORECASE)
                if mm: model = get_model(mm.group(1))
            if context == 'Unknown':
                cm = re.search(r'Context(?:_Length|\s*Size|\s*=\s*|\s*:\s*)(\d+)', content, re.IGNORECASE)
                if cm: context = get_context(cm.group(1), model)
            
            if 'Method: 2' in content or 'HARVEST' in content.upper(): method = 2
            elif 'Method: 1' in content or 'AGENT' in content.upper(): method = 1
    except: pass

    if 'cloud' in model.lower():
        context = 'Cloud'
        method = 1
        
    if context == 'Unknown' and 'cloud' not in model.lower():
        if 'titan' in f.lower() or '128k' in f.lower() or '131k' in f.lower(): context = '131K'
        else: context = '32K' if run == 1 else '64K'

    job_id = f"{model} [{context}] on {project} (Run {run})"

    return [{
        "Datetime": dt,
        "File": fname,
        "Path": f,
        "Type": file_type,
        "Job": job_id,
        "Project": project,
        "Model": model,
        "Context": context,
        "Run": run,
        "Method": method
    }]

def write_grouped_sheet(records, out_path):
    if not records: return pd.DataFrame()
    df = pd.DataFrame(records)
    
    # 1. Sort purely by time first
    df = df.sort_values(by="Datetime")
    
    jobs = []
    # 2. Group by Model, Context, Project, Run
    for key, group in df.groupby(['Model', 'Context', 'Project', 'Run']):
        group = group.sort_values(by='Datetime')
        
        current_job_files = []
        current_job_start = None
        
        for idx, row in group.iterrows():
            if current_job_start is None:
                current_job_start = row['Datetime']
                current_job_files.append(row)
            else:
                # 4-hour grouping window (14400 seconds)
                if (row['Datetime'] - current_job_start).total_seconds() <= 14400: 
                    current_job_files.append(row)
                else:
                    jobs.append(current_job_files)
                    current_job_files = [row]
                    current_job_start = row['Datetime']
                    
        if current_job_files:
            jobs.append(current_job_files)

    # 3. Sort Jobs Chronologically
    jobs.sort(key=lambda x: x[0]['Datetime'])

    output_rows = []
    cols = ['Datetime', 'File', 'Type', 'Job', 'Project', 'Model', 'Context', 'Run', 'Method', 'Status']
    
    for job_files in jobs:
        first = job_files[0]
        types = [f['Type'] for f in job_files]
        job_str = f"{first['Model']} [{first['Context']}] on {first['Project']} (Run {first['Run']})"
        
        method = 1 if 'Cloud' in first['Model'] else 'Unknown'
        for f in job_files:
            if f['Method'] != 'Unknown': method = f['Method']; break
                
        is_bad_m2 = False
        if first['Project'] == 'M2' and 'Cloud' not in first['Model']:
            if method == 2 or (method == 'Unknown' and first['Run'] < 3):
                is_bad_m2 = True

        has_data = any(t in ['Data CSV', 'TXT-Data', 'TXT-Log'] for t in types)

        if is_bad_m2: status = "🛑 Bad M2 (Method 2 Bug)"
        elif first['Model'] == 'Unknown' or first['Project'] == 'Unknown' or first['Context'] == 'Unknown': status = "❌ Unclassifiable"
        elif has_data: status = "✅ Complete"
        else: status = "⚠️ Incomplete (No CSV)"
            
        for f in job_files:
            row_dict = {
                'Datetime': f['Datetime'].strftime("%Y-%m-%d %H:%M:%S") if pd.notna(f['Datetime']) else "",
                'File': f['File'],
                'Type': f['Type'],
                'Job': job_str,
                'Project': f['Project'],
                'Model': f['Model'],
                'Context': f['Context'],
                'Run': f['Run'],
                'Method': method,
                'Status': status
            }
            output_rows.append(row_dict)
            
        # Insert a blank line separating this Job block from the next
        output_rows.append({k: "" for k in cols})

    pd.DataFrame(output_rows, columns=cols).to_csv(out_path, index=False)
    print(f"✅ Created {out_path} (One Line Per File, Job Blocks Separated by Blank Line, Sorted Chronologically)")
    
    return pd.DataFrame([r for r in output_rows if r.get('Status') == '✅ Complete'])

def main():
    print("=========================================================================================")
    print("🚀 Reorganizing ALL Files into ALL_Colab, ALL_DGX, and Admin directories...")
    print("=========================================================================================\n")
    
    all_files = glob.glob(os.path.join(ROOT_DIR, "**", "*.*"), recursive=True)
    
    # Do not process anything already inside /Trash_Bin/ or our new structural folders if they are nested deep
    valid_paths = [f for f in all_files if '/Trash_Bin/' not in f.replace('\\', '/') and '/.git/' not in f.replace('\\', '/') and '/venv/' not in f.replace('\\', '/')]
    
    colab_files = []
    dgx_files = []
    admin_files = []
    
    for f in valid_paths:
        fname = os.path.basename(f)
        
        # Trash known Colab junk
        if "UTSD_20260702_0115" in fname or "UTSD_20260702_0011" in fname:
            continue
            
        # Move Admin files to Admin directory
        if is_admin_file(fname):
            dest = os.path.join(ADMIN_DIR, fname)
            if f != dest:
                try: shutil.move(f, dest); admin_files.append(dest)
                except: pass
            continue
            
        # If it's a data file, push to Colab or DGX
        if f.lower().endswith(('.csv', '.log', '.txt')):
            if is_colab(fname) or 'ALL_Colab' in f:
                dest = os.path.join(COLAB_DIR, fname)
                if f != dest:
                    try: shutil.move(f, dest)
                    except: dest = f
                colab_files.append(dest)
            else:
                dest = os.path.join(DGX_DIR, fname)
                if f != dest:
                    try: shutil.move(f, dest)
                    except: dest = f
                dgx_files.append(dest)

    print(f"Moved {len(admin_files)} system/admin files to Admin/")
    print(f"Moved/Found {len(colab_files)} files in ALL_Colab")
    print(f"Moved/Found {len(dgx_files)} files in ALL_DGX")

    colab_records = []
    for f in colab_files: colab_records.extend(parse_file(f))
        
    dgx_records = []
    for f in dgx_files: dgx_records.extend(parse_file(f))

    valid_colab_df = write_grouped_sheet(colab_records, os.path.join(ROOT_DIR, "Colab_Files_Grouped.csv"))
    valid_dgx_df = write_grouped_sheet(dgx_records, os.path.join(ROOT_DIR, "DGX_Files_Grouped.csv"))

    print("\n📊 DGX LOCAL MODELS COMPLETION MATRIX (Based strictly on '✅ Complete' Jobs):")
    if not valid_dgx_df.empty:
        valid_dgx_df = valid_dgx_df[valid_dgx_df['Model'] != '']
        
        matrix = valid_dgx_df.groupby(['Model', 'Context', 'Project'])['Run'].nunique().reset_index(name='Completed_Runs')
        pivot = matrix.pivot(index=['Model', 'Context'], columns='Project', values='Completed_Runs').fillna(0).astype(int)
        
        for p in PROJECTS:
            if p not in pivot.columns: pivot[p] = 0
        pivot = pivot[PROJECTS]
        
        print(tabulate(pivot, headers="keys", tablefmt="pipe"))
        
        missing_jobs = []
        for m in MODELS:
            for c in CONTEXTS:
                for p in PROJECTS:
                    try: completed = pivot.loc[(m, c), p]
                    except KeyError: completed = 0
                    if 3 - completed > 0:
                        missing_jobs.append({"Model": m, "Context": c, "Project": p, "Needed": 3 - completed})
                        
        if not missing_jobs:
            print("\n🎉 ALL LOCAL MODELS COMPLETE! No missing jobs.")
        else:
            total_needed = sum(j['Needed'] for j in missing_jobs)
            print(f"\n🚨 MISSING RUNS HIT-LIST ({total_needed} total executions needed):")
            for job in missing_jobs:
                if job['Needed'] < 3:
                    print(f"  -> {job['Model']} [{job['Context']}] on {job['Project']}: Completed {3 - job['Needed']}/3 runs.")
            
            script_path = os.path.join(ROOT_DIR, "run_missing_jobs.sh")
            with open(script_path, "w") as f:
                f.write("#!/bin/bash\nROOT_DIR=\"$HOME/Desktop/DeepKG\"\nLAUNCH_DIR=\"$ROOT_DIR/deepcollector/localdgxfiles\"\n\n")
                f.write('''run_job() {
    local RUN_ID=$1; local MODEL=$2; local CONTEXT=$3; local CONCURRENCY=$4; local PROJECT=$5
    if [ "$MODEL" == "deepseek" ]; then export DC_TEMP="0.6"; export DC_TOKENS="8192"
    else export DC_TEMP="0.0"; export DC_TOKENS="4096"; fi
    export DC_FILE_PREFIX="${RUN_ID}_${MODEL}_${PROJECT}_Temp${DC_TEMP}_Ctx${CONTEXT}"
    local LOG_FILE="$ROOT_DIR/${DC_FILE_PREFIX}_Console.log"
    local VRAM_FILE="$ROOT_DIR/${DC_FILE_PREFIX}_VRAM.csv"
    echo -e "\\n--------------------------------------------------------"
    echo "⚡ STARTING: $MODEL | Context: $CONTEXT | Concurrency: $CONCURRENCY | Project: $PROJECT"
    echo "--------------------------------------------------------"
    docker rm -f vllm_engine >/dev/null 2>&1; sleep 5
    cd "$LAUNCH_DIR" || exit 1
    ./vram_monitor.sh "$VRAM_FILE" > /dev/null 2>&1 &
    VRAM_PID=$!
    disown $VRAM_PID
    ./start.sh "$MODEL" "$CONTEXT" "$CONCURRENCY" "$PROJECT" > "$LOG_FILE" 2>&1
    EXIT_CODE=$?
    kill $VRAM_PID 2>/dev/null; wait $VRAM_PID 2>/dev/null
    pkill -9 -P $VRAM_PID 2>/dev/null; pkill -9 -f "nvidia-smi" 2>/dev/null
    cd "$ROOT_DIR" || exit 1
    if [ $EXIT_CODE -ne 0 ]; then
        echo "❌ ERROR: start.sh failed for $MODEL $PROJECT."
        exit 1
    fi
    sleep 10
}\n\n''')
                f.write('echo "🚀 STARTING MISSING JOBS COMPLETION SUITE..."\n')
                model_map = {'Gemma': 'gemma4', 'Qwen': 'qwen', 'DeepSeek': 'deepseek'}
                ctx_map = {'32K': ('32768', '32'), '64K': ('65536', '8'), '131K': ('131072', '1')}
                missing_jobs.sort(key=lambda x: (x['Context'], x['Model'], x['Project']))
                for job in missing_jobs:
                    m_sh = model_map[job['Model']]
                    c_tok, conc = ctx_map[job['Context']]
                    for i in range(1, job['Needed'] + 1):
                        f.write(f'run_job "Run_Recovery_{i}_20260713" "{m_sh}" "{c_tok}" "{conc}" "{job["Project"]}"\n')
                f.write('echo "✅ ALL MISSING JOBS COMPLETE!"\n')
            os.system(f"chmod +x {script_path}")
            print(f"\n✅ Recovery script generated: {script_path}")

if __name__ == "__main__":
    main()
