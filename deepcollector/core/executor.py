# =============================================================================
# V1.0: Unified Executor Module
# =============================================================================
import os
import sys
import time
import warnings
import uuid
import copy
import re
import pandas as pd
from datetime import datetime
from tabulate import tabulate
from collections import Counter

from deepcollector.core.agent import CatalogAgent
from deepcollector.utils.initialization import initialize_apis, configure_llama_index
from deepcollector.kb.manager import KnowledgeBaseManager
from deepcollector.kb.quality import QualityAuditor
from deepcollector.utils.project_loader import ProjectLoader, ExternalKnowledge
from deepcollector.utils.profiler import profiler

# -----------------------------------------------------------------------------
# ??? CONSOLE LOGGER
# -----------------------------------------------------------------------------
class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log_file = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()

# -----------------------------------------------------------------------------
# ?? PLANNER LOGIC
# -----------------------------------------------------------------------------
def run_planner(kb_mgr):
    print("\n" + "="*80)
    print("?? DEEPCOLLECTOR JOB PLANNER & DIAGNOSTICS")
    print("="*80)

    df_jobs = kb_mgr.get_kb_data("Jobs")
    df_ds = kb_mgr.get_kb_data("Datasets")
    df_links = kb_mgr.get_kb_data("Project_Dataset_Link")

    if not df_jobs.empty and 'Start_Time' in df_jobs.columns:
        df_jobs['Start_Time_DT'] = pd.to_datetime(df_jobs['Start_Time'], errors='coerce')
    else:
        df_jobs['Start_Time_DT'] = pd.NaT

    completed_jobs = df_jobs[df_jobs['Status'].str.contains('COMPLETED', na=False)] if not df_jobs.empty else pd.DataFrame()

    merge_targets = []
    if not completed_jobs.empty:
        for pid in completed_jobs['ProjectID'].dropna().unique():
            if pid in ["GLOBAL", "PROJ_UNKNOWN", "PROJ_LOST", "PROJ_GLOBAL_MAINTENANCE"]: continue
            proj_jobs = completed_jobs[completed_jobs['ProjectID'] == pid]
            agent_jobs = proj_jobs[proj_jobs['Mode'].isin(['AGENT', 'HARVEST'])]
            merge_jobs = proj_jobs[proj_jobs['Mode'] == 'MERGE']
            agent_count = len(agent_jobs)
            if agent_count > 1:
                last_agent_time = agent_jobs['Start_Time_DT'].max()
                last_merge_time = merge_jobs['Start_Time_DT'].max() if not merge_jobs.empty else pd.Timestamp.min
                if pd.isna(last_merge_time) or last_agent_time > last_merge_time:
                    merge_targets.append((pid, agent_count))

    if merge_targets:
        print("\n??? STEP 1: PROJECT-LEVEL MERGES")
        merge_table = [{"Set ProjectName": str(pid).replace('PROJ_', ''), "Set MODE": "MERGE", "Reason": f"{count} overlapping runs"} for pid, count in merge_targets]
        print(tabulate(merge_table, headers="keys", tablefmt="simple"))
    else:
        print("\n? STEP 1: No project-level MERGEs required.")

    repair_targets = []
    if not df_ds.empty and not df_links.empty:
        ds_to_proj = dict(zip(df_links['DatasetID'], df_links['ProjectID']))
        df_ds['Assigned_Project'] = df_ds['DatasetID'].map(ds_to_proj)
        missing_markers = {"", "[missing]", "nan", "none"}
        proj_repair_needs = Counter()
        proj_totals = Counter()

        fields_to_check = ["Primary URL", "Link to Data (Actual Source)", "Other URL", "Num Time Points", "Total Variables"]

        for _, row in df_ds.iterrows():
            pid = row.get('Assigned_Project', 'Unknown')
            if pd.isna(pid) or pid in ["GLOBAL", "PROJ_UNKNOWN", "PROJ_LOST", "PROJ_GLOBAL_MAINTENANCE", "Unknown"]: continue
            proj_totals[pid] += 1
            needs_repair = False
            for field in fields_to_check:
                val = str(row.get(field, "")).strip().lower()
                if val in missing_markers:
                    needs_repair = True; break
                conf_col = f"{field} (C)"
                try:
                    if float(row.get(conf_col, 0.0)) < 0.80: needs_repair = True; break
                except: pass
            if needs_repair: proj_repair_needs[pid] += 1

        for pid, need_count in proj_repair_needs.items():
            total = proj_totals[pid]
            pct = (need_count / total) * 100
            if pct > 30 and total > 2:
                proj_jobs = completed_jobs[completed_jobs['ProjectID'] == pid] if not completed_jobs.empty else pd.DataFrame()
                repair_jobs = proj_jobs[proj_jobs['Mode'] == 'REPAIR']
                other_jobs = proj_jobs[proj_jobs['Mode'].isin(['AGENT', 'HARVEST', 'MERGE'])]
                last_repair_time = repair_jobs['Start_Time_DT'].max() if not repair_jobs.empty else pd.Timestamp.min
                last_other_time = other_jobs['Start_Time_DT'].max() if not other_jobs.empty else pd.Timestamp.min

                if pd.isna(last_repair_time) or last_repair_time < last_other_time:
                    repair_targets.append((pid, need_count, total, pct))

    if repair_targets:
        print("\n??? STEP 2: PROJECT-LEVEL REPAIRS")
        repair_targets.sort(key=lambda x: x[3], reverse=True)
        repair_table = [{"Set ProjectName": str(pid).replace('PROJ_', ''), "Set MODE": "REPAIR", "Reason": f"{need}/{tot} datasets ({pct:.1f}%) need fixes"} for pid, need, tot, pct in repair_targets]
        print(tabulate(repair_table, headers="keys", tablefmt="simple"))
    else:
        print("\n? STEP 2: No project-level REPAIRs required.")

    print("\n?? STEP 3: GLOBAL ACTIONS")
    global_table = [
        {"Order": "1st", "Set ProjectName": "GLOBAL", "Set MODE": "MERGE", "Description": "Cross-project deduplication."},
        {"Order": "2nd", "Set ProjectName": "GLOBAL", "Set MODE": "REVIEW", "Description": "Quality audit & move flaky datasets to Quarantine."}
    ]
    print(tabulate(global_table, headers="keys", tablefmt="simple"))

    try:
        ek = ExternalKnowledge(kb_mgr.config, kb_mgr.gc, verbosity=0)
        ek.load()
        if ek.projects:
            completed_pids = set(completed_jobs['ProjectID'].dropna().unique()) if not completed_jobs.empty else set()
            missing_projects = []

            for p in ek.projects:
                pid_raw = ""
                for k, v in p.items():
                    if any(term in str(k).lower() for term in ["canonical", "projectid", "id"]):
                        val = str(v).strip()
                        m = re.search(r'=HYPERLINK\([^,]+,\s*"([^"]+)"\)', val, re.IGNORECASE)
                        if m: val = m.group(1)
                        pid_raw = val.upper()
                        break

                if not pid_raw and p:
                    val = str(list(p.values())).strip()
                    m = re.search(r'=HYPERLINK\([^,]+,\s*"([^"]+)"\)', val, re.IGNORECASE)
                    if m: val = m.group(1)
                    pid_raw = val.upper()

                pid = f"PROJ_{pid_raw}" if pid_raw and not pid_raw.startswith("PROJ_") else pid_raw

                method_str = ""
                for k, v in p.items():
                    if "method" in str(k).lower():
                        method_str = str(v)
                        break
                try:
                    clean_val = re.sub(r'[^0-9-]', '', method_str)
                    method_val = int(clean_val) if clean_val else 1
                except ValueError:
                    method_val = 1

                if method_val == -1: continue

                if pid not in completed_pids and pid != "PROJ_" and pid_raw != "LOST" and pid != "PROJ_LOST":
                    missing_projects.append(pid.replace('PROJ_', ''))

            if missing_projects:
                print("\n?? STEP 4: UNRUN PROJECTS (Found in Master Sheet but not in KB)")
                print(f"    You have {len(missing_projects)} active projects left to run.")
                print("    Next 5 suggestions: " + ", ".join(missing_projects[:5]))
            else:
                print("\n? STEP 4: All active projects from the Master Sheet have been run!")

    except Exception as e:
        print(f"    (Could not check Master Sheet for unrun projects: {e})")

    print("="*80 + "\n")

# -----------------------------------------------------------------------------
# ?? CORE EXECUTOR FUNCTION
# -----------------------------------------------------------------------------
def execute_jobs(mode: str, project_names: list, base_config, gc_client=None, dry_run=False):
    warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")
    warnings.filterwarnings("ignore", category=UserWarning, module="google.generativeai")

    # Setup Logging
    run_timestamp = time.strftime('%Y%m%d_%H%M')
    proj_str = "_".join(project_names)[:30]
    log_filename = f"{proj_str}_{run_timestamp}_ConsoleLog.txt"

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeLogger(log_filename)
    sys.stderr = sys.stdout

    try:
        print(f"🚀 Initializing Executor (Mode: {mode})...")
        profiler.reset()

        if mode == "PLAN":
            kb_manager = KnowledgeBaseManager(base_config)
            kb_manager.initialize_connection(gc_client)
            if kb_manager.read_and_validate_kb(): run_planner(kb_manager)
            else: print("    ? Failed to load Knowledge Base.")

        elif mode == "REVIEW":
            keys, models = initialize_apis(base_config)
            kb_manager = KnowledgeBaseManager(base_config)
            kb_manager.initialize_connection(gc_client)
            if kb_manager.read_and_validate_kb():
                job_id = f"JOB_{uuid.uuid4().hex[:6].upper()}"
                auditor = QualityAuditor(kb_manager, models=models, llm_limit=base_config.LLM_ARBITRATION_LIMIT)
                auditor.run_audit_and_clean(run_quarantine=True, dry_run=dry_run, job_id=job_id, project_id="PROJ_GLOBAL_MAINTENANCE", job_comment="Global Review")

        elif mode in ["HARVEST", "AGENT", "REPAIR", "MERGE"]:
            keys, models = initialize_apis(base_config)
            if mode != "MERGE": configure_llama_index(base_config, models, keys)

            total_projects = len(project_names)
            for i, p_name in enumerate(project_names):
                print(f"\n{'='*70}")
                print(f"🚀 BATCH PROGRESS: [{i+1}/{total_projects}] Processing Project: {p_name}")
                print(f"{'='*70}")

                current_job_id = f"JOB_{uuid.uuid4().hex[:6].upper()}"
                current_project_id = "PROJ_GLOBAL_MAINTENANCE" if p_name == "GLOBAL" else f"PROJ_{p_name.upper()}"
                job_comment = f"{p_name} Batch {mode} execution."

                current_config = copy.deepcopy(base_config)
                current_config.CURRENT_PROJECT_ID = current_project_id
                current_config.CURRENT_PROJECT_NAME = p_name

                run_mode = mode

                if p_name != "GLOBAL":
                    print(f"🔄 Loading Project Context for {p_name}...")
                    loader = ProjectLoader(current_config, gc_client, verbosity=current_config.VERBOSITY_LEVEL)

                    if loader.resolve_configuration():
                        print(f"    ✅ Context Loaded: '{current_config.PROJECT_CONTEXT[:80]}...'")
                        if current_config.PROJECT_METHOD == 2 and run_mode == "AGENT":
                            print(f"    ??  Project defines Method 2 in Master Sheet. Switching to HARVEST mode.")
                            run_mode = "HARVEST"
                    else:
                        print(f"    ?? Skipping '{p_name}': Invalid Project Configuration or missing from Master Sheet.")
                        continue

                try:
                    agent = CatalogAgent(current_config, gc_client, keys=keys, models=models)
                    agent.job_id = current_job_id
                    if p_name == "GLOBAL": agent.config.CURRENT_PROJECT_ID = "GLOBAL"
                    agent.execute_workflow(mode=run_mode, job_comment=job_comment, merge_dry_run=dry_run)
                    print(f"\n🎉 Successfully finished [{i+1}/{total_projects}]: {p_name}")
                except Exception as e:
                    print(f"\n? Error processing {p_name}: {e}")
                    print("?? Catching error and safely continuing to next project in queue...")

                if i < total_projects - 1:
                    print("? Pausing for 5 seconds before next project...")
                    time.sleep(5)

            print(f"\n{'='*70}\n🏁 BATCH PROCESSING COMPLETE. Processed {total_projects} items.\n{'='*70}")

        if mode != "PLAN":
            print("\n" + "="*40)
            print("⏱️ OVERALL PROFILING & HEALTH REPORT")
            print("="*40)
            try:
                df_profile = profiler.get_report()
                if not df_profile.empty: print(tabulate(df_profile, headers='keys', tablefmt='pipe', showindex=False))
            except: pass

    finally:
        # Restore sys.stdout safely before doing Drive Uploads
        sys.stdout.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        # Upload Log to Google Drive
        log_folder_id = getattr(base_config, 'GOOGLE_DRIVE_LOG_FOLDER_ID', None)
        if log_folder_id:
            print(f"\n☁️ Uploading Console Log ({log_filename}) to Google Drive...")
            try:
                from googleapiclient.discovery import build
                from google.auth import default
                from googleapiclient.http import MediaFileUpload

                creds, _ = default()
                drive_service = build('drive', 'v3', credentials=creds)

                file_metadata = {'name': log_filename, 'parents': [log_folder_id]}
                media = MediaFileUpload(log_filename, mimetype='text/plain')
                file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
                print(f"✅ Success! Log File uploaded to Drive with ID: {file.get('id')}")
            except ImportError:
                print("?? googleapiclient not installed. Skipping Drive upload.")
            except Exception as e:
                print(f"? Failed to upload log file to Google Drive: {e}")