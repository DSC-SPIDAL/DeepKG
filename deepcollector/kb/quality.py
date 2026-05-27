# =============================================================================
# V59.9: Quality Auditor (Description Bloat Fix)
# =============================================================================
import pandas as pd
import uuid
import time
import re
from datetime import datetime
from collections import defaultdict, Counter
from deepcollector.kb.merger import UniversalOracle

try:
    from deepcollector.utils.profiler import profiler
except ImportError:
    class DummyProfiler:
        def track(self, c): return lambda f: f
    profiler = DummyProfiler()

class QualityAuditor:
    def __init__(self, kb_manager, models=None, llm_limit=None):
        self.kb_manager = kb_manager
        self.models = models
        self.verbosity = getattr(kb_manager.config, 'VERBOSITY_LEVEL', 1)

        self.oracle = UniversalOracle(models=models, verbosity=self.verbosity)
        if llm_limit is not None:
            self.oracle.llm_limit = llm_limit
        else:
            self.oracle.llm_limit = getattr(kb_manager.config, 'LLM_ARBITRATION_LIMIT', 150)

        self.CONFIDENCE_THRESHOLD = 0.80
        self.FLAKY_THRESHOLD = 0.50

        self.project_stats = defaultdict(lambda: {
            'total': 0, 'low_conf': 0, 'dubious': 0, 'norm_error': 0,
            'missing_counts': Counter(), 'missing_time_points_list': [], 'norm_error_list': []
        })

        self.COL_DATASET_ID = "DatasetID"
        self.COL_PROJECT_ID = "ProjectID"
        self.COL_URL = "Primary URL"
        self.COL_DATA_URL = "Link to Data (Actual Source)"
        self.COL_OTHER_URL = "Other URL"
        self.COL_TYPE = "Type"
        self.EMPTY_MARKERS = self.oracle.MISSING

    def _get_display_name(self, row):
        did = str(row.get(self.COL_DATASET_ID, "UnknownID"))
        eff_name = self.oracle._clean_name(row.get("Variant Name", "") or row.get("Canonical Name", ""))
        return f"{did} ({eff_name})"

    @profiler.track("Quality: Full Audit & Relink")
    def run_audit_and_clean(self, run_quarantine=True, dry_run=False, job_id=None, project_id="GLOBAL", job_comment=""):
        start_time = datetime.now()
        job_id = job_id or f"JOB_{uuid.uuid4().hex[:6].upper()}"

        mode_str = "🚫 DRY RUN" if dry_run else "💾 LIVE"
        print("\n" + "="*60)
        print(f"🔍 STARTING DATA QUALITY AUDIT & RELINKING [{mode_str}] (V59.9)")
        print(f"🚀 Job ID: {job_id}")
        if job_comment:
            print(f"📝 Job Comment: {job_comment}")
        print("="*60)

        from deepcollector.kb.manager import SheetLock
        lock = SheetLock(self.kb_manager, job_id, self.verbosity)
        if not lock.acquire(timeout_seconds=7200):
            print("    ❌ Failed to acquire lock for Quality Audit.")
            return

        try:
            if not self.kb_manager.read_and_validate_kb():
                print("    ❌ Failed to load Knowledge Base.")
                return

            df = self.kb_manager.get_kb_data("Datasets")
            if df.empty:
                print("    ⚠️ Datasets tab is empty. Aborting.")
                return

            print("    🔗 Resolving Project Links...")
            df_links = self.kb_manager.get_kb_data("Project_Dataset_Link")
            if df_links.empty or self.COL_DATASET_ID not in df_links.columns:
                print("    🛑 ABORT: Linking table missing/empty.")
                return

            if run_quarantine:
                df, df_links = self._quarantine_low_confidence(df, df_links, dry_run=dry_run)

            ds_to_proj = dict(zip(df_links[self.COL_DATASET_ID], df_links[self.COL_PROJECT_ID]))
            df['Project_ID'] = df[self.COL_DATASET_ID].map(ds_to_proj).fillna('Unknown')

            if self.models:
                print(f"\n    🧠 Universal Oracle Arbitration: ENABLED (Limit: {self.oracle.llm_limit} calls)")
            else:
                print(f"\n    ⚪ Universal Oracle Arbitration: DISABLED")

            print(f"\n🧠 Running Universal Oracle Deduplication...")

            df_clean, dupes_log = self._advanced_deduplication(df)
            duplicates_removed = len(df) - len(df_clean)

            for pid in df_clean['Project_ID'].unique():
                self.project_stats[str(pid)]['total'] = len(df_clean[df_clean['Project_ID'] == pid])

            print("\n🤖 ORACLE USAGE SUMMARY:")
            print(f"   Total LLM Calls: {self.oracle.stats['llm_calls']} / {self.oracle.llm_limit}")
            print(f"   ✅ Merges Approved: {self.oracle.stats['approved']}")
            print(f"   ❌ Merges Rejected: {self.oracle.stats['rejected']}")
            if self.oracle.stats['errors'] > 0:
                print(f"   ⚠️ API Errors/Quota: {self.oracle.stats['errors']}")

            print("\n--- Deduplication Results ---")
            print(f"    Original Count (Post-Quarantine): {len(df)}")
            print(f"    Clean Count:    {len(df_clean)}")
            print(f"    Duplicates Removed: {duplicates_removed}")

            if not dupes_log.empty and not df_links.empty:
                print(f"\n🔗 RELINKING MODE: Reassigning {len(dupes_log)} merged datasets into 'Project_Dataset_Link' tab...")
                df_links_clean = df_links.copy()
                drop_to_keep = dict(zip(dupes_log['DatasetID (Dropped)'], dupes_log['DatasetID (Kept)']))
                before_len = len(df_links_clean)
                relinked_count = 0

                for idx, row in df_links_clean.iterrows():
                    ds_id = row.get(self.COL_DATASET_ID)
                    if ds_id in drop_to_keep:
                        curr_id = ds_id
                        visited = set()
                        while curr_id in drop_to_keep and curr_id not in visited:
                            visited.add(curr_id)
                            curr_id = drop_to_keep[curr_id]
                        df_links_clean.at[idx, self.COL_DATASET_ID] = curr_id

                        old_comments = str(row.get('Data Preparation Comments', ''))
                        relink_note = f"[RELINKED from {ds_id}]"
                        if old_comments and old_comments.lower() != "nan" and old_comments not in self.EMPTY_MARKERS:
                            if relink_note not in old_comments:
                                df_links_clean.at[idx, 'Data Preparation Comments'] = f"{old_comments}; {relink_note}"
                        else:
                            df_links_clean.at[idx, 'Data Preparation Comments'] = relink_note
                        relinked_count += 1

                df_links_clean = df_links_clean.drop_duplicates(subset=[self.COL_PROJECT_ID, self.COL_DATASET_ID])
                after_len = len(df_links_clean)
                print(f"    ✨ Relinking complete. Updated {relinked_count} rows and removed {before_len - after_len} redundant links.")

                if self.kb_manager.is_connected:
                    if dry_run:
                        print("    🚫 [DRY RUN] Skipping write to 'Project_Dataset_Link'.")
                    else:
                        print("    💾 Writing updated links to 'Project_Dataset_Link'...")
                        self.kb_manager.write_dataframe_to_tab("Project_Dataset_Link", df_links_clean)

            print("\n📊 Analyzing Final Catalog Quality...")
            quality_report, _ = self._analyze_quality(df_clean)
            self._print_quality_summary(quality_report, len(df_clean))
            self._print_project_health_report()

            if dry_run:
                print("\n🚫 DRY RUN COMPLETE. No changes were saved to Google Sheets.")
            else:
                if self.kb_manager.is_connected:
                    print("\n💾 Writing to 'Datasets'...")
                    self.kb_manager.write_dataframe_to_tab("Datasets", df_clean)

                    if not dupes_log.empty:
                        print(f"    💾 Writing {len(dupes_log)} records to 'Datasets_Duplicates_Log'...")
                        self.kb_manager.write_dataframe_to_tab("Datasets_Duplicates_Log", dupes_log)
                    print("    ✅ Cleanup & Relinking process complete.")

                duration = (datetime.now() - start_time).total_seconds()
                job_data = {
                    "JobID": job_id,
                    "ProjectID": project_id,
                    "Mode": "REVIEW",
                    "Start_Time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "End_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Duration_Sec": f"{duration:.2f}",
                    "Status": "COMPLETED",
                    "Items_Found": len(df_clean),
                    "Operational_Parameters": str({"Quarantine_Threshold": self.FLAKY_THRESHOLD}),
                    "JOB_COMMENT": job_comment
                }
                self.kb_manager.log_job_execution(job_data)

        except Exception as e:
            print(f"    ❌ Critical Error during quality audit: {e}")

        finally:
            lock.release()

    def _quarantine_low_confidence(self, df: pd.DataFrame, df_links: pd.DataFrame, dry_run: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        print(f"\n🩺 Running Flaky Dataset Quarantine (Threshold < {self.FLAKY_THRESHOLD})...")

        if 'Overall Confidence' not in df.columns:
            print("    ⚠️ 'Overall Confidence' column not found. Skipping quarantine.")
            return df, df_links

        df['_num_conf'] = pd.to_numeric(df['Overall Confidence'], errors='coerce').fillna(1.0)

        flaky_mask = df['_num_conf'] < self.FLAKY_THRESHOLD
        df_flaky = df[flaky_mask].copy()
        df_healthy = df[~flaky_mask].copy()

        df_healthy = df_healthy.drop(columns=['_num_conf'])
        df_flaky = df_flaky.drop(columns=['_num_conf'])

        if df_flaky.empty:
            print("    ✅ No flaky datasets found.")
            return df_healthy, df_links

        flaky_ids = df_flaky[self.COL_DATASET_ID].tolist()
        print(f"    ☢️ Found {len(flaky_ids)} datasets with explicitly low confidence.")

        initial_links = len(df_links)
        df_links_flaky = df_links[df_links[self.COL_DATASET_ID].isin(flaky_ids)].copy()
        df_links_healthy = df_links[~df_links[self.COL_DATASET_ID].isin(flaky_ids)].copy()

        if self.kb_manager.is_connected:
            if dry_run:
                print(f"    🚫 [DRY RUN] Would move {len(flaky_ids)} datasets to 'Datasets_Quarantined'.")
            else:
                print(f"    💾 Archiving flaky datasets to 'Datasets_Quarantined' tab...")
                try:
                    existing_quarantine = self.kb_manager.get_kb_data("Datasets_Quarantined")
                except Exception:
                    existing_quarantine = pd.DataFrame()

                if not existing_quarantine.empty:
                    df_flaky = pd.concat([existing_quarantine, df_flaky], ignore_index=True)
                    df_flaky = df_flaky.drop_duplicates(subset=[self.COL_DATASET_ID])
                self.kb_manager.write_dataframe_to_tab("Datasets_Quarantined", df_flaky)

                print(f"    💾 Preserving provenance in 'Links_Quarantined' tab...")
                try:
                    existing_q_links = self.kb_manager.get_kb_data("Links_Quarantined")
                except Exception:
                    existing_q_links = pd.DataFrame()

                if not existing_q_links.empty:
                    df_links_flaky = pd.concat([existing_q_links, df_links_flaky], ignore_index=True)
                    df_links_flaky = df_links_flaky.drop_duplicates(subset=[self.COL_PROJECT_ID, self.COL_DATASET_ID])
                self.kb_manager.write_dataframe_to_tab("Links_Quarantined", df_links_flaky)

        return df_healthy, df_links_healthy

    def _advanced_deduplication(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        merge_cols = [(c[:-4], c) for c in df.columns if c.endswith('(C)')]
        additive_cols = ["Description", "Assignment Rationale", "Project Citations", "Project WebLinks", "Primary URL", "Link to Data (Actual Source)", "Other URL"]
        av_sort = [c for c in ['Overall Confidence', 'Job_Updated'] if c in df.columns]
        df = df.sort_values(by=av_sort, ascending=[False]*len(av_sort))

        records = df.to_dict('records')
        kept_records = []
        duplicates_log = []
        skip_indices = set()

        for i, row in enumerate(records):
            if i % 20 == 0:
                print(f"    ⏳ Oracle Processing {i}/{len(records)}...")
            if i in skip_indices: continue

            is_duplicate = False
            match_reason = ""
            kept_idx = -1

            for k_idx, kept in enumerate(kept_records):
                try:
                    is_duplicate, match_reason = self.oracle.evaluate_pair(row, kept)
                except Exception as e:
                    if self.verbosity >= 2:
                        print(f"      ⚠️ Oracle error comparing records (Safe Skip): {e}")
                    is_duplicate = False

                if is_duplicate:
                    kept_idx = k_idx
                    break

            if is_duplicate and kept_idx != -1:
                skip_indices.add(i)
                kept_row = kept_records[kept_idx]
                merge_notes = []

                for col in additive_cols:
                    if col in kept_row and col in row:
                        val1, val2 = str(kept_row[col]), str(row[col])

                        if val2 not in self.EMPTY_MARKERS:
                            if val1 in self.EMPTY_MARKERS:
                                kept_row[col] = val2
                                merge_notes.append(f"Filled {col}")
                            elif "URL" in col or "Link" in col:
                                set1 = {v.strip() for v in re.split(r'[,|]', val1) if v.strip()}
                                set2 = {v.strip() for v in re.split(r'[,|]', val2) if v.strip()}
                                new_items = set2 - set1
                                if new_items:
                                    kept_row[col] = val1 + " | " + " | ".join(new_items)
                                    merge_notes.append(f"Added {col}")
                            else:
                                # CRITICAL FIX: Only REPLACE with the longer text, do NOT concatenate descriptions with | anymore
                                if len(val2) > len(val1):
                                    kept_row[col] = val2
                                    merge_notes.append(f"Replaced {col} with longer text")

                for val_col, conf_col in merge_cols:
                    try:
                        c_curr = float(row.get(conf_col, 0)); c_kept = float(kept_row.get(conf_col, 0))
                        v_curr = str(row.get(val_col, ""))
                        if v_curr not in self.EMPTY_MARKERS and c_curr > c_kept:
                            kept_row[val_col] = v_curr; kept_row[conf_col] = c_curr; merge_notes.append(f"Upgraded {val_col}")
                    except: pass

                duplicates_log.append({
                    "DatasetID (Dropped)": row.get('DatasetID', ''),
                    "DatasetID (Kept)": kept_row.get('DatasetID', ''),
                    "Reason": match_reason,
                    "Merge Notes": "; ".join(merge_notes[:5]),
                    "Dropped Name": row.get("Variant Name", ""),
                    "Kept Name": kept_row.get("Variant Name", "")
                })
                kept_records[kept_idx] = kept_row
            else:
                kept_records.append(row)

        df_clean = pd.DataFrame(kept_records)
        df_dupes = pd.DataFrame(duplicates_log)
        return df_clean, df_dupes

    def _analyze_quality(self, df: pd.DataFrame):
        df = df.copy()
        conf_cols = [c for c in df.columns if c.endswith('(C)')]
        stats = {'by_column': {}, 'total_low_conf_cells': 0, 'rows_with_issues': 0}

        for col in conf_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        for idx, row in df.iterrows():
            pid = str(row.get('Project_ID', 'Unknown'))
            display_name = self._get_display_name(row)
            row_issues = 0

            for col in conf_cols:
                clean_name = col.replace(' (C)', '')
                if "frequency" in clean_name.lower() or "time interval" in clean_name.lower():
                    continue

                if row[col] < self.CONFIDENCE_THRESHOLD:
                    stats['by_column'][clean_name] = stats['by_column'].get(clean_name, 0) + 1
                    stats['total_low_conf_cells'] += 1
                    self.project_stats[pid]['missing_counts'][clean_name] += 1
                    row_issues += 1

                    if clean_name == "Num Time Points":
                        self.project_stats[pid]['missing_time_points_list'].append(display_name)

            if row_issues > 0:
                stats['rows_with_issues'] += 1
                self.project_stats[pid]['low_conf'] += 1

            type_val = str(row.get(self.COL_TYPE, '')).lower().strip()
            url_val = str(row.get(self.COL_URL, '')).lower().strip()
            data_val = str(row.get(self.COL_DATA_URL, '')).lower().strip()

            if type_val in self.EMPTY_MARKERS or (url_val in self.EMPTY_MARKERS and data_val in self.EMPTY_MARKERS):
                self.project_stats[pid]['dubious'] += 1

        return stats, df

    def _print_quality_summary(self, stats, total_rows):
        print("\n--- 📊 Final Quality Summary ---")
        print(f"Total Datasets: {total_rows}")
        pct_issues = (stats['rows_with_issues'] / total_rows) * 100 if total_rows > 0 else 0
        print(f"Datasets with ≥1 Low-Conf Field (excluding Frequency): {stats['rows_with_issues']} ({pct_issues:.1f}%)")
        print("\n--- Low Confidence by Field ---")
        sorted_cols = sorted(stats['by_column'].items(), key=lambda x: x, reverse=True)
        for col, count in sorted_cols:
            pct = (count / total_rows) * 100 if total_rows > 0 else 0
            if count > 0: print(f"  • {col}: {count} ({pct:.1f}%)")

    def _print_project_health_report(self):
        print("\n" + "="*80)
        print("🏗️ PROJECT HEALTH & DIAGNOSTICS")
        print("="*80)
        print(f"{'PROJECT ID':<20} | {'TOT':<4} | {'BAD':<4} | {'NORM':<4} | {'STATUS':<10} | {'PRIMARY DEFECTS'}")
        print("-" * 100)

        try:
            if not hasattr(self, 'project_stats') or not isinstance(self.project_stats, dict):
                return

            safe_list = []
            for k in list(self.project_stats.keys()):
                v = self.project_stats[k]
                if isinstance(v, dict):
                    record = {
                        'pid': str(k),
                        'total': int(v.get('total', 0)),
                        'low_conf': int(v.get('low_conf', 0)),
                        'norm_error': int(v.get('norm_error', 0)),
                        'missing_counts': v.get('missing_counts')
                    }
                    safe_list.append(record)

            safe_list.sort(key=lambda x: x['total'], reverse=True)

            for s in safe_list:
                tot = s['total']
                if tot == 0: continue

                defects = []
                missing = s['missing_counts']
                if missing and hasattr(missing, 'most_common'):
                    try:
                        for field, count in missing.most_common(3):
                            pct = int((count / tot) * 100) if tot > 0 else 0
                            defects.append(f"{field} ({pct}%)")
                    except Exception:
                        pass

                defect_str = ", ".join(defects) if defects else "None"
                low = s['low_conf']
                norm = s['norm_error']

                rate = (low / tot) * 100 if tot > 0 else 0
                status = "✅ OK"
                if norm > 0: status = "⚠️ NORM"
                if rate > 30: status = "🛑 RERUN"
                if norm > 5: status = "🛑 CRIT"

                safe_pid = s['pid'][:20]
                print(f"{safe_pid:<20} | {tot:<4} | {low:<4} | {norm:<4} | {status:<10} | {defect_str}")

        except Exception as e:
            print(f"    ⚠️ Report generation failed safely: {e}")

print("✅ deepcollector/kb/quality.py written (Description Bloat Fix).")