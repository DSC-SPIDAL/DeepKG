# =============================================================================
# V86.25: KnowledgeBaseManager (100% Full Un-Truncated + GSpread V5/V6 Fix)
# =============================================================================
import pandas as pd
import re
import uuid
import time
import json
import ast
import random
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

try:
    import pytz
except ImportError:
    pytz = None

try:
    import gspread
    from gspread.exceptions import SpreadsheetNotFound, WorksheetNotFound, APIError
except ImportError:
    gspread = None
    SpreadsheetNotFound = Exception
    WorksheetNotFound = Exception
    APIError = Exception

try:
    from deepcollector.utils.profiler import profiler
except ImportError:
    class DummyProfiler:
        def track(self, c):
            return lambda f: f
    profiler = DummyProfiler()

class SheetLock:
    LOCK_PREFIX = "sys_Lock_GLOBAL_MUTEX"
    EXPIRY_MINUTES = 10

    def __init__(self, manager, job_id, verbosity=1):
        self.manager = manager
        self.job_id = job_id
        self.verbosity = verbosity
        # Static name guarantees 400 APIError on concurrent creation attempts
        self.lock_name = self.LOCK_PREFIX

    def acquire(self, timeout_seconds=600, poll_interval=10) -> bool:
        start_time = time.time()

        if self.verbosity >= 1:
            print(f"🔒 [Lock] Job {self.job_id} requesting truly atomic write lock...")

        while (time.time() - start_time) < timeout_seconds:
            try:
                # 1. Atomic Collision: Only the first execution succeeds
                ws = self.manager.spreadsheet.add_worksheet(title=self.lock_name, rows=1, cols=1)
                ws.update_acell('A1', f"{self.job_id}_{int(time.time())}")
                if self.verbosity >= 1:
                    print(f"    🔒 [Lock] Acquired safely by {self.job_id}.")
                return True
            except Exception:
                try:
                    # 2. If locked, evaluate for zombie expiry safely
                    ws = self.manager.spreadsheet.worksheet(self.lock_name)
                    val = ws.acell('A1').value
                    if val:
                        lock_timestamp = int(val.split("_")[-1])
                        if time.time() - lock_timestamp > (self.EXPIRY_MINUTES * 60):
                            if self.verbosity >= 1:
                                print(f"    ⚠️ [Lock] Breaking expired zombie lock: {ws.title}")
                            self.manager.spreadsheet.del_worksheet(ws)
                except Exception:
                    pass

                if self.verbosity >= 1:
                    print(f"    ⏳ [Lock] Collision or API error. Waiting {poll_interval}s...")
                time.sleep(poll_interval)

        print("❌ [Lock] Failed to acquire lock (Timeout).")
        return False

    def release(self):
        try:
            ws = self.manager.spreadsheet.worksheet(self.lock_name)
            self.manager.spreadsheet.del_worksheet(ws)
            if self.verbosity >= 1:
                print(f"    🔓 [Lock] Released safely.")
        except Exception:
            pass


class KnowledgeBaseManager:
    def __init__(self, config: Any):
        self.config = config
        self.sheet_id = getattr(config, 'GOOGLE_SHEET_KB_ID', None)
        self.kb_data: Dict[str, pd.DataFrame] = {}
        self.gc = None
        self.spreadsheet = None
        self.is_connected = False
        self.verbosity = getattr(config, 'VERBOSITY_LEVEL', 1)
        self.KB_SCHEMA = getattr(config, 'KB_SCHEMA', {})
        self.CATALOG_SCHEMA = getattr(config, 'CATALOG_SCHEMA', {})
        self.EXTRACTED_FIELDS = getattr(config, 'EXTRACTED_FIELDS', [])
        self.MISSING_DATA_PLACEHOLDERS = getattr(config, 'MISSING_DATA_PLACEHOLDERS', set())

        self.FIELD_CONFIDENCE_MAP = {
            "Domain": "Domain (C)",
            "Frequency": "Frequency (C)",
            "Num Time Points": "Num Time Points (C)",
            "Num Locations/Series": "Num Locations/Series (C)",
            "Variables per Location": "Variables per Location (C)",
            "Total Variables": "Total Variables (C)",
            "Primary URL": "Primary URL (C)",
            "Link to Data (Actual Source)": "Link to Data (Actual Source) (C)",
            "Other URL": "Other URL (C)"
        }

    def initialize_connection(self, authenticated_gc: Any):
        if not self.sheet_id or not authenticated_gc:
            if self.verbosity >= 1:
                print("⚠️ [KB Manager] Missing ID or Client. Disabled.")
            return

        self.gc = authenticated_gc
        try:
            self.spreadsheet = self.gc.open_by_key(self.sheet_id)
            self.is_connected = True
            if self.verbosity >= 1:
                print(f"🌐 [KB Manager] Connected to '{self.spreadsheet.title}'")
        except Exception as e:
            print(f"❌ [KB Manager] Connection failed: {e}")

    @profiler.track("KB: Read")
    def read_and_validate_kb(self) -> bool:
        if not self.is_connected:
            return False

        valid = True
        for tab, cols in self.KB_SCHEMA.items():
            try:
                ws = self.spreadsheet.worksheet(tab)

                # =================================================================
                # CRITICAL FIX: Safe handling for gspread v6+
                # =================================================================
                try:
                    if hasattr(ws, 'get_values'):
                        data = ws.get_values(value_render_option='FORMULA')
                    else:
                        try:
                            data = ws.get_all_values(value_render_option='FORMULA')
                        except TypeError:
                            data = ws.get_all_values()
                except Exception:
                    data = ws.get_all_values()

                if data:
                    best_match = 0
                    header_idx = 0
                    for i, row in enumerate(data[:10]):
                        match_count = sum(1 for c in cols if c in row)
                        if match_count > best_match:
                            best_match = match_count
                            header_idx = i

                    if best_match == 0:
                        if self.verbosity >= 1:
                            print(f"    ⚠️ [KB] Header row not found in '{tab}'. Falling back to empty DataFrame.")
                        self.kb_data[tab] = pd.DataFrame(columns=cols)
                        continue

                    df = pd.DataFrame(data[header_idx+1:], columns=data[header_idx])

                    if "DatasetID" in cols and "DatasetID" not in df.columns:
                        print(f"    🚨 [CRITICAL] 'DatasetID' column missing from {tab} headers! Check Google Sheet for structural corruption.")

                    for col in cols:
                        if col not in df.columns:
                            df[col] = ""

                    def extract_url(val):
                        if isinstance(val, str) and val.startswith('=HYPERLINK'):
                            import re
                            urls = re.findall(r'(https?://[^\s|\"]+)', val)
                            unique_urls = []
                            for u in urls:
                                if u not in unique_urls:
                                    unique_urls.append(u)
                            return ", ".join(unique_urls) if unique_urls else val
                        return val

                    for col in df.columns:
                        if "URL" in col or "Link" in col:
                            df[col] = df[col].apply(extract_url)

                    df = df.reindex(columns=cols).fillna("").astype(str)
                    self.kb_data[tab] = df
                    if self.verbosity >= 1:
                        print(f"    ✅ [KB] Loaded '{tab}' ({len(df)} rows).")
                else:
                    self.kb_data[tab] = pd.DataFrame(columns=cols)
            except WorksheetNotFound:
                if self.verbosity >= 1:
                    print(f"    ⚠️ [KB] Tab '{tab}' not found. Initializing empty.")
                self.kb_data[tab] = pd.DataFrame(columns=cols)
            except Exception as e:
                print(f"    ❌ [KB] Error reading '{tab}': {e}")
                valid = False
        return valid

    def get_kb_data(self, tab_name: str) -> pd.DataFrame:
        return self.kb_data.get(tab_name, pd.DataFrame())

    def wipe_kb(self):
        if not self.is_connected:
            return
        print("⚠️ [KB] Wiping all data tabs...")
        for tab, cols in self.KB_SCHEMA.items():
            try:
                try:
                    ws = self.spreadsheet.worksheet(tab)
                except WorksheetNotFound:
                    ws = self.spreadsheet.add_worksheet(title=tab, rows=100, cols=20)
                ws.clear()
                ws.append_row(cols)
                self.kb_data[tab] = pd.DataFrame(columns=cols)
                print(f"    🗑️ Wiped '{tab}'.")
            except Exception as e:
                print(f"    ❌ Failed to wipe '{tab}': {e}")

    def wipe_project(self, project_id: str, job_id: str):
        if not self.is_connected:
            return
        print(f"\n⚠️ [KB] Executing Targeted Wipe for Project: {project_id}...")
        lock = SheetLock(self, job_id, self.verbosity)
        if not lock.acquire():
            print("    ❌ Failed to acquire lock for project wipe.")
            return

        try:
            self.read_and_validate_kb()
            df_links = self.get_kb_data("Project_Dataset_Link")
            df_ds = self.get_kb_data("Datasets")

            if df_links.empty or df_ds.empty:
                print("    ℹ️ No data to wipe.")
                return

            proj_links = df_links[df_links["ProjectID"] == project_id]
            if proj_links.empty:
                print("    ℹ️ Project has no linked datasets to wipe.")
                return

            ds_ids_to_check = proj_links["DatasetID"].tolist()
            df_links_clean = df_links[df_links["ProjectID"] != project_id]
            remaining_links = df_links_clean["DatasetID"].tolist()
            ds_ids_to_delete = [ds for ds in ds_ids_to_check if ds not in remaining_links]

            df_ds_clean = df_ds[~df_ds["DatasetID"].isin(ds_ids_to_delete)]

            self.write_dataframe_to_tab("Project_Dataset_Link", df_links_clean)
            self.write_dataframe_to_tab("Datasets", df_ds_clean)
            self.kb_data["Project_Dataset_Link"] = df_links_clean
            self.kb_data["Datasets"] = df_ds_clean

            print(f"    🗑️ Safely Deleted {len(ds_ids_to_delete)} exclusive datasets and {len(proj_links)} links for {project_id}.")
        except Exception as e:
            print(f"    ❌ Wipe Failed: {e}")
        finally:
            lock.release()

    @staticmethod
    def _col_num_to_letter(n):
        string = ""
        while n > 0:
            n, remainder = divmod(n - 1, 26)
            string = chr(65 + remainder) + string
        return string

    @profiler.track("KB: Write")
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=2, max=20), retry=retry_if_exception_type((APIError, Exception)))
    def write_dataframe_to_tab(self, tab_name: str, df: pd.DataFrame):
        if not self.is_connected:
            return

        try:
            try:
                ws = self.spreadsheet.worksheet(tab_name)
            except WorksheetNotFound:
                if tab_name in self.KB_SCHEMA:
                    ws = self.spreadsheet.add_worksheet(title=tab_name, rows=100, cols=20)
                else:
                    return

            cols = self.KB_SCHEMA.get(tab_name, df.columns.tolist())
            df_write = df.reindex(columns=cols)

            def safe_stringify(val):
                # 1. Process iterables BEFORE pd.isna evaluates
                if isinstance(val, (list, dict, tuple, set)):
                    try:
                        # Join lists to ensure val.startswith("http") triggers downstream
                        if isinstance(val, list):
                            val = ", ".join(str(v) for v in val)
                        else:
                            val = json.dumps(val)
                    except Exception:
                        val = str(val)

                # 2. Now guaranteed scalar/string, pd.isna evaluation is safe
                if pd.isna(val) or val is None:
                    return "[missing]"

                s = str(val).strip()
                if not s:
                    return "[missing]"
                return s

            for col in df_write.columns:
                df_write[col] = df_write[col].apply(safe_stringify)

            def sanitize_cell(col_name, val):
                val = str(val).strip()
                if ("URL" in col_name or "Link" in col_name) and val.startswith("http") and not val.startswith("="):
                    urls = [u.strip() for u in re.split(r'[,|]', val) if u.strip()]
                    if not urls: return f"'{val}"

                    primary_url = str(urls)
                    safe_url = primary_url.replace('"', '""').replace('\n', '').replace('\r', '')

                    if len(urls) == 1:
                        return f'=HYPERLINK("{safe_url}", "{safe_url}")'
                    else:
                        secondary_text = " | ".join(["Alt: " + str(u).replace('"', '""').replace('\n', '').replace('\r', '') for u in urls[1:]])
                        return f'=HYPERLINK("{safe_url}", "{safe_url}") & " | {secondary_text}"'
                else:
                    if val.lstrip().startswith(('=', '+', '-', '@')):
                        return f"'{val}"
                    return val

            for col in df_write.columns:
                df_write[col] = df_write[col].apply(lambda x: sanitize_cell(col, x))

            data = [df_write.columns.values.tolist()] + df_write.values.tolist()
            ws.clear()

            num_rows = len(data)
            num_cols = len(data) if num_rows > 0 else 1

            try:
                ws.resize(rows=max(100, num_rows + 5), cols=max(20, num_cols + 2))
            except Exception:
                pass

            range_addr = "A1"

            try:
                ws.update(values=data, range_name=range_addr, value_input_option='USER_ENTERED')
            except TypeError:
                try:
                    ws.update(range_addr, data, value_input_option='USER_ENTERED')
                except TypeError:
                    ws.update(data, value_input_option='USER_ENTERED')

            if self.verbosity >= 1:
                print(f"    📝 [KB] Wrote {num_rows - 1} records to '{tab_name}'.")

        except Exception as e:
            if self.verbosity >= 1:
                print(f"    ⚠️ [KB Retry] Write to '{tab_name}' failed: {type(e).__name__} - {e}")
                print(f"    🐛 [DEBUG TRACE]:\n{traceback.format_exc()}")
            raise e

    def log_job_execution(self, job_data: Dict[str, Any]):
        if not self.is_connected:
            return

        try:
            tab_name = "Jobs"
            try:
                ws = self.spreadsheet.worksheet(tab_name)
            except WorksheetNotFound:
                ws = self.spreadsheet.add_worksheet(title=tab_name, rows=100, cols=20)
                ws.append_row(self.KB_SCHEMA["Jobs"])

            row = []
            for col in self.KB_SCHEMA["Jobs"]:
                val = job_data.get(col, "")
                if isinstance(val, (dict, list)):
                    val = json.dumps(val)
                row.append(str(val))

            ws.append_row(row)
            if self.verbosity >= 1:
                print(f"    📋 [Job Log] Recorded Job {job_data.get('JobID')} to KB.")
        except Exception as e:
            print(f"    ⚠️ [Job Log Error] Failed to log job: {e}")

    @profiler.track("KB: Reconcile & Write")
    def reconcile_and_write(self, state, current_project_name, job_id=None):
        if not self.is_connected:
            return

        job_id = job_id or f"JOB_{uuid.uuid4().hex[:6]}"

        lock = SheetLock(self, job_id, self.verbosity)
        if not lock.acquire():
            raise RuntimeError("Failed to acquire KB Lock for write-back.")

        try:
            if self.verbosity >= 1:
                print(f"🔄 [KB] JIT Refresh: Downloading latest KB state under lock...")
            if not self.read_and_validate_kb():
                raise RuntimeError("Failed to refresh KB data under lock.")

            if self.verbosity >= 1:
                print(f"🔄 [KB Reconciliation] Processing {len(state.catalog)} items (Job: {job_id})...")

            kb_dfs = {name: df.copy() for name, df in self.kb_data.items()}
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            self._reconcile_projects(state, kb_dfs, current_project_name, current_time)
            citation_ids = self._reconcile_citations(state, kb_dfs)
            weblink_ids = self._reconcile_weblinks(state, kb_dfs)
            link_info = self._reconcile_datasets(state, kb_dfs, current_time, job_id)
            self._manage_linkages(state, kb_dfs, link_info, citation_ids, weblink_ids, current_time, job_id)

            for tab, df in kb_dfs.items():
                self.write_dataframe_to_tab(tab, df)

            self.kb_data = kb_dfs

        except Exception as e:
            print(f"❌ [KB Write Error] Critical Failure: {e}")
            raise e
        finally:
            lock.release()

    def _reconcile_projects(self, state, kb_dfs, project_name, time_str):
        df = kb_dfs.get("Projects")
        if state.project_id in df["ProjectID"].values:
            mask = df["ProjectID"] == state.project_id
            df.loc[mask, "Last Analyzed"] = time_str
            df.loc[mask, "Name"] = project_name
        else:
            new_row = {
                "ProjectID": state.project_id,
                "Name": project_name,
                "Last Analyzed": time_str,
                "Type": "Project",
                "Source URL": getattr(self.config, 'INITIAL_URLS', ['']) if getattr(self.config, 'INITIAL_URLS', []) else ""
            }
            kb_dfs["Projects"] = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    def _reconcile_datasets(self, state, kb_dfs, time_str, job_id):
        df_ds = kb_dfs.get("Datasets")
        link_info = []

        for item in state.catalog:
            name = item.get("Dataset Name", {}).get("value")
            if not name or name == "[missing]":
                continue

            record, field_confs, assign_conf = self._normalize_dataset_entry(state, item, time_str)

            canon = record["Canonical Name"]
            variant = record["Variant Name"]
            mask = (df_ds["Canonical Name"].str.lower() == canon.lower()) & \
                   (df_ds["Variant Name"].str.lower() == variant.lower())

            ds_id = None
            if mask.any():
                idx = mask.idxmax()
                ds_id = df_ds.loc[idx, "DatasetID"]
                row_updated = False

                for field_col, conf_col in self.FIELD_CONFIDENCE_MAP.items():
                    new_val = record.get(field_col, "")
                    new_conf = field_confs.get(field_col, 0.0)

                    try:
                        old_conf = float(df_ds.loc[idx, conf_col])
                    except Exception:
                        old_conf = 0.0

                    if new_conf > old_conf and new_val not in self.MISSING_DATA_PLACEHOLDERS:
                        df_ds.loc[idx, field_col] = new_val
                        df_ds.loc[idx, conf_col] = str(new_conf)
                        row_updated = True

                if len(record["Description"]) > len(str(df_ds.loc[idx, "Description"])) + 10:
                    df_ds.loc[idx, "Description"] = record["Description"]
                    row_updated = True

                current_aliases = str(df_ds.loc[idx, "Aliases"])
                new_alias = record["Variant Name"]
                merged_aliases = self._merge_aliases(current_aliases, new_alias)
                if merged_aliases != current_aliases:
                    df_ds.loc[idx, "Aliases"] = merged_aliases
                    row_updated = True

                if row_updated:
                    df_ds.loc[idx, "Job_Updated"] = job_id
                    df_ds.loc[idx, "Project_Updated"] = state.project_id
                    df_ds.loc[idx, "Date_Updated"] = time_str

            else:
                ds_id = self._generate_new_id("DS_", df_ds, "DatasetID")
                record["DatasetID"] = ds_id
                record["Job_Created"] = job_id
                record["Project_Created"] = state.project_id
                record["Date_Created"] = time_str
                record["Job_Updated"] = job_id
                record["Project_Updated"] = state.project_id
                record["Date_Updated"] = time_str

                for field_col, conf_col in self.FIELD_CONFIDENCE_MAP.items():
                    record[conf_col] = str(field_confs.get(field_col, 0.0))

                df_ds = pd.concat([df_ds, pd.DataFrame([record])], ignore_index=True)

            link_info.append({
                "DatasetID": ds_id,
                "Actual Data URL Used": item.get("Link to Data (Actual Source)", {}).get("value", ""),
                "Assignment Confidence": f"{assign_conf:.4f}"
            })

        kb_dfs["Datasets"] = df_ds
        return link_info

    def _merge_aliases(self, current_str: str, new_val: str) -> str:
        if not current_str or current_str == "[missing]":
            current_str = ""
        if not new_val or new_val == "[missing]":
            return current_str

        existing = []
        for x in current_str.split(','):
            x = x.strip()
            if x and x.lower() not in self.MISSING_DATA_PLACEHOLDERS and "[missing]" not in x.lower():
                existing.append(x)

        existing_lower = {x.lower() for x in existing}

        if new_val.lower() not in existing_lower and new_val.lower() not in self.MISSING_DATA_PLACEHOLDERS and "[missing]" not in new_val.lower():
            existing.append(new_val)

        return ", ".join(existing) if existing else "[missing]"

    def _sanitize_dimensionality(self, time_val, var_val, dataset_name=None):
        val_str = str(var_val).strip()

        if (val_str.startswith("{") and val_str.endswith("}")) or (val_str.count(":") > 2 and "{" in val_str):
            try:
                start = val_str.find("{")
                end = val_str.rfind("}")
                if start != -1 and end != -1:
                    candidate = val_str[start:end+1]
                    data = ast.literal_eval(candidate)
                    if isinstance(data, dict):
                        if dataset_name:
                            target = dataset_name.lower()
                            if dataset_name in data:
                                return str(data[dataset_name])
                            for k, v in data.items():
                                if str(k).lower() in target or target in str(k).lower():
                                    return str(v)
                        return "[extraction_error: dict_returned]"
            except Exception:
                return "[extraction_error: complex_string]"

        if re.search(r'\d+\s*(?:-|to)\s*\d+', val_str, re.IGNORECASE):
            return val_str

        try:
            t_clean = "".join(filter(str.isdigit, str(time_val)))
            v_clean = "".join(filter(str.isdigit, str(val_str)))

            if not t_clean or not v_clean:
                return val_str

            t_int = int(t_clean)
            v_int = int(v_clean)
        except Exception:
            pass

        return val_str

    def _normalize_dataset_entry(self, state, item, time_str):
        name = item.get("Dataset Name", {}).get("value")

        field_confs = {}
        total_conf = 0.0
        count = 0

        CATALOG_TO_SCHEMA = {
            "Domain": "Domain",
            "Time interval between points": "Frequency",
            "Number of Time Points": "Num Time Points",
            "Number of Locations/Series": "Num Locations/Series",
            "Variables per Location": "Variables per Location",
            "Total Variables": "Total Variables",
            "Primary URL": "Primary URL",
            "Link to Data (Actual Source)": "Link to Data (Actual Source)",
            "Other URL": "Other URL",
            "Primary Source Repository": "Primary Creator"
        }

        for cat_field, schema_col in CATALOG_TO_SCHEMA.items():
            data = item.get(cat_field, {})
            val = data.get("value", "")
            conf = data.get("confidence", 0.0)
            if val in self.MISSING_DATA_PLACEHOLDERS:
                conf = 0.0

            field_confs[schema_col] = conf
            total_conf += conf
            count += 1

        avg_cell_conf = (total_conf / count) if count else 0
        try:
            assign_conf = float(item.get("Assignment Confidence", {}).get("value", 0))
        except Exception:
            assign_conf = 0.0

        overall_conf = assign_conf * avg_cell_conf

        def get_val(f):
            return item.get(f, {}).get("value", "")

        raw_time = get_val("Number of Time Points")
        raw_vars = get_val("Total Variables")
        raw_vars_loc = get_val("Variables per Location")

        clean_vars = self._sanitize_dimensionality(raw_time, raw_vars, name)
        clean_vars_loc = self._sanitize_dimensionality(raw_time, raw_vars_loc, name)

        ds_type = get_val("Type")
        if not ds_type or ds_type == "[missing]":
             ds_type = "Dataset"

        record = {
            "Canonical Name": get_val("Canonical Name") or name,
            "Variant Name": name,
            "Aliases": get_val("Aliases"),
            "Description": get_val("Detailed Description"),
            "Domain": get_val("Domain"),
            "Frequency": get_val("Time interval between points"),
            "Num Time Points": raw_time,
            "Num Locations/Series": get_val("Number of Locations/Series"),
            "Variables per Location": clean_vars_loc,
            "Total Variables": clean_vars,
            "Primary Creator": get_val("Primary Source Repository"),
            "Primary URL": get_val("Primary URL"),
            "Link to Data (Actual Source)": get_val("Link to Data (Actual Source)"),
            "Other URL": get_val("Other URL"),
            "Overall Confidence": f"{overall_conf:.4f}",
            "Type": ds_type
        }
        return record, field_confs, assign_conf

    def _manage_linkages(self, state, kb_dfs, link_info, cite_ids, web_ids, time_str, job_id):
        pid = state.project_id

        df_pdl = kb_dfs.get("Project_Dataset_Link")
        df_pdl = df_pdl[df_pdl["ProjectID"] != pid]

        for info in link_info:
            new = {
                **info,
                "LinkID": self._generate_new_id("PDL_", df_pdl, "LinkID"),
                "ProjectID": pid,
                "Link_Date": time_str,
                "Linked_By_Job": job_id
            }
            df_pdl = pd.concat([df_pdl, pd.DataFrame([new])], ignore_index=True)

        kb_dfs["Project_Dataset_Link"] = df_pdl

    def _reconcile_citations(self, state, kb_dfs):
        df = kb_dfs.get("Citations")
        ids = {}
        for c in state.discovered_citations:
            title = c.get("Title", "").strip()
            if not title: continue

            match = df[df["Title"].str.lower() == title.lower()]
            if not match.empty:
                ids[title] = match.iloc["CitationID"]
            else:
                new_id = self._generate_new_id("CITE_", df, "CitationID")
                row = {
                    "CitationID": new_id,
                    "Title": title,
                    "Authors": c.get("Authors", ""),
                    "Venue": c.get("Venue", ""),
                    "Year": c.get("Year", ""),
                    "DOI": c.get("DOI", ""),
                    "URL": c.get("URL", ""),
                    "Full Citation Text": c.get("Full Citation Text", "")
                }
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                ids[title] = new_id

        kb_dfs["Citations"] = df
        return ids

    def _reconcile_weblinks(self, state, kb_dfs):
        df = kb_dfs.get("WebLinks")
        ids = {}
        for w in state.discovered_weblinks:
            url = w.get("URL", "").strip()
            if not url: continue

            match = df[df["URL"] == url]
            if not match.empty:
                ids[url] = match.iloc["WebLinkID"]
            else:
                new_id = self._generate_new_id("WEB_", df, "WebLinkID")
                row = {
                    "WebLinkID": new_id,
                    "URL": url,
                    "Resource Type": w.get("Resource Type", ""),
                    "Description": w.get("Description", "")
                }
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                ids[url] = new_id

        kb_dfs["WebLinks"] = df
        return ids

    def _generate_new_id(self, prefix, df, col):
        if df.empty or col not in df.columns:
            return f"{prefix}001"
        try:
            nums = df[col].astype(str).str.extract(rf"{prefix}(\d+)").dropna().astype(int)
            return f"{prefix}{int(nums.max().iloc) + 1:03d}" if not nums.empty else f"{prefix}001"
        except Exception:
            return f"{prefix}{uuid.uuid4().hex[:6]}"

print("✅ deepcollector/kb/manager.py written (100% Full Un-Truncated + GSpread V5/V6 Safe Fetch).")
