# =============================================================================
# V58.17: Project Merger (PRO-Oracle Upgrade & JSON Artifact Scrubbing)
# =============================================================================
import pandas as pd
import difflib
import re
import json
import time
import sys
from collections import defaultdict
from typing import Dict, Tuple, List, Any
from tabulate import tabulate
from pydantic import BaseModel, Field

try:
    from google.genai import types
except ImportError:
    types = None

class OracleResponseSchema(BaseModel):
    is_same: bool = Field(description="True if records describe the exact same entity.")
    rationale: str = Field(description="Brief reason for the decision.")

class SingletonVerificationSchema(BaseModel):
    in_project: bool = Field(description="True if dataset belongs to the Target Entity.")
    exists: bool = Field(description="True if dataset actually exists.")
    rationale: str = Field(description="Short explanation.")

class UniversalOracle:
    def __init__(self, models=None, verbosity=1, enable_search=False):
        self.models = models
        self.verbosity = verbosity
        self.enable_search = enable_search
        self.llm_cache = {}
        self.llm_limit = 1500
        self.model_cooldowns = {}
        self.model_stats = defaultdict(lambda: {"count": 0, "time": 0.0, "time_sq": 0.0})
        self.stats = {'llm_calls': 0, 'approved': 0, 'rejected': 0, 'errors': 0, 'hard_blocks': 0, 'rate_limits_hit': 0}
        self.MISSING = {"", "nan", "none", "[missing]", "[skipped]", "unknown", "n/a", "null"}
        self.STOP_WORDS = {"dataset", "data", "benchmark", "collection", "repository", "archive", "series", "time", "prediction", "forecasting", "competition"}
        self.GENERIC_URLS = ["search", "query", "google.com", "bing.com", "kaggle.com", "archive.ics.uci.edu", "zenodo.org", "github.com", "timeseriesclassification.com", "huggingface.co", "hf.co"]
        self.DISCRIMINATORS = {"france", "belgium", "germany", "spain", "italy", "uk", "usa", "china", "japan", "australia", "nord", "pjm", "california", "texas", "ny", "london", "ottawa", "halifax", "hourly", "daily", "weekly", "monthly", "yearly", "minute", "second", "test", "train", "val", "validation", "small", "large", "full", "subset", "np", "de", "fr", "be", "es", "it", "pems03", "pems04", "pems07", "pems08", "m1", "m3", "m4", "m5", "gef12", "gfc12", "gfc14", "gfc17"}

    def _call_llm_with_cascade(self, prompt: str, pool_preference: str = "FLASH", **kwargs) -> str:
        if not self.models or not getattr(self.models, 'CLIENT', None):
            sys.exit("LLM Client missing in Oracle.")

        all_models = (self.models.FLASH_POOL or []) + (self.models.PRO_POOL or []) if pool_preference == "FLASH" else (self.models.PRO_POOL or []) + (self.models.FLASH_POOL or [])
        base_sleep = 1.0 if pool_preference == "FLASH" else 3.0

        if not all_models: sys.exit("No models available.")
        base_config = kwargs.get("config", None)

        for attempt in range(4):
            for current_idx, model_name in enumerate(all_models):
                if time.time() < self.model_cooldowns.get(model_name, 0): continue
                api_start = time.time()
                
                req_kwargs = {"model": model_name, "contents": prompt}
                
                if types:
                    c_obj = types.GenerateContentConfig()
                    if base_config:
                        for attr in ['temperature', 'top_p', 'top_k', 'candidate_count', 'max_output_tokens', 'stop_sequences', 'response_mime_type', 'response_schema', 'system_instruction', 'tools']:
                            if hasattr(base_config, attr) and getattr(base_config, attr) is not None:
                                setattr(c_obj, attr, getattr(base_config, attr))
                                
                    if self.enable_search:
                        c_obj.tools = [types.Tool(google_search=types.GoogleSearch())]
                        
                    # INJECT THINKING CONFIG EXCLUSIVELY FOR 3.1-PRO
                    if "3.1-pro" in model_name or "3-pro" in model_name:
                        c_obj.thinking_config = types.ThinkingConfig(thinking_budget=4096)
                        
                    req_kwargs["config"] = c_obj

                try:
                    time.sleep(base_sleep)
                    response = self.models.CLIENT.models.generate_content(**req_kwargs)
                    dur = time.time() - api_start
                    self.model_stats[model_name]["time"] += dur
                    self.model_stats[model_name]["time_sq"] += dur ** 2
                    self.model_stats[model_name]["count"] += 1
                    return response.text

                except Exception as e:
                    dur = time.time() - api_start
                    self.model_stats[model_name]["time"] += dur
                    self.model_stats[model_name]["time_sq"] += dur ** 2
                    self.model_stats[model_name]["count"] += 1

                    err_str = str(e).lower()
                    
                    if "404" in err_str or "not found" in err_str:
                        self.stats['errors'] += 1
                        if self.verbosity >= 1: print(f"        ⚠️ Model {model_name} Not Found (404). Cascading...")
                        self.model_cooldowns[model_name] = time.time() + 86400 
                        if current_idx < len(all_models) - 1: continue
                    elif "429" in err_str or "exhausted" in err_str or "quota" in err_str or "503" in err_str:
                        self.stats['rate_limits_hit'] += 1
                        if self.verbosity >= 2: print(f"        ⏭️ Quota/503 hit on {model_name}. Cascading...")
                        self.model_cooldowns[model_name] = time.time() + 120
                        if current_idx < len(all_models) - 1: continue
                    else:
                        self.stats['errors'] += 1
                        if self.verbosity >= 2: print(f"        ⚠️ API Error on {model_name}: {str(e)[:100]}")
                        if current_idx < len(all_models) - 1: continue

            max_cooldown = max(self.model_cooldowns.values()) if self.model_cooldowns else time.time()
            sleep_time = max(1.0, max_cooldown - time.time() + 1.0)
            if sleep_time > 120: sleep_time = 15
            if self.verbosity >= 1: print(f"        💤 Models exhausted. Sleeping {sleep_time:.1f}s for quota recovery...")
            time.sleep(sleep_time)

        if self.verbosity >= 1: print("\n🛑 CRITICAL ABORT: Failed across all available models after multiple attempts.")
        raise RuntimeError("LLM Cascade failed: All models exhausted or on cooldown.")

    def _are_names_distinct_variants(self, n1, n2):
        n1 = str(n1).lower().strip(); n2 = str(n2).lower().strip()
        if n1 == n2: return False

        if len(n1) <= 6 and len(n2) <= 6 and n1 != n2:
            self.stats['hard_blocks'] += 1
            return True

        m1 = re.search(r'(\d+)$', n1); m2 = re.search(r'(\d+)$', n2)
        if m1 and m2 and m1.group(1) != m2.group(1):
            if n1[:m1.start()].strip() == n2[:m2.start()].strip():
                self.stats['hard_blocks'] += 1
                return True
        tokens1 = set(re.findall(r'[a-z0-9]+', n1)); tokens2 = set(re.findall(r'[a-z0-9]+', n2))
        d1 = self.DISCRIMINATORS.intersection(tokens1)
        d2 = self.DISCRIMINATORS.intersection(tokens2)

        if (d1 or d2) and (d1 != d2):
            self.stats['hard_blocks'] += 1
            return True
        return False

    def _clean_name(self, name):
        try:
            if not name: return ""
            n_str = str(name).lower().strip()
            if n_str in self.MISSING: return ""
            n_str = re.sub(r'\(.*?\)', '', n_str)
            return " ".join([t for t in re.split(r'[\s_\-]+', n_str) if t not in self.STOP_WORDS and t.isalnum()])
        except Exception:
            return ""

    def _extract_digits(self, val):
        try:
            if not val: return -1
            v_str = str(val).lower().strip()
            if v_str in self.MISSING: return -1
            return int(re.findall(r'\d+', v_str.replace(',', ''))[0])
        except Exception:
            return -1

    def _clean_url(self, url):
        try:
            if not url: return ""
            u_str = str(url).strip().lower()
            if u_str in self.MISSING: return ""
            # NEW: Strip nasty JSON list artifacts left by Gemini Flash (e.g., '["url"]')
            u_str = re.sub(r'^[\[\'\"]+|[\]\'\"]+$', '', u_str)
            if '?' in u_str: u_str = u_str.split('?')[0]
            if hasattr(u_str, 'rstrip'): return u_str.rstrip('/')
            return str(u_str)
        except Exception:
            return str(url)

    def _is_generic_url(self, url):
        if not url or len(str(url)) < 15: return True
        for bad in self.GENERIC_URLS:
            if bad in str(url): return True
        return False

    def evaluate_pair(self, row_a: Dict, row_b: Dict) -> Tuple[bool, str]:
        name_a = str(row_a.get("Variant Name", "") or row_a.get("Canonical Name", "")).strip()
        name_b = str(row_b.get("Variant Name", "") or row_b.get("Canonical Name", "")).strip()

        if self._are_names_distinct_variants(name_a, name_b): return False, "HARD_DISCRIMINATOR_CONFLICT"

        clean_a = self._clean_name(name_a); clean_b = self._clean_name(name_b)
        t_a = self._extract_digits(row_a.get("Num Time Points", ""))
        t_b = self._extract_digits(row_b.get("Num Time Points", ""))
        v_a = self._extract_digits(row_a.get("Total Variables", ""))
        v_b = self._extract_digits(row_b.get("Total Variables", ""))
        url_a = self._clean_url(row_a.get("Primary URL", "")); url_b = self._clean_url(row_b.get("Primary URL", ""))
        data_a = self._clean_url(row_a.get("Link to Data (Actual Source)", "")); data_b = self._clean_url(row_b.get("Link to Data (Actual Source)", ""))
        type_a = str(row_a.get("Type", "")).strip(); type_b = str(row_b.get("Type", "")).strip()

        ratio = difflib.SequenceMatcher(None, clean_a, clean_b).ratio()
        is_substring = (clean_a in clean_b or clean_b in clean_a) and len(clean_a) > 4 and len(clean_b) > 4
        match_reason = None

        fuzzy_threshold = 0.90 if len(clean_a) <= 8 or len(clean_b) <= 8 else 0.82

        if (url_a and url_b and url_a != "[missing]" and url_a == url_b and not self._is_generic_url(url_a)) or \
           (data_a and data_b and data_a != "[missing]" and data_a == data_b and not self._is_generic_url(data_a)):
            match_reason = "EXACT_URL_MATCH_WEAK_NAME"
        elif t_a > 10 and t_b > 10 and v_a > 0 and v_b > 0 and t_a == t_b and v_a == v_b and ratio >= 0.40: match_reason = "FINGERPRINT_STRONG"
        elif clean_a and clean_b and clean_a == clean_b: match_reason = "EXACT_NAME"
        elif ratio >= 0.90: match_reason = "FUZZY_STRONG"
        elif ratio >= fuzzy_threshold or is_substring: match_reason = "FUZZY_WEAK"
        elif len(clean_a) > 3 and len(clean_b) > 3 and (clean_a.startswith(clean_b) or clean_b.startswith(clean_a)): match_reason = "PREFIX_MATCH_WEAK"

        if not match_reason: return False, ""

        if type_a and type_b and type_a.lower() != type_b.lower() and type_a.lower() not in self.MISSING and type_b.lower() not in self.MISSING:
            match_reason = f"{match_reason}_TYPE_MISMATCH"

        if "WEAK" in match_reason or "MISMATCH" in match_reason or "FINGERPRINT" in match_reason:
            if self.models and hasattr(self.models, 'CLIENT') and self.models.CLIENT:
                if self.stats['llm_calls'] < self.llm_limit:
                    is_llm_match = self._verify_with_llm(row_a, row_b, match_reason)
                    if is_llm_match: return True, f"{match_reason} (LLM Verified)"
                    else: return False, f"{match_reason} (LLM Rejected)"
                else: return False, f"{match_reason} (LLM Limit Reached)"
            else: return False, f"{match_reason} (LLM Disabled)"

        return True, match_reason

    def _verify_with_llm(self, row_a, row_b, match_type):
        name_a = row_a.get("Variant Name", "") or row_a.get("Canonical Name", "")
        name_b = row_b.get("Variant Name", "") or row_b.get("Canonical Name", "")
        type_a = str(row_a.get("Type", "Dataset")).strip()
        type_b = str(row_b.get("Type", "Dataset")).strip()
        desc_a = str(row_a.get("Description", ""))[:300]
        desc_b = str(row_b.get("Description", ""))[:300]
        t_a = row_a.get("Num Time Points", ""); v_a = row_a.get("Total Variables", "")
        t_b = row_b.get("Num Time Points", ""); v_b = row_b.get("Total Variables", "")
        url_a = row_a.get("Primary URL", ""); url_b = row_b.get("Primary URL", "")

        entity_a = (str(name_a), str(t_a), str(v_a), str(url_a))
        entity_b = (str(name_b), str(t_b), str(v_b), str(url_b))
        cache_key = tuple(sorted([entity_a, entity_b]))

        if cache_key in self.llm_cache: return self.llm_cache[cache_key]

        task_desc = f"Arbitrating (Rule: {match_type}): '{name_a[:35]}' vs '{name_b[:35]}'"
        prompt = (
            f"Act as a Data Archivist deduplicating a Time Series dataset catalog.\n"
            f"Determine if these two records describe the EXACT SAME underlying entity, even if names vary.\n"
            f"CRITICAL: Pay close attention to version numbers and geographic regions.\n"
            f"Type labels are sometimes applied inconsistently. Do NOT reject solely on Type mismatch.\n\n"
            f"RECORD 1:\nName: {name_a}\nType: {type_a}\nTime Points: {t_a}\nVariables: {v_a}\nURL: {url_a}\nDescription: {desc_a}\n\n"
            f"RECORD 2:\nName: {name_b}\nType: {type_b}\nTime Points: {t_b}\nVariables: {v_b}\nURL: {url_b}\nDescription: {desc_b}\n\n"
            f"Triggered by Rule: {match_type}"
        )

        try:
            self.stats['llm_calls'] += 1
            kwargs = {}
            if types: kwargs["config"] = types.GenerateContentConfig(response_mime_type="application/json", response_schema=OracleResponseSchema)
            
            # 🔥 NEW: Force the Oracle to use PRO models instead of FLASH for higher accuracy
            response_text = self._call_llm_with_cascade(prompt, pool_preference="PRO", **kwargs)

            try:
                result = json.loads(response_text)
                is_match = result.get('is_same', False)
            except json.JSONDecodeError:
                match = re.search(r"\{.*?\}", response_text.strip(), re.DOTALL)
                if match:
                    try:
                        result = json.loads(match.group())
                        is_match = result.get('is_same', False)
                    except json.JSONDecodeError:
                        is_match = False
                else: is_match = False

            self.llm_cache[cache_key] = is_match
            if is_match:
                self.stats['approved'] += 1
                if self.verbosity >= 1: print(f"      ✅ Confirmed: {task_desc}")
            else:
                self.stats['rejected'] += 1
                if self.verbosity >= 1: print(f"      ❌ Rejected:  {task_desc}")
            return is_match

        except Exception as e:
            self.stats['errors'] += 1
            if self.verbosity >= 1: print(f"      ⚠️ Error/Timeout. Defaulting to Rejected. ({str(e)[:40]}): {task_desc}")
            return False

class ProjectMerger:
    def __init__(self, kb_manager, verbosity=1, tools=None, models=None):
        self.kb_manager = kb_manager
        self.verbosity = verbosity
        self.tools = tools
        enable_search = getattr(self.kb_manager.config, 'ENABLE_ORACLE_SEARCH', False)
        self.oracle = UniversalOracle(models=models, verbosity=verbosity, enable_search=enable_search)
        self.MISSING = self.oracle.MISSING
        self.DATA_FIELDS = ["Domain", "Frequency", "Num Time Points", "Num Locations/Series", "Total Variables", "Variables per Location", "Primary Creator", "Primary URL", "Link to Data (Actual Source)", "Other URL", "Description", "Type"]

    def _build_project_grounding_context(self, project_context, urls):
        if not urls: return ""
        if self.verbosity >= 1: print(f"    🌐 [Merger] Fetching grounding text from {len(urls)} URL(s)...")
        combined_text = ""
        for i, url in enumerate(urls[:5]):
            try:
                fetched = self.tools.tool_load_url(url)
                if fetched and isinstance(fetched, list) and len(fetched) > 0 and "content" in fetched[0]:
                    combined_text += f"\n\n--- SOURCE {i+1}: {url} ---\n{fetched[0]['content']}"
            except Exception as e:
                if self.verbosity >= 2: print(f"      ⚠️ Failed to fetch {url}: {e}")
        if not combined_text.strip(): return ""
        final_text = re.sub(r'\s+', ' ', combined_text).strip()[:1000000]
        if self.verbosity >= 1: print(f"      ✅ Loaded {len(final_text)} characters of raw context for Oracle Verification.")
        return final_text

    def execute_merge(self, project_id: str, job_id: str, dry_run: bool = False, models_verifier=None, enable_singleton_verification: bool = None):
        if models_verifier: self.oracle.models = models_verifier
        if enable_singleton_verification is None: enable_singleton_verification = getattr(self.kb_manager.config, 'ENABLE_SINGLETON_VERIFICATION', True)

        is_global = (project_id == "GLOBAL")
        if is_global: enable_singleton_verification = False

        if enable_singleton_verification and (not self.oracle.models or not getattr(self.oracle.models, 'CLIENT', None)):
            sys.exit("LLM Client missing in MERGE mode.")

        project_context = getattr(self.kb_manager.config, 'PROJECT_CONTEXT', project_id.replace("PROJ_", ""))
        mode_str = "🚫 DRY RUN" if dry_run else "💾 LIVE"
        scope_str = "🌍 GLOBAL SCAN" if is_global else f"🎯 PROJECT: {project_id}"
        search_str = "🔍 Web Grounding ON" if self.oracle.enable_search else "📖 Closed Book"
        print(f"\n🤝 STARTING MERGE: {scope_str} [{mode_str}] | {search_str} (V58.17)")

        if not enable_singleton_verification:
            if is_global: print("    🌍 GLOBAL MODE: Bypassing Singleton Project-Relevance Checks (Deduplication Only).")
            else: print("    ⏩ Note: LLM Singleton Verification is bypassed.")

        project_source_text = ""
        if not is_global and enable_singleton_verification and self.tools and getattr(self.kb_manager.config, 'INITIAL_URLS', []):
            project_source_text = self._build_project_grounding_context(project_context, getattr(self.kb_manager.config, 'INITIAL_URLS', []))

        from deepcollector.kb.manager import SheetLock
        lock = SheetLock(self.kb_manager, job_id, self.verbosity)
        if not lock.acquire(timeout_seconds=14400):
            print("    ❌ Failed to acquire lock for merge operation.")
            return

        try:
            if not self.kb_manager.read_and_validate_kb():
                print("    ❌ Failed to load Knowledge Base.")
                return

            df_ds = self.kb_manager.get_kb_data("Datasets")
            df_links = self.kb_manager.get_kb_data("Project_Dataset_Link")

            if df_ds.empty: print("    ℹ️ No data."); return

            if is_global:
                project_datasets = df_ds.to_dict('records')
                linked_ids = set()
            else:
                proj_links = df_links[df_links["ProjectID"] == project_id]
                linked_job_map = dict(zip(proj_links["DatasetID"], proj_links["Linked_By_Job"]))
                linked_ids = proj_links["DatasetID"].tolist()
                project_datasets = df_ds[df_ds["DatasetID"].isin(linked_ids)].to_dict('records')
                for row in project_datasets: row["_Active_Linked_Job"] = linked_job_map.get(row["DatasetID"], "UnknownJob")

            if not project_datasets: print(f"    ℹ️ No datasets found."); return

            total_rows = len(project_datasets)
            print(f"    📥 Loaded {total_rows} rows. Grouping & Verifying with Oracle...")

            groups = {}
            project_datasets.sort(key=lambda x: float(x.get("Overall Confidence", 0) or 0), reverse=True)

            for i, row in enumerate(project_datasets):
                if i > 0 and i % 20 == 0: print(f"      ⏳ Grouping progress: {i}/{total_rows} datasets evaluated...")
                key, match_type = self._find_group_key(row, groups)
                if key: groups[key].append(row)
                else: groups[row.get("DatasetID")] = [row]

            golden_records = []
            all_dropped_ids = []
            report_rows = []
            unmerged_singletons = []

            for grp_id, rows in groups.items():
                if len(rows) == 1:
                    unmerged_singletons.append(rows[0])
                    golden_records.append(rows[0])
                    continue

                golden = rows[0].copy()
                kept_id = golden["DatasetID"]

                dropped_ids = []
                dropped_names = []
                for r in rows[1:]:
                    dropped_ids.append(r["DatasetID"])
                    dropped_names.append(r.get("Variant Name", "") or r.get("Canonical Name", "Unknown"))

                changes = []
                for challenger in rows[1:]: self._merge_into_golden(golden, challenger, changes)

                projs = {str(r.get("Project_Created")) for r in rows if r.get("Project_Created")}
                projs.update({str(r.get("Project_Updated")) for r in rows if r.get("Project_Updated")})
                projs = {p for p in projs if p and p.lower() != "nan"}

                report_rows.append({
                    "Golden ID": kept_id, "Golden Name": golden.get("Variant Name", "")[:30],
                    "Absorbed Names": ", ".join(dropped_names)[:60], "Merged IDs": ", ".join(dropped_ids),
                    "Projects": ", ".join(sorted(projs)), "Changes": len(changes)
                })
                all_dropped_ids.extend(dropped_ids)

                if not dry_run:
                    golden["Job_Updated"] = job_id
                    golden["Date_Updated"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    if not is_global: golden["Project_Updated"] = project_id

                if "_Active_Linked_Job" in golden: del golden["_Active_Linked_Job"]
                golden_records.append(golden)

            singleton_actions = {}
            if unmerged_singletons:
                singleton_actions = self._evaluate_and_print_singletons(unmerged_singletons, project_id, project_context, is_global, run_llm=enable_singleton_verification, project_source_text=project_source_text)

            final_golden_records = []
            dropped_singletons = []
            unlink_singletons = []
            lost_singletons = []

            for r in golden_records:
                ds_id = r["DatasetID"]
                action = singleton_actions.get(ds_id, "KEEP")
                total_links_count = len(df_links[df_links["DatasetID"] == ds_id]["ProjectID"].unique())

                if action == "DROP":
                    if total_links_count > 1 and not is_global:
                        unlink_singletons.append(ds_id); final_golden_records.append(r)
                    else: dropped_singletons.append(ds_id)
                elif action == "LOST":
                    if total_links_count > 1 and not is_global:
                        unlink_singletons.append(ds_id); final_golden_records.append(r)
                    else: lost_singletons.append(ds_id); final_golden_records.append(r)
                else: final_golden_records.append(r)

            if dropped_singletons: print(f"    🗑️ Hallucinations Dropped (Will be removed from DB completely): {len(dropped_singletons)}")
            if unlink_singletons: print(f"    ✂️ Unlinked from {project_id} (Does not belong here, but kept in DB): {len(unlink_singletons)}")
            if lost_singletons: print(f"    🗂️ Reassigned completely to PROJ_LOST: {len(lost_singletons)}")

            print("\n📊 MERGE REPORT")
            print(f"    Input: {len(project_datasets)} -> Output: {len(final_golden_records)}")
            print(f"    Duplicates Absorbed: {len(all_dropped_ids)}")

            if report_rows:
                pd.DataFrame(report_rows).to_csv("Merge_Log.csv", index=False)
                print("    📝 Full Merge Log saved to: Merge_Log.csv")
                print(f"\n    🟢 SUCCESSFUL MERGES ({len(report_rows)} groups consolidated):")
                print(tabulate(report_rows, headers="keys", tablefmt="simple"))
            else: print("    ℹ️ No duplicates found.")

            if dry_run:
                print("\n🚫 DRY RUN: Changes NOT saved.")
                return

            print("\n💾 SAVING CHANGES...")

            if lost_singletons:
                df_proj = self.kb_manager.get_kb_data("Projects")
                if "PROJ_LOST" not in df_proj["ProjectID"].values:
                    df_proj = pd.concat([df_proj, pd.DataFrame([{"ProjectID": "PROJ_LOST", "Name": "Lost / Incidental Datasets", "Type": "System"}])], ignore_index=True)
                    self.kb_manager.write_dataframe_to_tab("Projects", df_proj)

            if is_global: df_final = pd.DataFrame(final_golden_records)
            else:
                df_ds_clean = df_ds[~df_ds["DatasetID"].isin(linked_ids)]
                df_final = pd.concat([df_ds_clean, pd.DataFrame(final_golden_records)], ignore_index=True)

            self.kb_manager.write_dataframe_to_tab("Datasets", df_final)

            if all_dropped_ids or dropped_singletons or lost_singletons or unlink_singletons:
                if all_dropped_ids:
                    id_map = {}
                    for grp_id, rows in groups.items():
                        if len(rows) > 1:
                            for r in rows[1:]: id_map[r["DatasetID"]] = rows[0]["DatasetID"]
                    mask = df_links["DatasetID"].isin(id_map.keys())
                    if mask.any():
                        df_links.loc[mask, "DatasetID"] = df_links.loc[mask, "DatasetID"].map(id_map).fillna(df_links.loc[mask, "DatasetID"])
                        curr_c = df_links.loc[mask, "Data Preparation Comments"].fillna("").astype(str)
                        df_links.loc[mask, "Data Preparation Comments"] = curr_c.apply(lambda x: f"{x} [OracleMerge]" if "[OracleMerge]" not in x else x)

                if dropped_singletons:
                    df_links = df_links[~df_links["DatasetID"].isin(dropped_singletons)].copy()
                if unlink_singletons and not is_global:
                    df_links = df_links[~((df_links["DatasetID"].isin(unlink_singletons)) & (df_links["ProjectID"] == project_id))].copy()
                if lost_singletons:
                    mask_lost = df_links["DatasetID"].isin(lost_singletons)
                    if not is_global: mask_lost = mask_lost & (df_links["ProjectID"] == project_id)
                    df_links.loc[mask_lost, "ProjectID"] = "PROJ_LOST"
                    curr_c = df_links.loc[mask_lost, "Data Preparation Comments"].fillna("").astype(str)
                    df_links.loc[mask_lost, "Data Preparation Comments"] = curr_c.apply(lambda x: f"{x} [Reassigned to PROJ_LOST]" if "[Reassigned to PROJ_LOST]" not in x else x)

                df_links = df_links.drop_duplicates(subset=["ProjectID", "DatasetID"])
                self.kb_manager.write_dataframe_to_tab("Project_Dataset_Link", df_links)
                print(f"    🔗 Project Links Updated.")

            print(f"✅ Merge Complete.")

        except Exception as e:
            print(f"    ❌ [Merge Error] Critical Failure: {e}")

        finally:
            lock.release()

    def _verify_singleton_context_with_llm(self, row, project_id, project_context, project_source_text=""):
        if not self.oracle.models or not hasattr(self.oracle.models, 'CLIENT'): return "N/A", "N/A", "LLM models missing"
        name = row.get("Variant Name", "") or row.get("Canonical Name", "Unknown")
        type_val = str(row.get("Type", "Dataset")).strip()
        desc = str(row.get("Description", ""))[:500]
        url = str(row.get("Primary URL", ""))
        proj_name = project_id.replace("PROJ_", "")
        source_context_block = f"--- BEGIN SOURCE ARTICLES / REPOSITORIES ---\n{project_source_text}\n--- END SOURCE ARTICLES ---\n\n" if project_source_text else ""
        prompt = (
            f"Act as a Data Archivist. We are auditing a catalog of time-series datasets and collections.\n\n"
            f"TARGET PARENT ENTITY:\nName: {proj_name}\nContext/Description: {project_context}\n\n{source_context_block}"
            f"CANDIDATE DATASET TO EVALUATE:\nName: {name}\nType: {type_val}\nURL: {url}\nDescription: {desc}\n\n"
            f"TASK:\n1. Real-World Existence: Does this candidate dataset actually exist in the real world?\n"
            f"2. Contextual Relevance: Does this dataset belong to the Target Parent Entity? \n"
            f"   - VERY IMPORTANT: Datasets frequently belong to multiple collections, repositories, or listicles.\n"
            f"   - If 'SOURCE ARTICLES / REPOSITORIES' text is provided above, you MUST scan it. If the candidate dataset is explicitly mentioned, utilized, or listed anywhere in that text, it unequivocally belongs to this Project. Do NOT reject it just because it is hosted elsewhere.\n"
            f"   - Be highly lenient with naming variations. If the Target features 'Ozone' and candidate is 'Ozone Level Detection Data Set (UCI)', that is a MATCH.\n"
        )
        try:
            self.oracle.stats['llm_calls'] += 1
            kwargs = {}
            if types: kwargs["config"] = types.GenerateContentConfig(response_mime_type="application/json", response_schema=SingletonVerificationSchema)
            response_text = self.oracle._call_llm_with_cascade(prompt, pool_preference="PRO" if project_source_text else "PRO", **kwargs)
            try:
                res = json.loads(response_text)
                return "Yes" if res.get("in_project") else "No", "Yes" if res.get("exists") else "No", res.get("rationale", "")[:80]
            except json.JSONDecodeError:
                match = re.search(r"\{.*?\}", response_text.strip(), re.DOTALL)
                if match:
                    try:
                        res = json.loads(match.group())
                        return "Yes" if res.get("in_project") else "No", "Yes" if res.get("exists") else "No", res.get("rationale", "")[:80]
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            self.oracle.stats['errors'] += 1
            return "Error", "Error", str(e)[:40]
        return "Unknown", "Unknown", "Failed to parse JSON"

    def _evaluate_and_print_singletons(self, singletons, project_id, project_context, is_global, run_llm=False, project_source_text=""):
        print(f"\n    🟡 UNMERGED DATASETS ({len(singletons)} Singletons)")
        if is_global: run_llm = False; print("    ⏩ GLOBAL Mode: Skipping LLM context verification for singletons (Already verified at project level).")
        elif run_llm and self.oracle.models and hasattr(self.oracle.models, 'CLIENT'):
            search_badge = " (Search Grounding ON)" if self.oracle.enable_search else ""
            if project_source_text: search_badge += f" (+ {len(project_source_text)} chars of Source Text)"
            print(f"    🕵️ Running Context-Aware LLM Verification against '{project_id}'{search_badge}...")
        else: run_llm = False

        s_data = []
        for i, r in enumerate(singletons):
            did = r.get("DatasetID", "UnknownID")
            name = r.get("Variant Name", "") or r.get("Canonical Name", "Unknown")
            clean_name = self.oracle._clean_name(name)
            try: conf_val = float(r.get("Overall Confidence", 0))
            except: conf_val = 0.0

            job = str(r["_Active_Linked_Job"]) if not is_global and "_Active_Linked_Job" in r else str(r.get("Job_Created", ""))
            if not job or job.lower() in self.MISSING or job == "nan": job = "UnknownJob"
            in_proj, exists, rationale = "Skipped", "Skipped", "LLM Check Bypassed"

            if conf_val >= 0.98:
                rt_action, in_proj, exists, rationale = "✅ KEEP (Harvester/Registry Source - 100% Valid)", "Yes", "Yes", "Deterministic Source - Auto Verified"
                if self.verbosity >= 1: print(f"      🤖 [{i+1}/{len(singletons)}] '{name[:35]}': {rt_action}")
            elif run_llm:
                in_proj, exists, rationale = self._verify_singleton_context_with_llm(r, project_id, project_context, project_source_text=project_source_text)
                if exists == "No": rt_action = "🗑️ DROP (Hallucinated)"
                elif exists == "Yes" and in_proj == "No": rt_action = "✂️ UNLINK / LOST (Out of Scope)"
                else: rt_action = "✅ KEEP (Valid)"
                if self.verbosity >= 1: print(f"      🤖 [{i+1}/{len(singletons)}] '{name[:35]}': {rt_action} -> Exists:{exists}, InProj:{in_proj}")
            else: rt_action = "✅ KEEP (Valid)"
            s_data.append({"id": did, "name": name, "clean_name": clean_name, "job": job, "row": r, "in_proj": in_proj, "exists": exists, "rationale": rationale})

        report_data, actions, display_name = [], {}, project_id.replace("PROJ_", "") if project_id else "Target Project"
        for current in s_data:
            if current["exists"] == "No": actions[current["id"]] = "DROP"
            elif current["exists"] == "Yes" and current["in_proj"] == "No": actions[current["id"]] = "LOST"
            else: actions[current["id"]] = "KEEP"
            best_match_name, best_match_id, best_score = "None", "", 0.0
            for candidate in s_data:
                if current["job"] != candidate["job"]:
                    score = difflib.SequenceMatcher(None, current["clean_name"], candidate["clean_name"]).ratio()
                    if score > best_score: best_score, best_match_name, best_match_id = score, candidate["name"], candidate["id"]
            match_str = f"{best_match_id} ({best_match_name[:20]}) [Sim: {best_score:.2f}]" if best_score > 0 else "No Cross-Job Match"
            report_data.append({"Dataset ID": current["id"], "Name": current["name"][:30], "Job": current["job"], f"In {display_name}?": current["in_proj"], "Real Dataset?": current["exists"], "LLM Rationale": current["rationale"], "Closest Cross-Job Match": match_str})

        if not is_global:
            for job in sorted(list(set(d["Job"] for d in report_data))):
                if self.verbosity >= 1:
                    print(f"\n      🔹 LINKED TO {display_name} BY RUN: {job}")
                    print("      " + tabulate([{k: v for k, v in d.items() if k != "Job"} for d in report_data if d["Job"] == job], headers="keys", tablefmt="simple").replace("\n", "\n      "))
        return actions

    def _find_group_key(self, row, groups):
        for key, members in groups.items():
            is_dup, reason = self.oracle.evaluate_pair(row, members[0])
            if is_dup: return key, reason
        return None, None

    def _merge_into_golden(self, golden, challenger, changes):
        for field in self.DATA_FIELDS:
            val_g, val_c = str(golden.get(field, "")).strip(), str(challenger.get(field, "")).strip()
            conf_f = f"{field} (C)"
            try: c_g, c_c = float(golden.get(conf_f, 0)), float(challenger.get(conf_f, 0))
            except: c_g, c_c = 0, 0

            if c_g == 1.0: continue

            if "URL" in field or "Link" in field:
                if val_c and val_c not in self.MISSING:
                    if val_g and val_g not in self.MISSING:
                        clean_c = re.search(r'=HYPERLINK\("([^"]+)"', val_c, re.IGNORECASE)
                        base_c = clean_c.group(1) if clean_c else val_c
                        if base_c not in val_g:
                            safe_append = base_c.replace('"', '""')
                            golden[field], golden[conf_f] = f"{val_g} & \" | Alt: {safe_append}\"", max(c_g, c_c)
                            changes.append(f"Appended {field}")
                    elif not val_g or val_g in self.MISSING:
                        golden[field], golden[conf_f] = val_c, c_c
                        changes.append(f"Filled {field}")
            elif "Description" in field or "Rationale" in field:
                if val_c and val_c not in self.MISSING:
                    if not val_g or val_g in self.MISSING:
                        golden[field], golden[conf_f] = val_c, c_c
                        changes.append(f"Filled {field}")
                    elif len(val_c) > len(val_g):
                        golden[field], golden[conf_f] = val_c, max(c_g, c_c)
                        changes.append(f"Replaced {field} with longer text")
            else:
                if c_c == 1.0 or ((val_g in self.MISSING) and (val_c not in self.MISSING)) or ((val_c not in self.MISSING) and c_c > c_g):
                    golden[field], golden[conf_f] = val_c, 1.0 if c_c == 1.0 else c_c
                    changes.append(f"Merge {field}")
