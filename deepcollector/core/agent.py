# =============================================================================
# V209: CatalogAgent (Strict CSV Folder Export & Clean Naming Convention)
# =============================================================================
import pandas as pd
import time
import json
import re
import io
import uuid
import requests
import asyncio
import difflib
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from tabulate import tabulate

try:
    import gspread
    from google.colab import auth
    from google.auth import default
except ImportError:
    gspread = None

try:
    from deepcollector.config.settings import AppConfig
    from deepcollector.core.state import CatalogState, CellData
    from deepcollector.tools.research import ResearchTools
    from deepcollector.tools.ddi import DataInspectionTools
    from deepcollector.kb.manager import KnowledgeBaseManager, SheetLock

    try:
        from deepcollector.kb.merger import ProjectMerger
    except ImportError:
        ProjectMerger = None

    from deepcollector.core.rag_engine import RAGEngine
    from deepcollector.harvesting.uci_harvester import UCIHarvester
    from deepcollector.utils.initialization import initialize_apis
    from deepcollector.kb.maintenance import DatasetDoctor
    from deepcollector.tools.deep_research_runner import DeepResearchRunner
    from deepcollector.utils.analytics import PerformanceAnalyzer
    import deepcollector.utils.profiler as profiler_module

except ImportError as e:
    print(f"❌ [Agent] Critical imports failed: {e}")
    AppConfig = object
    CatalogState = object
    ResearchTools = object
    RAGEngine = object
    PerformanceAnalyzer = object
    profiler_module = None
    ProjectMerger = None

class AdvancedKnowledgeProcessor:
    def __init__(self, doc_url: str, tools: Any, target_project_name: str = None):
        self.doc_url = doc_url
        self.tools = tools
        self.target_project_name = target_project_name
        self.html_content = None
        self.references_map = {}
        self.processed_entries = []
        try:
            import bs4
            self.parser = 'lxml' if 'lxml' in globals() else 'html.parser'
            self.BeautifulSoup = bs4.BeautifulSoup
        except ImportError:
            self.BeautifulSoup = None

    def process(self):
        try:
            if not self.BeautifulSoup:
                return []

            if not self._fetch_content():
                return []

            self._extract_references()
            self._extract_and_augment_entries()
            return self.processed_entries
        except Exception as e:
            print(f"      ⚠️ [Knowledge Injection Error] Failed to parse Master Registry: {e}. Skipping injection.")
            return []

    def _fetch_content(self):
        try:
            import requests
            resp = requests.get(self.doc_url, timeout=20)
            if resp.status_code == 200:
                self.html_content = resp.content
                return True
        except Exception:
            pass
        return False

    def _safe_get_text(self, element):
        if not element: return ""
        try:
            if hasattr(element, 'get_text'):
                return element.get_text(strip=True)
            elif hasattr(element, 'text'):
                return str(element.text).strip()
        except Exception:
            pass
        return str(element).strip()

    def _extract_references(self):
        soup = self.BeautifulSoup(self.html_content, self.parser)
        for elem in soup.find_all(['p', 'li']):
            text = self._safe_get_text(elem)
            match = re.match(r'^\[(\d+)\]\s*(.+)', text)
            if match:
                self.references_map[f"[{match.group(1)}]"] = match.group(2)

    def _extract_and_augment_entries(self):
        soup = self.BeautifulSoup(self.html_content, self.parser)
        tables = soup.find_all('table')
        if not tables:
            return

        try:
            main_table = max(tables, key=lambda t: len(t.find_all('tr')))
        except Exception:
            return

        for row in main_table.find_all('tr'):
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 3:
                name = self._safe_get_text(cells[0])
                desc = self._safe_get_text(cells[2])

                if not name or "Name" in name:
                    continue

                if self.target_project_name:
                    n_clean = name.lower().strip()
                    t_clean = self.target_project_name.lower().strip()
                    if n_clean not in t_clean and t_clean not in n_clean:
                        continue

                aug_desc = desc
                citations = [f"{rid}: {rtext}" for rid, rtext in self.references_map.items() if rid in desc]
                if citations:
                    aug_desc += "\n\nReferences:\n" + "\n".join(citations)

                self.processed_entries.append({
                    "Name": name,
                    "Project summary with references": desc,
                    "Augmented Description": aug_desc
                })

class CatalogAgent:
    VERSION = "V209"

    def __init__(self, config: AppConfig, authenticated_gc: Any, keys: Any = None, models: Any = None):
        self.config = config
        self.authenticated_gc = authenticated_gc
        self.verbosity = getattr(config, 'VERBOSITY_LEVEL', 1)

        self.job_id = f"JOB_{uuid.uuid4().hex[:6].upper()}"
        self.stop_workflow_flag = False
        self.deep_research_failed = False

        self.state = CatalogState(
            config,
            getattr(config, 'PROJECT_CONTEXT', ''),
            getattr(config, 'CURRENT_PROJECT_ID', '')
        )

        if keys and models:
            self.keys = keys
            self.models = models
        else:
            if 'initialize_apis' in globals() and initialize_apis:
                self.keys, self.models = initialize_apis(config)
            else:
                self.keys, self.models = None, None

        self.kb_manager = KnowledgeBaseManager(config)
        self.tools = ResearchTools(config, self.keys, self.models)
        self.ddi_tools = DataInspectionTools(config)
        self.rag_engine = RAGEngine(config, self.tools)
        self.doctor = DatasetDoctor(self.kb_manager, self.tools, self.verbosity)

        if ProjectMerger:
            self.merger = ProjectMerger(self.kb_manager, verbosity=self.verbosity, tools=self.tools, models=self.models)
        else:
            self.merger = None

        self.deep_researcher = None
        is_local = getattr(self.config, 'LLM_BACKEND', '') in ["LOCAL_PRO", "LOCAL_CLASSROOM"]
        has_client = getattr(self.models, 'CLIENT', None) is not None

        if has_client or is_local:
            self.deep_researcher = DeepResearchRunner(
                client=getattr(self.models, 'CLIENT', None),
                config=self.config,
                verbosity=self.verbosity,
                tools=self.tools
            )

        self.analyzer = PerformanceAnalyzer()

        method = getattr(config, 'PROJECT_METHOD', 1)
        if method == 2:
            self.harvester = UCIHarvester(config, self.tools)
        else:
            self.harvester = None

        self.MAX_DISCOVERY_ITERATIONS = getattr(config, 'MAX_DISCOVERY_ITERATIONS', 3)
        self.MAX_GROUNDING_ITERATIONS = getattr(config, 'MAX_GROUNDING_ITERATIONS', 5)
        self.MAX_EXTRACTION_ITERATIONS = getattr(config, 'MAX_EXTRACTION_ITERATIONS', 15)
        self.MIN_ASSIGNMENT_CONFIDENCE_GATE = getattr(config, 'MIN_ASSIGNMENT_CONFIDENCE_GATE', 0.70)
        self.CONFIDENCE_THRESHOLD = getattr(config, 'CONFIDENCE_THRESHOLD', 0.80)
        self.GROUNDING_FIELDS = getattr(config, 'GROUNDING_FIELDS', [])
        self.EXTRACTED_FIELDS = getattr(config, 'EXTRACTED_FIELDS', [])
        self.DATA_INSPECTION_ENABLED = getattr(config, 'DATA_INSPECTION_ENABLED', True)

    @staticmethod
    def _track(category):
        if profiler_module and hasattr(profiler_module, 'profiler'):
            return profiler_module.profiler.track(category)
        return lambda f: f

    def execute_workflow(self, mode="AGENT", job_comment="", merge_dry_run=False):
        @self._track("Agent: Workflow Wall-Clock Time")
        def _exec():
            start_time = datetime.now()
            print(f"\n🚀 Starting Job {self.job_id} ({mode}) for: '{self.state.context}'\n" + "="*60)
            if job_comment:
                print(f"📝 Job Comment: {job_comment}")

            job_status = "FAILED"
            try:
                if mode == "MERGE":
                    self.kb_manager.initialize_connection(self.authenticated_gc)
                    if self.merger:
                        self.merger.execute_merge(
                            self.config.CURRENT_PROJECT_ID,
                            self.job_id,
                            dry_run=merge_dry_run,
                            models_verifier=self.models
                        )
                    else:
                        print("❌ Merger module not available.")
                    job_status = "COMPLETED"
                    self._print_llm_summary()
                    return

                elif mode == "REPAIR":
                    self.execute_repair_workflow()

                elif mode == "HARVEST":
                    self.analyzer.start_phase("Bootstrapping", len(self.state.catalog))
                    if getattr(self.config, 'GSPREAD_AVAILABLE', False):
                        self.kb_manager.initialize_connection(self.authenticated_gc)
                        if getattr(self.config, 'WIPE_CURRENT_PROJECT_ONLY', False):
                            self.kb_manager.wipe_project(self.config.CURRENT_PROJECT_ID, self.job_id)
                    self.analyzer.end_phase(len(self.state.catalog))
                    self.execute_harvester_workflow()

                else:
                    self.analyzer.start_phase("Bootstrapping", len(self.state.catalog))
                    success = self.phase_0_bootstrapping()
                    self.analyzer.end_phase(len(self.state.catalog))

                    if not success:
                        print("\n🛑 Bootstrapping failed (Sources inaccessible). Terminating to prevent poor quality results.")
                        job_status = "ABORTED (Source Fetch Failed)"
                        self.stop_workflow_flag = True
                        return
                    else:
                        self.execute_standard_workflow()

                if getattr(self.config, '_CUDA_OOM_ABORT', False):
                    print("\n🛑 Execution stopped gracefully due to insufficient local GPU Memory (CUDA OOM).")
                    job_status = "ABORTED (CUDA OOM)"
                    self.stop_workflow_flag = True
                elif self.stop_workflow_flag:
                    print("\n🛑 Execution stopped to prevent inferior answers (Deep Research or Source Fetch Failed).")
                    if job_status == "FAILED":
                        job_status = "ABORTED (Validation Failed)"

                if not getattr(self.config, '_CUDA_OOM_ABORT', False):
                    if mode != "REPAIR":
                        self.analyzer.start_phase("Phase 3.5: Maintenance", len(self.state.catalog))
                        if self.state.catalog:
                            pre_count = len(self.state.catalog)
                            self.state.catalog = self.doctor.execute_maintenance(self.state.catalog)
                            post_count = len(self.state.catalog)
                            self.analyzer.record_deletions(pre_count - post_count)
                        self.analyzer.end_phase(len(self.state.catalog))

                    if mode == "REPAIR":
                        self.phase_4_repair_write_back()
                    else:
                        self.phase_4_write_back()

                    if getattr(self.config, 'EXPORT_TO_NEW_SHEET', True):
                        self._export_run_data()

                    self.analyzer.print_report()
                    self._print_llm_summary()

                    df_final_report = self.get_catalog_report(full_details=False)
                    if not df_final_report.empty:
                        print("\n" + "="*90)
                        print("📊 FINAL DISCOVERED DATASETS CATALOG")
                        print("="*90)
                        print(tabulate(df_final_report, headers="keys", tablefmt="pipe", showindex=False))

                    job_status = "COMPLETED"

            except Exception as e:
                if getattr(self.config, '_CUDA_OOM_ABORT', False):
                    print(f"\n🛑 [Workflow Interrupted] Graceful halt due to CUDA Out of Memory.")
                    job_status = "ABORTED (CUDA OOM)"
                else:
                    print(f"\n❌ [Workflow Crash] {e}")
                    import traceback
                    print(traceback.format_exc())
                    job_status = f"ERROR: {str(e)[:40]}"
            finally:
                duration = (datetime.now() - start_time).total_seconds()

                if not self.kb_manager.is_connected and getattr(self.config, 'GSPREAD_AVAILABLE', False):
                    try:
                        self.kb_manager.initialize_connection(self.authenticated_gc)
                    except:
                        pass

                if self.kb_manager.is_connected:
                    job_data = {
                        "JobID": self.job_id,
                        "ProjectID": self.config.CURRENT_PROJECT_ID,
                        "Mode": mode,
                        "Start_Time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "End_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Duration_Sec": f"{duration:.2f}",
                        "Status": job_status,
                        "Items_Found": len(self.state.catalog),
                        "Operational_Parameters": self.config.get_operational_report(),
                        "JOB_COMMENT": job_comment
                    }
                    self.kb_manager.log_job_execution(job_data)

                if mode != "MERGE":
                    print(f"\n⏱️ Workflow Wall-Clock Time: {duration:.2f}s")

        return _exec()

    def _format_stats(self, stats):
        count = stats.get('count', 0)
        t = stats.get('time', 0.0)
        t_sq = stats.get('time_sq', 0.0)

        if count > 1:
            mean = t / count
            variance = max(0, (t_sq / count) - (mean ** 2))
            std = math.sqrt(variance)
            return f"{count:>4} calls | {t:>6.2f}s cum. CPU time | Mean: {mean:>5.2f}s | SD: {std:>5.2f}s"
        elif count == 1:
            return f"{count:>4} calls | {t:>6.2f}s cum. CPU time | Mean: {t:>5.2f}s | SD:   N/A "
        return "   0 calls"

    def _print_llm_summary(self):
        print("\n🤖 UNIFIED MODEL USAGE & TIMING STATS (Includes Parallel Cumulative CPU Time):")

        total_gemini_calls = 0
        total_gemma_calls = 0

        print("\n  [RAG & Search Operations]")
        if hasattr(self.tools, 'model_usage_stats') and self.tools.model_usage_stats:
            for model_name, stats in sorted(self.tools.model_usage_stats.items()):
                if stats['count'] > 0:
                    print(f"   {model_name:<35}: {self._format_stats(stats)}")
                    if "gemini" in model_name.lower():
                        total_gemini_calls += stats['count']
                    elif "gemma" in model_name.lower():
                        total_gemma_calls += stats['count']
        else:
            print("   No RAG/Search calls made.")

        print("\n  [Oracle Arbitrations (Merge/Review)]")
        if self.merger and hasattr(self.merger, 'oracle'):
            if hasattr(self.merger.oracle, 'model_stats') and self.merger.oracle.model_stats:
                for model_name, stats in sorted(self.merger.oracle.model_stats.items()):
                    if stats['count'] > 0:
                        print(f"   {model_name:<35}: {self._format_stats(stats)}")
                        if "gemini" in model_name.lower():
                            total_gemini_calls += stats['count']
                        elif "gemma" in model_name.lower():
                            total_gemma_calls += stats['count']
            else:
                print("   No Oracle LLM calls made.")

            print(f"   -------------------------------------------------------------")
            print(f"   Oracle Merges Approved:     {self.merger.oracle.stats['approved']}")
            print(f"   Oracle Merges Rejected:     {self.merger.oracle.stats['rejected']}")
            print(f"   Hard Discriminator Blocks:  {self.merger.oracle.stats['hard_blocks']}")
            print(f"   Rate Limits Hit (Cascaded): {self.merger.oracle.stats['rate_limits_hit']}")
            if self.merger.oracle.stats['errors'] > 0:
                print(f"   🛑 Fatal API Errors:        {self.merger.oracle.stats['errors']}")

        print("\n  [Aggregated LLM Call Totals]")
        print(f"   Total Gemini (Cloud) Calls: {total_gemini_calls}")
        print(f"   Total Gemma (Local) Calls:  {total_gemma_calls}")

    def _checkpoint_save(self, phase_name):
        try:
            df = self.get_catalog_report(full_details=False)
            if not df.empty:
                filename = f"checkpoint_{self.config.CURRENT_PROJECT_ID}_{self.job_id}.csv"
                df.to_csv(filename, index=False)
                if self.verbosity >= 2:
                    print(f"    💾 Auto-Checkpoint saved to {filename} ({phase_name})")
        except Exception:
            pass

    def execute_harvester_workflow(self):
        print(f"\n🚀 HARVESTER WORKFLOW")
        if self.harvester:
            try:
                success = self.harvester.execute_harvest(self.state)
                if not success:
                    print("🛑 [Agent] Harvester failed. Aborting workflow to prevent empty write-backs.")
                    self.stop_workflow_flag = True
            except Exception as e:
                print(f"❌ [Harvester Error] Caught critical failure: {e}")
                self.stop_workflow_flag = True

    def execute_repair_workflow(self):
        print(f"\n🛠️ REPAIR WORKFLOW (Fixing [missing] & low confidence values)")
        self.state.current_phase = "REPAIR"

        self.analyzer.start_phase("Repair: KB Hydration", len(self.state.catalog))
        self.kb_manager.initialize_connection(self.authenticated_gc)
        if not self.kb_manager.read_and_validate_kb():
            print("    ❌ Failed to load KB for repair.")
            return

        self._load_catalog_from_kb(self.config.CURRENT_PROJECT_ID)
        print(f"    📥 Loaded {len(self.state.catalog)} datasets from KB for targeted repair.")
        self.analyzer.end_phase(len(self.state.catalog))

        if not self.state.catalog:
            print("    ✅ No data found requiring repair based on specified thresholds.")
            return

        pre_stats = self.state.capture_confidence_metrics("Pre-Repair")

        urls = getattr(self.config, 'INITIAL_URLS', [])
        if urls:
            loaded_any = False
            for url in urls:
                fetched_data = self.tools.tool_load_url(url)
                if fetched_data:
                    self.state.add_data_and_index(fetched_data)
                    loaded_any = True

            if not loaded_any:
                print("    🛑 [CRITICAL] Failed to load ANY of the provided INITIAL_URLS. Aborting repair to prevent hallucination.")
                self.stop_workflow_flag = True
                return
        else:
             print("    ⚠️ No INITIAL_URLS found. The agent will attempt to search based on dataset names.")

        self.analyzer.start_phase("Phase 2: Grounding (Repair)", len(self.state.catalog))
        self.phase_2_grounding()
        self.analyzer.end_phase(len(self.state.catalog))

        if getattr(self.config, '_CUDA_OOM_ABORT', False):
            return

        self.analyzer.start_phase("Phase 3: Extraction (Repair)", len(self.state.catalog))
        self.phase_3_extraction()
        self.analyzer.end_phase(len(self.state.catalog))

        post_stats = self.state.capture_confidence_metrics("Post-Repair")
        self._print_repair_report(pre_stats, post_stats)

    def _print_repair_report(self, pre, post):
        print("\n" + "="*50)
        print("📊 REPAIR IMPACT REPORT")
        print("="*50)
        print(f"{'Metric':<20} | {'Before':<10} | {'After':<10} | {'Delta':<10}")
        print("-" * 60)

        c_pre = pre.get('completeness', 0.0) * 100
        c_post = post.get('completeness', 0.0) * 100
        print(f"{'Completeness':<20} | {c_pre:.1f}%      | {c_post:.1f}%      | {c_post-c_pre:+.1f}%")

        a_pre = pre.get('avg_conf', 0.0)
        a_post = post.get('avg_conf', 0.0)
        print(f"{'Avg Confidence':<20} | {a_pre:.2f}        | {a_post:.2f}        | {a_post-a_pre:+.2f}")
        print("-" * 60)

    def _load_catalog_from_kb(self, project_id=None):
        df_ds = self.kb_manager.get_kb_data("Datasets")
        if df_ds.empty:
            return

        if project_id and project_id not in ["PROJ_UNKNOWN", "PROJ_GLOBAL_MAINTENANCE", "GLOBAL"]:
            df_links = self.kb_manager.get_kb_data("Project_Dataset_Link")
            if not df_links.empty and "DatasetID" in df_links.columns:
                linked_ds = df_links[df_links["ProjectID"] == project_id]["DatasetID"].tolist()
                df_ds = df_ds[df_ds["DatasetID"].isin(linked_ds)]

        schema_mapping = {
            "Time interval between points": "Frequency",
            "Number of Time Points": "Num Time Points",
            "Number of Locations/Series": "Num Locations/Series",
            "Variables per Location": "Variables per Location",
            "Total Variables": "Total Variables",
            "Primary URL": "Primary URL",
            "Link to Data (Actual Source)": "Link to Data (Actual Source)",
            "Other URL": "Other URL",
            "Detailed Description": "Description",
            "Primary Source Repository": "Primary Creator"
        }

        missing_set = self.kb_manager.MISSING_DATA_PLACEHOLDERS

        for _, row in df_ds.iterrows():
            name = row.get("Variant Name", "")
            if not name or name == "[missing]":
                name = row.get("Canonical Name", "Unknown")

            needs_repair = False
            for c_f, k_f in schema_mapping.items():
                val_str = str(row.get(k_f, "")).strip().lower()

                if val_str in missing_set or val_str in ["nan", "none", ""]:
                    needs_repair = True
                    break

                conf_col = f"{k_f} (C)"
                try:
                    conf = float(str(row.get(conf_col, "0.0")))
                except:
                    conf = 0.0

                threshold = 0.90 if c_f in self.GROUNDING_FIELDS else self.CONFIDENCE_THRESHOLD
                if conf < threshold:
                    needs_repair = True
                    break

            if not needs_repair:
                continue

            item = self.state._initialize_new_item(name)
            for col in self.state.CATALOG_SCHEMA.keys():
                kb_col = schema_mapping.get(col, col)

                val_raw = str(row.get(kb_col, "[missing]")).strip()
                if val_raw.lower() in ["nan", "none", ""]:
                    val_raw = "[missing]"

                conf_col = f"{kb_col} (C)"
                try:
                    conf = float(str(row.get(conf_col, "0.0")))
                except:
                    conf = 0.0

                if val_raw.lower() in missing_set:
                    conf = 0.0
                elif conf == 0.0:
                    conf = 1.0

                item[col] = {"value": val_raw, "confidence": conf, "telemetry_context": "Loaded for Repair", "anchor_ref_id": None}

            item["Dataset Name"] = {"value": name, "confidence": 1.0, "telemetry_context": "KB", "anchor_ref_id": None}
            item["Canonical Name"] = {"value": str(row.get("Canonical Name", name)), "confidence": 1.0, "telemetry_context": "KB", "anchor_ref_id": None}
            item["Type"] = {"value": str(row.get("Type", "Dataset")), "confidence": 1.0, "telemetry_context": "KB", "anchor_ref_id": None}
            item["Assignment Confidence"] = {"value": str(row.get("Overall Confidence", "1.0")), "confidence": 1.0, "telemetry_context": "KB", "anchor_ref_id": None}
            item["DatasetID"] = {"value": str(row.get("DatasetID", "")), "confidence": 1.0, "telemetry_context": "KB", "anchor_ref_id": None}
            item["Aliases"] = {"value": str(row.get("Aliases", "")), "confidence": 1.0, "telemetry_context": "KB", "anchor_ref_id": None}

            self.state.catalog.append(item)

    def _apply_golden_kb_fastpath(self):
        print(f"\n{'='*20} PHASE 1.5: GOLDEN KB FAST-PATH {'='*20}")
        if not getattr(self.config, 'ENABLE_GOLDEN_FASTPATH', True):
            print("    ⏭️ Golden Fast-Path is disabled in config.")
            return

        df_ds = self.kb_manager.get_kb_data("Datasets")
        if df_ds is None or df_ds.empty:
            print("    ℹ️ KB is empty. Skipping fast-path.")
            return

        schema_mapping = {
            "Domain": "Domain",
            "Time interval between points": "Frequency",
            "Number of Time Points": "Num Time Points",
            "Number of Locations/Series": "Num Locations/Series",
            "Variables per Location": "Variables per Location",
            "Total Variables": "Total Variables",
            "Primary URL": "Primary URL",
            "Link to Data (Actual Source)": "Link to Data (Actual Source)",
            "Other URL": "Other URL",
            "Primary Source Repository": "Primary Creator",
            "Detailed Description": "Description",
            "Comments on Data Preparation": "Comments on Data Preparation"
        }

        def norm(n):
            return re.sub(r'[^a-z0-9]', '', str(n).lower())

        kb_name_map = {}
        for idx, row in df_ds.iterrows():
            try:
                conf = float(row.get("Overall Confidence", 0.0))
            except:
                conf = 0.0

            if conf >= 0.85:
                v_name = norm(row.get("Variant Name", ""))
                c_name = norm(row.get("Canonical Name", ""))
                aliases = [norm(x) for x in str(row.get("Aliases", "")).split(",") if x.strip()]

                keys = [v_name, c_name] + aliases
                for k in keys:
                    if k and (k not in kb_name_map or conf > kb_name_map[k]['conf']):
                        row_copy = row.copy()
                        row_copy['_num_conf'] = conf
                        kb_name_map[k] = {'row': row_copy, 'conf': conf}

        fast_path_count = 0
        skipped_fields_total = 0
        for item in self.state.catalog:
            name = item.get("Dataset Name", {}).get("value", "")
            if not name or name == "[missing]":
                continue

            clean_target = norm(name)
            match = kb_name_map.get(clean_target)

            if match:
                row = match['row']
                hydrated = False

                for cat_field in list(self.state.CATALOG_SCHEMA.keys()):
                    if cat_field in ["Dataset Name", "Assignment Confidence", "Assignment Rationale", "Type", "Canonical Name", "Aliases"]:
                        continue

                    kb_col = schema_mapping.get(cat_field, cat_field)
                    val = str(row.get(kb_col, "[missing]")).strip()

                    if val and val.lower() not in ["nan", "none", ""] and val not in self.kb_manager.MISSING_DATA_PLACEHOLDERS:
                        self.state.update_cell_data(name, cat_field, {"value": val, "confidence": 1.0, "telemetry_context": "KB Golden Fast-Path"})
                        hydrated = True
                        skipped_fields_total += 1

                for id_field in ["Canonical Name", "Type"]:
                    curr_val = item.get(id_field, {}).get("value", "[missing]")
                    if curr_val in self.kb_manager.MISSING_DATA_PLACEHOLDERS:
                        kb_val = str(row.get(id_field, "")).strip()
                        if kb_val and kb_val.lower() != "nan" and kb_val not in self.kb_manager.MISSING_DATA_PLACEHOLDERS:
                            self.state.update_cell_data(name, id_field, {"value": kb_val, "confidence": 1.0, "telemetry_context": "KB Golden Fast-Path"})

                if hydrated:
                    fast_path_count += 1
                    if self.verbosity >= 2:
                        print(f"    🌟 Fast-Path Matched: '{name[:30]}' -> KB Entity '{row.get('Variant Name')[:30]}' (Conf: {match['conf']:.2f})")

        print(f"    ⚡ Fast-Patched {fast_path_count} datasets! Pre-filled {skipped_fields_total} fields to skip RAG.")
        self.analyzer.record_cell_change('FILL', skipped_fields_total)

    def execute_standard_workflow(self):
        print(f"\n🚀 STANDARD RAG WORKFLOW (Method 1)")
        self.analyzer.start_phase("Phase 1a: Deep Research", len(self.state.catalog))
        self.phase_1_deep_research()
        self.analyzer.end_phase(len(self.state.catalog))

        if self.stop_workflow_flag:
            print("    🛑 Workflow halted securely at Phase 1a.")
            return

        self.analyzer.start_phase("Phase 1b: Standard Discovery", len(self.state.catalog))
        self.phase_1_standard_loop()
        self.analyzer.end_phase(len(self.state.catalog))

        if self.stop_workflow_flag:
            return

        self.analyzer.start_phase("Phase 1.5: Golden KB Fast-Path", len(self.state.catalog))
        self._apply_golden_kb_fastpath()
        self.analyzer.end_phase(len(self.state.catalog))

        self.analyzer.start_phase("Phase 2: Grounding", len(self.state.catalog))
        self.phase_2_grounding()
        self.analyzer.end_phase(len(self.state.catalog))

        if getattr(self.config, '_CUDA_OOM_ABORT', False):
            return

        self.analyzer.start_phase("Phase 3: Extraction", len(self.state.catalog))
        self.phase_3_extraction()
        self.analyzer.end_phase(len(self.state.catalog))

    def _run_preflight_crawler(self, text_content: str):
        if not text_content or not isinstance(text_content, str):
            return

        print(f"    🕷️ [Pre-Flight Crawler] Scanning context for outbound repositories...")
        found_urls = re.findall(r'(https?://(?:github\.com|huggingface\.co|zenodo\.org|kaggle\.com|archive\.ics\.uci\.edu)[^\s,\">|\)\]]+|https?://[^\s,\">|\)\]]+\.pdf)', text_content)
        unique_urls = list(set([u.rstrip('.,;:') for u in found_urls]))

        if unique_urls:
            print(f"      🔗 Found {len(unique_urls)} secondary URLs.")
            for u in unique_urls[:5]:
                print(f"        🕸️ Fetching & Indexing: {u}")
                self.state.add_data_and_index(self.tools.tool_load_url(u))

    def phase_0_bootstrapping(self) -> bool:
        self.state.current_phase = "BOOTSTRAPPING"
        print("\n=== PHASE 0: BOOTSTRAPPING ===")

        if getattr(self.config, 'GSPREAD_AVAILABLE', False):
            self.kb_manager.initialize_connection(self.authenticated_gc)
            if self.kb_manager.read_and_validate_kb():
                ds_data = self.kb_manager.get_kb_data("Datasets")
                if ds_data is not None and not ds_data.empty:
                    self.state.add_data_and_index(ds_data.to_dict('records'))

        urls = getattr(self.config, 'INITIAL_URLS', [])
        if urls:
            print(f"🌐 Loading {len(urls)} initial URLs...")
            loaded_any = False
            for url in urls:
                fetched_data = self.tools.tool_load_url(url)
                if fetched_data and isinstance(fetched_data, list) and len(fetched_data) > 0:
                    loaded_any = True
                    self.state.add_data_and_index(fetched_data)
                    if getattr(self.config, 'ENABLE_PREFLIGHT_CRAWLER', False):
                        for f_item in fetched_data:
                            if isinstance(f_item, dict):
                                content = f_item.get("content", "")
                                if content:
                                    self._run_preflight_crawler(content)

            if not loaded_any:
                print("    🛑 [CRITICAL] Could not access ANY of the provided initial URLs.")
                return False

        return True

    def phase_1_deep_research(self):
        if not getattr(self.config, 'ENABLE_DEEP_RESEARCH', False):
            return

        print("\n--- Phase 1a: Deep Research (Agentic) ---")
        citation_text = self.state.context

        if hasattr(self.config, 'INITIAL_URLS') and self.config.INITIAL_URLS:
            citation_text += f" (Source URL: {self.config.INITIAL_URLS})"
        else:
            citation_text += " (No specific URL provided, rely on context)"

        print(f"    ℹ️ Deep Research Context: {citation_text[:100]}...")

        required_columns = ["Dataset Name", "Entity Type", "Domain", "Number of Variables", "Number of Time Points", "Time interval", "Primary Source", "Primary Home Page URL", "Link to Data (Actual Source)", "Other URLs", "Detailed Description", "Comments"]

        prompt = (f"Act as a Data Archivist. You are tasked with mapping the SPECIFIC project/collection: '{self.state.context}'.\n"
                  f"There is a collection of Time series datasets described in the following context/citation:\n{citation_text}\n\n"
                  f"YOUR TASK: Produce a Dataset Catalog table describing ONLY the datasets that are officially part of or evaluated in THIS specific project.\n"
                  f"REQUIRED COLUMNS:\n- {', '.join(required_columns)}\n\n"
                  "INSTRUCTIONS:\n"
                  "1. Each dataset should be one row.\n"
                  f"2. RUTHLESS SCOPE RULE: ONLY extract datasets that are the PRIMARY FOCUS of '{self.state.context}'. Absolutely IGNORE datasets that are mentioned as 'related work', 'other competitions', or navigation links (e.g., if target is M1, ignore M3, M4, M5 entirely).\n"
                  "3. Entity Type MUST be exactly one of: 'Dataset', 'Collection', or 'Provider'.\n"
                  "4. Datasets should be specific and never have generic names like Dataset, Dataset name, Time Series.\n"
                  "5. **CONTENT RULE:** Do NOT use generic labels, for any table cells. Use specific names.\n"
                  "6. Be exhaustive for the target project ONLY.\n"
                  "7. VERY IMPORTANT URL RULES: 'Primary Home Page URL' must be the dataset's official site. 'Link to Data (Actual Source)' MUST be the direct data repository (HuggingFace, Zenodo, Kaggle, GitHub) or raw file endpoints (.zip, .csv). 'Other URLs' should hold academic papers (e.g. arXiv). DO NOT use Google Scholar or Vertex AI search tracking links for any of these.\n"
                  "8. IMPORTANT: Use '|||' as a clean separator between columns.\n"
                  "9. **CRITICAL ANTI-LOOP WARNING:** Limit your deep exploration. If you find yourself repeating the same search query or repeating the same chain of thought, IMMEDIATELY HALT EXPLORATION. Break out of the research phase and synthesize your report based strictly on the data you have found up to that point. Do not get stuck in infinite loops."
                  )

        if self.deep_researcher:
            dr_items = self.deep_researcher.execute_research(prompt)
            if not dr_items:
                if getattr(self.config, 'ABORT_ON_DEEP_RESEARCH_FAILURE', True):
                    print("    🛑 [STRICT ABORT] Deep Research failed, timed out, or returned 0 results.")
                    print("    🛑 User Policy Enforcement: Aborting workflow to prevent inferior answers.")
                    self.stop_workflow_flag = True
                else:
                    print("    ⚠️ [FALLBACK ACTIVATED] Deep Research failed or timed out.")
                    print("    ⚠️ User Policy Enforcement: Switching to Rigorous Standard Discovery Mode.")
                    self.deep_research_failed = True
                return

            idx_docs = [i for i in dr_items if isinstance(i, dict) and i.get("is_index_doc")]
            catalog_items = [i for i in dr_items if isinstance(i, dict) and not i.get("is_index_doc")]

            if idx_docs:
                self.state.add_data_and_index(idx_docs)
                print(f"    ✅ Indexed Deep Research Report.")

            if catalog_items:
                print(f"    📥 Bootstrapping {len(catalog_items)} datasets from Deep Research:")
                for item in catalog_items:
                    type_str = item.get("Type", {}).get("value", "Unknown")
                    print(f"      + {item.get('Dataset Name', {}).get('value', 'Unknown')} [{type_str}]")
                self.state.update_catalog_batch(catalog_items, allow_new_datasets=True)
                self.analyzer.record_cell_change('FILL', len(catalog_items) * 8)

    def phase_1_standard_loop(self):
        print(f"\n--- Phase 1b: Standard Discovery Loop ---")

        if getattr(self, 'deep_research_failed', False):
            self.MAX_DISCOVERY_ITERATIONS = max(self.MAX_DISCOVERY_ITERATIONS, 4)
            self.config.SEARCH_NUM_RESULTS = max(getattr(self.config, 'SEARCH_NUM_RESULTS', 8), 12)
            print(f"\n🔥 [RIGOROUS MODE] Increasing Discovery Iterations to {self.MAX_DISCOVERY_ITERATIONS} and Search Limit to {self.config.SEARCH_NUM_RESULTS}")

        self.state.iteration = 0

        # Access KI URL dynamically from config
        ki_url = getattr(self.config, 'KNOWLEDGE_MASTER_DOC_URL', None)
        current_proj_name = getattr(self.config, 'CURRENT_PROJECT_NAME', None)

        if ki_url and current_proj_name:
            print(f"\n--- Discovery Stage 0: Knowledge Injection (Master Registry) ---")
            processor = AdvancedKnowledgeProcessor(ki_url, self.tools, target_project_name=current_proj_name)
            ki_data = processor.process()
            if ki_data and isinstance(ki_data, list) and len(ki_data) > 0:
                idx_data = [{"url": "KI_DOC_MASTER", "content": d['Augmented Description'], "title": d['Name'], "type": "Knowledge Injection"} for d in ki_data if isinstance(d, dict)]
                self.state.add_data_and_index(idx_data)
                print(f"    ✅ Found and Indexed match: {ki_data[0].get('Name', 'Unknown')}")
        else:
             if self.verbosity >= 1:
                 print("    ℹ️  Skipping Knowledge Injection (Missing Master URL or Project Name).")

        while self.state.iteration < self.MAX_DISCOVERY_ITERATIONS:
            self.state.iteration += 1
            print(f"\n--- Discovery Iteration {self.state.iteration}/{self.MAX_DISCOVERY_ITERATIONS} ---")

            added = self.rag_engine.discover_datasets_from_index(self.state)
            self.analyzer.record_cell_change('FILL', added * 2)

            search_plan = self.rag_engine.plan_discovery_search(self.state)
            actions_taken = False
            if search_plan:
                print(f"🧠 [Planner] Generating search strategy...")
                for action in search_plan:
                    if not isinstance(action, dict): continue
                    q = action.get("query")
                    if q:
                        print(f"🛠️ Executing Search: {q}")
                        limit = getattr(self.config, 'SEARCH_NUM_RESULTS', 8)
                        results = self.tools.tool_search_and_fetch(q, num_results=limit)
                        if results:
                            self.state.add_data_and_index(results)
                            actions_taken = True

            if not actions_taken and self.state.iteration > 1:
                print("💡 No new actions. Stopping Discovery.")
                break

        self._validate_discovered_entities(self.state)
        self._apply_relevance_gate()
        self.state.capture_confidence_metrics("End of Phase 1")

    def _validate_discovered_entities(self, state):
        print(f"\n--- Entity Validation (Collection Awareness & Naming) ---")

        leaf_regex = [
            r"\bett[hm]\d\b", r"\bpems[-_ ]?(sf|bay|0)\b", r"san francisco traffic",
            r"traffic \(pems\)", r"\becl\b", r"electricity consuming load",
            r"exchange rate", r"illness", r"weather \(mpi", r"taxi", r"solar"
        ]

        exact_collections = {"pems", "ett", "monash", "utsd", "tslib", "m3", "m4", "m5", "gefcom", "lotsa", "timebench", "chronos", "gluonts", "timeseriesclassification"}
        collection_keywords = ["archive", "repository", "library", "collection", "benchmark", "corpus", "unified time series", "benchmark suite", "competition", "dataset suite"]
        provider_keywords = ["ecmwf", "bank", "reserve", "authority", "commission", "physionet", "kaggle", "caltrans", "nrel", "noaa", "google", "microsoft", "uci"]

        current_project_name = getattr(self.config, 'CURRENT_PROJECT_NAME', "").lower().strip()
        context_str = state.context.lower()
        subject_match = re.match(r"^([^:-]+)", context_str)
        context_subject = subject_match.group(1).strip() if subject_match else ""

        self_ref_terms = set()
        if current_project_name:
            self_ref_terms.add(current_project_name)
        if context_subject:
            self_ref_terms.add(context_subject)

        self_ref_count = 0

        for item in state.catalog:
            name_obj = item.get("Dataset Name", {})
            name = name_obj.get("value", "") if isinstance(name_obj, dict) else str(name_obj)
            name_lower = name.lower().strip()

            curr_type = state.get_cell_data(name, "Type").get("value", "Dataset")

            is_collection = False
            is_provider = False
            is_leaf = False

            if any(re.search(p, name_lower) for p in leaf_regex):
                is_leaf = True
            elif name_lower in exact_collections:
                is_collection = True
            else:
                if any(k in name_lower for k in collection_keywords):
                    is_collection = True
                elif any(k in name_lower for k in provider_keywords):
                    is_provider = True

            if "pems" in name_lower and not is_collection:
                is_leaf = True
            if "ett" in name_lower and not is_collection:
                is_leaf = True
            if re.search(r"\bm\b", name_lower) and not is_collection:
                if len(name_lower) > 4:
                    is_leaf = True

            if is_leaf:
                if curr_type != "Dataset":
                    state.update_cell_data(name, "Type", {"value": "Dataset", "confidence": 0.99})
            elif is_collection:
                if curr_type != "Collection":
                    state.update_cell_data(name, "Type", {"value": "Collection", "confidence": 0.95})
            elif is_provider:
                if curr_type != "Provider":
                    state.update_cell_data(name, "Type", {"value": "Provider", "confidence": 0.95})

            is_self_ref = False
            if name_lower in self_ref_terms:
                is_self_ref = True
            else:
                for term in self_ref_terms:
                    if name_lower == f"{term} dataset" or (len(term) > 3 and difflib.SequenceMatcher(None, name_lower, term).ratio() > 0.9):
                        is_self_ref = True
                        break

            if is_self_ref:
                state.update_cell_data(name, "Assignment Confidence", {"value": "0.0", "confidence": 1.0})
                self_ref_count += 1
                continue

        print(f"    📊 Classification complete. Validated Types and Deprecated {self_ref_count} items.")

    def _apply_relevance_gate(self):
        print(f"\n--- Relevance Gate (Threshold: {self.MIN_ASSIGNMENT_CONFIDENCE_GATE}) ---")
        kept = []
        for item in self.state.catalog:
            try:
                val = float(self.state.get_cell_data(None, "Assignment Confidence", item).get("value", 0))
            except:
                val = 0.0

            if val >= self.MIN_ASSIGNMENT_CONFIDENCE_GATE:
                kept.append(item)

        self.state.catalog = kept
        print(f"    📉 Gate applied. Retained {len(kept)} items.")

    def phase_2_grounding(self):
        self.state.current_phase = "GROUNDING"
        self.state.iteration = 0
        print(f"\n{'='*20} PHASE 2: GROUNDING {'='*20}")
        current_gaps = self._audit_catalog(threshold=0.90, specific_fields=self.GROUNDING_FIELDS)

        while self.state.iteration < self.MAX_GROUNDING_ITERATIONS:
            if not current_gaps:
                break

            self.state.iteration += 1
            print(f"\n--- Grounding Iteration {self.state.iteration} ---")

            fills, refs, confs = self.rag_engine.execute_cellular_rag(self.state, self.GROUNDING_FIELDS)
            self.analyzer.record_cell_change('FILL', fills)
            self.analyzer.record_cell_change('REFINE', refs)
            self.analyzer.record_cell_change('CONFIRM', confs)

            if self.DATA_INSPECTION_ENABLED:
                self._execute_ddi_sweep(current_gaps)

            current_gaps = self._audit_catalog(threshold=0.90, specific_fields=self.GROUNDING_FIELDS)

            if current_gaps:
                self._plan_and_execute_extraction_search(current_gaps)

            self._checkpoint_save(f"Grounding_Iter_{self.state.iteration}")

    def phase_3_extraction(self):
        self.state.current_phase = "EXTRACTION"
        self.state.iteration = 0
        print(f"\n{'='*20} PHASE 3: EXTRACTION {'='*20}")

        fields = [f for f in self.EXTRACTED_FIELDS if f not in self.GROUNDING_FIELDS]
        self._lock_collections_fields(fields)
        current_gaps = self._audit_catalog(threshold=self.CONFIDENCE_THRESHOLD, specific_fields=fields, skip_collections=True)

        stall_counter = 0
        prev_gap_count = len(current_gaps)
        STALL_THRESHOLD = 3

        while self.state.iteration < self.MAX_EXTRACTION_ITERATIONS:
            if not current_gaps:
                break

            self.state.iteration += 1
            print(f"\n--- Extraction Iteration {self.state.iteration} ---")

            fills, refs, confs = self.rag_engine.execute_cellular_rag(self.state, fields)
            self.analyzer.record_cell_change('FILL', fills)
            self.analyzer.record_cell_change('REFINE', refs)
            self.analyzer.record_cell_change('CONFIRM', confs)

            if current_gaps:
                self._plan_and_execute_extraction_search(current_gaps)

            self._checkpoint_save(f"Extraction_Iter_{self.state.iteration}")

            current_gaps = self._audit_catalog(threshold=self.CONFIDENCE_THRESHOLD, specific_fields=fields, skip_collections=True)
            gap_reduction = prev_gap_count - len(current_gaps)

            if gap_reduction < 2:
                stall_counter += 1
            else:
                stall_counter = 0

            prev_gap_count = len(current_gaps)

            if stall_counter >= STALL_THRESHOLD:
                print("🛑 [Early Termination] Extraction stalled. Stopping.")
                break

    def _lock_collections_fields(self, fields):
        dim_fields = {
            "Number of Time Points", "Number of Locations/Series",
            "Variables per Location", "Total Variables", "Time interval between points"
        }
        fields_to_lock = [f for f in fields if f in dim_fields]

        for item in self.state.catalog:
            if item.get("Type", {}).get("value") in ["Collection", "Provider"]:
                for f in fields_to_lock:
                    if self.state.get_cell_data(None, f, item).get("value") == "[missing]":
                        self.state.update_cell_data(item["Dataset Name"]["value"], f, {"value": "[Skipped]", "confidence": 1.0})

    def phase_4_write_back(self):
        print(f"\n{'='*20} PHASE 4: WRITE-BACK {'='*20}")
        if not self.kb_manager.is_connected:
            print("⚠️ [Write-back] KB Manager not connected.")
            return

        try:
            self.kb_manager.reconcile_and_write(
                self.state,
                getattr(self.config, 'CURRENT_PROJECT_NAME', 'Unknown Project'),
                job_id=self.job_id
            )
        except Exception as e:
            print(f"❌ [Write-back Error] {e}")

    def phase_4_repair_write_back(self):
        print(f"\n{'='*20} PHASE 4: REPAIR WRITE-BACK {'='*20}")
        if not self.kb_manager.is_connected:
            print("⚠️ [Write-back] KB Manager not connected.")
            return

        from deepcollector.kb.manager import SheetLock
        lock = SheetLock(self.kb_manager, self.job_id, self.verbosity)

        if not lock.acquire(timeout_seconds=300):
            print("    ❌ Failed to acquire KB Lock for repair write-back.")
            return

        try:
            if not self.kb_manager.read_and_validate_kb():
                raise RuntimeError("Failed to refresh KB.")

            df_ds = self.kb_manager.get_kb_data("Datasets")
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            updates = 0
            for item in self.state.catalog:
                name = item.get("Dataset Name", {}).get("value")
                if not name or name == "[missing]":
                    continue

                record, field_confs, _ = self.kb_manager._normalize_dataset_entry(self.state, item, time_str)

                mask = (df_ds["Variant Name"].astype(str).str.strip().str.lower() == str(name).strip().lower()) | \
                       (df_ds["Canonical Name"].astype(str).str.strip().str.lower() == str(name).strip().lower())

                if mask.any():
                    idx = mask.idxmax()
                    row_updated = False

                    for field_col, conf_col in self.kb_manager.FIELD_CONFIDENCE_MAP.items():
                        new_val = record.get(field_col, "")
                        new_conf = field_confs.get(field_col, 0.0)

                        try:
                            old_conf = float(df_ds.loc[idx, conf_col])
                        except:
                            old_conf = 0.0

                        old_val_str = str(df_ds.loc[idx, field_col]).strip().lower()
                        is_missing = old_val_str in self.kb_manager.MISSING_DATA_PLACEHOLDERS or old_val_str in ["nan", "none", ""]

                        if (new_conf > old_conf or is_missing) and new_val not in self.kb_manager.MISSING_DATA_PLACEHOLDERS:
                            df_ds.loc[idx, field_col] = new_val
                            df_ds.loc[idx, conf_col] = str(new_conf)
                            row_updated = True

                    if len(str(record.get("Description", ""))) > len(str(df_ds.loc[idx, "Description"])) + 10:
                        df_ds.loc[idx, "Description"] = record["Description"]
                        row_updated = True

                    if row_updated:
                        df_ds.loc[idx, "Job_Updated"] = self.job_id
                        df_ds.loc[idx, "Date_Updated"] = time_str
                        df_ds.loc[idx, "Project_Updated"] = self.config.CURRENT_PROJECT_ID
                        updates += 1

            if updates > 0:
                self.kb_manager.write_dataframe_to_tab("Datasets", df_ds)
                print(f"    ✅ Repair Write-back complete. Updated {updates} rows in 'Datasets' tab.")
            else:
                print("    ℹ️ No updates required in KB.")

        except Exception as e:
            print(f"❌ [Write-back Error] {e}")
        finally:
            lock.release()

    # =========================================================================
    # CRITICAL EXPORT OVERRIDE: Strict CSV Export
    # Replaces standalone Google Sheet creation with a direct Google Drive
    # API upload to the specified target folder dynamically via AppConfig.
    # =========================================================================
    def _export_run_data(self):
        print(f"\n{'='*20} RUN EXPORT {'='*20}")
        try:
            df = self.get_catalog_report(full_details=True)
            if df.empty:
                print("    ℹ️ No data to export.")
                return

            # Access the globally configured folder ID safely (This is for the DATA)
            folder_id = getattr(self.config, 'GOOGLE_DRIVE_SHEET_FOLDER_ID', None)
            if not folder_id:
                print("    ⚠️ Export Folder ID not configured in settings. Skipping Drive upload.")
                return

            import numpy as np
            # Python 3.11 Safe Infinity & NaN stripping
            df = df.replace([np.inf, -np.inf], np.nan).fillna("")
            df = df.astype(str)
            df = df.replace(r'(?i)^(nan|inf|-inf|none|<na>)$', '', regex=True)

            # -------------------------------------------------------------
            # Enforce Naming Convention: ProjectName_YYYYMMDD_HHMM_JobID.csv
            # -------------------------------------------------------------
            proj_name = str(getattr(self.config, 'CURRENT_PROJECT_NAME', 'UNKNOWN'))
            safe_proj_name = re.sub(r'[^A-Za-z0-9_\-]', '_', proj_name).strip('_')
            timestamp = time.strftime('%Y%m%d_%H%M')
            filename = f"{safe_proj_name}_{timestamp}_{self.job_id}.csv"

            # Save Locally
            df.to_csv(filename, index=False)
            print(f"🎉 [Export] Data saved to local CSV: {filename}")

            # Upload to Google Drive natively
            if hasattr(self, 'authenticated_gc') and self.authenticated_gc:
                try:
                    from googleapiclient.discovery import build
                    from google.auth import default
                    from googleapiclient.http import MediaFileUpload

                    print(f"    ☁️ Uploading {filename} to target Google Drive Folder...")

                    creds, _ = default()
                    drive_service = build('drive', 'v3', credentials=creds)

                    file_metadata = {
                        'name': filename,
                        'parents': [folder_id]
                    }

                    # Force text/csv to prevent Google Drive from converting to a Sheet
                    media = MediaFileUpload(filename, mimetype='text/csv')
                    file = drive_service.files().create(
                        body=file_metadata,
                        media_body=media,
                        fields='id'
                    ).execute()

                    print(f"    ✅ Success! CSV cleanly uploaded to Drive with ID: {file.get('id')}")
                except ImportError:
                    print("    ⚠️ googleapiclient not installed. Skipping Drive upload.")
                except Exception as e:
                    print(f"    ❌ Failed to upload to Google Drive folder: {e}")

        except Exception as e:
            print(f"❌ [Export Error] Failed: {e}")
            import traceback
            print(traceback.format_exc())

    def _audit_catalog(self, threshold, specific_fields, skip_collections=False):
        gaps = []
        fields = specific_fields or self.EXTRACTED_FIELDS
        for item in self.state.catalog:
            if skip_collections and item.get("Type", {}).get("value") in ["Collection", "Provider"]:
                continue
            try:
                if float(item.get("Assignment Confidence", {}).get("value", 0)) <= 0.0:
                    continue
            except:
                pass
            name = item["Dataset Name"]["value"]
            for f in fields:
                data = self.state.get_cell_data(name, f)
                if data["confidence"] < threshold:
                    gaps.append({"Dataset": name, "Field": f, "Confidence": data["confidence"]})
        return gaps

    def _execute_ddi_sweep(self, gaps):
        print("    🔍 Running DDI Sweep...")
        inspected = set()
        for gap in gaps:
            name = gap['Dataset']
            if name in inspected:
                continue
            url = self.state.get_cell_data(name, "Link to Data (Actual Source)").get("value")
            if url and (url.lower().endswith(('.csv', '.txt', '.zip', '.npz')) or "raw.githubusercontent" in url.lower()):
                res = self.tools.tool_inspect_data_file(url)
                if res.get("status") == "success":
                    self.state.update_cell_data(name, "Variables per Location", {"value": str(res.get("column_count")), "confidence": 1.0})
                    self.analyzer.record_cell_change('FILL', 1)
                    inspected.add(name)

    def _plan_and_execute_extraction_search(self, gaps):
        ds_gaps = {}
        for g in gaps:
            ds_gaps.setdefault(g['Dataset'], []).append(g['Field'])
        for ds, fields in list(ds_gaps.items())[:15]:
            clean_fields = [f.replace(" (C)", "") for f in fields]
            eff_name = self.state.get_effective_name(ds)
            field_str = str(clean_fields).lower()
            if "url" in field_str or "source" in field_str or "link" in field_str:
                query = f"official download url repository github dataset '{eff_name}'"
            else:
                query = f"{eff_name} dataset {', '.join(clean_fields)}"
            print(f"🛠️ Executing Search: {query}")
            res = self.tools.tool_search_and_fetch(query, num_results=getattr(self.config, 'SEARCH_NUM_RESULTS', 8))
            if res:
                self.state.add_data_and_index(res)

    def get_catalog_report(self, full_details=False):
        if not self.state.catalog:
            return pd.DataFrame()
        data = []
        for item in self.state.catalog:
            try:
                if float(item.get("Assignment Confidence", {}).get("value", 0)) <= 0.0:
                    continue
            except:
                pass
            row = {}
            high_conf = 0
            total = 0
            item_type = item.get("Type", {}).get("value", "[missing]")
            if item_type == "[missing]":
                row["Type"] = "Dataset"
            else:
                row["Type"] = item_type
            for k, v in item.items():
                if not k.startswith("_"):
                    val = v.get("value", "[missing]")
                    conf = v.get("confidence", 0.0)
                    if k == "Type" and val == "[missing]":
                        val = "Dataset"
                    row[k] = val
                    if conf >= 0.80:
                        high_conf += 1
                    total += 1
                    if full_details:
                        row[f"{k} (Telemetry)"] = v.get("telemetry_context", "")
            row["Completeness (High Conf %)"] = f"{(high_conf/total)*100:.1f}%" if total else "0%"
            data.append(row)
        return pd.DataFrame(data)

print("✅ deepcollector/core/agent.py written (Strict CSV Folder Export & Clean Naming Convention Fix).")