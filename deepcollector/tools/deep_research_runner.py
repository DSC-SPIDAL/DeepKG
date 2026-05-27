# =============================================================================
# V23: Deep Research Runner (Configurable Agent Model & Timeout Protection)
# =============================================================================
import time
import json
import re
import warnings
import pandas as pd
import io
import csv
from typing import List, Dict, Any, Optional

try:
    from deepcollector.utils.profiler import profiler
except ImportError:
    class DummyProfiler:
        def track(self, c): return lambda f: f
    profiler = DummyProfiler()

class DeepResearchRunner:
    def __init__(self, client: Any, config: Any = None, verbosity: int = 1, tools: Any = None):
        self.client = client
        self.config = config
        self.verbosity = verbosity
        self.tools = tools

        # Pull model name from config, fallback to default if missing
        self.AGENT_NAME = getattr(self.config, 'DEEP_RESEARCH_AGENT_MODEL', "deep-research-pro-preview-12-2025") if self.config else "deep-research-pro-preview-12-2025"

        warnings.filterwarnings("ignore", category=UserWarning, module="google.genai.interactions")

    @profiler.track("Tool: Deep Research Agent")
    def execute_research(self, prompt: str) -> List[Dict[str, Any]]:
        is_local = getattr(self.config, 'LLM_BACKEND', '') in ["LOCAL_PRO", "LOCAL_CLASSROOM"]

        # --- LOCAL GEMMA DEEP RESEARCH ---
        if is_local and getattr(self.config, 'ENABLE_LOCAL_DEEP_RESEARCH', True) and self.tools:
            return self._execute_local_research(prompt)

        # --- GOOGLE CLOUD DEEP RESEARCH ---
        if not self.client:
            if self.verbosity >= 1: print("    ❌ [Deep Research] Client not initialized.")
            return []

        max_retries = getattr(self.config, 'DEEP_RESEARCH_MAX_RETRIES', 6) if self.config else 6
        force_async = getattr(self.config, 'FORCE_ASYNC_POLLING', False) if self.config else False
        poll_interval = getattr(self.config, 'DEEP_RESEARCH_POLLING_INTERVAL_SECONDS', 15) if self.config else 15

        timeout_mins = 60
        max_wait_time = 3600

        for attempt in range(1, max_retries + 1):
            print(f"\n🧠 [Deep Research] Submitting job to '{self.AGENT_NAME}' (Attempt {attempt}/{max_retries}): '{prompt[:60]}...'")

            task_id = None
            final_report_text = ""
            last_log_count = 0
            t0_deep = time.time()
            attempt_failed = False

            try:
                if force_async:
                    print("    ⚡ Forcing Async Polling (Bypassing fragile stream)...")
                    response = self.client.interactions.create(
                        agent=self.AGENT_NAME,
                        input=prompt,
                        agent_config={
                            "type": "deep-research",
                            "thinking_summaries": "auto"
                        },
                        background=True,
                        stream=False
                    )
                    task_id = response.id
                    print(f"    🚀 Task ID: {task_id} | Status: BACKGROUND INITIATED")
                else:
                    response_stream = self.client.interactions.create(
                        agent=self.AGENT_NAME,
                        input=prompt,
                        agent_config={
                            "type": "deep-research",
                            "thinking_summaries": "auto"
                        },
                        background=True,
                        stream=True
                    )

                    for chunk in response_stream:
                        if hasattr(chunk, 'interaction') and chunk.interaction:
                            if not task_id:
                                task_id = chunk.interaction.id
                                print(f"    🚀 Task ID: {task_id} | Status: STREAMING")

                        if hasattr(chunk, 'logs') and chunk.logs:
                            for log in chunk.logs:
                                last_log_count += 1
                                if self.verbosity >= 1:
                                    print(f"    🔎 {log.message}")

                        if hasattr(chunk, 'delta') and chunk.delta:
                            delta = chunk.delta
                            dtype = getattr(delta, 'type', '')
                            content = getattr(delta, 'content', '')
                            text_val = getattr(content, 'text', str(content)) if content else getattr(delta, 'text', '')

                            if dtype == "thought_summary" and text_val:
                                if self.verbosity >= 1:
                                    clean_thought = text_val.replace('\n', ' ').strip()
                                    print(f"    🧠 Thinking: {clean_thought}")
                            elif dtype == "text" and text_val:
                                final_report_text += text_val

            except Exception as e:
                print(f"\n    ⚠️ Stream connection closed or interrupted ({e}). Switching to Async Monitor...")

            if not task_id:
                print(f"\n    ❌ [Deep Research] Failed to initialize task ID on Google's end.")
                attempt_failed = True
            else:
                print(f"    ⏳ Monitoring Task ID: {task_id} (Max wait: {timeout_mins}m)")

                print("    ", end="", flush=True)

                while True:
                    elapsed = time.time() - t0_deep
                    if elapsed > max_wait_time:
                        print(f"\n    🛑 [TIMEOUT] Job hung for over {timeout_mins} minutes.")
                        attempt_failed = True
                        break

                    try:
                        job = self.client.interactions.get(id=task_id)
                        status = str(job.status).upper()

                        print(".", end="", flush=True)

                        if hasattr(job, 'logs') and job.logs:
                            current_logs = job.logs
                            if len(current_logs) > last_log_count:
                                print()
                                for log in current_logs[last_log_count:]:
                                    if self.verbosity >= 1:
                                        print(f"    🔎 (Async) {log.message}")
                                last_log_count = len(current_logs)
                                print("    ", end="", flush=True)

                        if status == "COMPLETED":
                            print(f"\n    ✅ [Deep Research] Job Completed in {int(elapsed)}s on Attempt {attempt}.")

                            if not final_report_text and hasattr(job, 'outputs'):
                                for out in job.outputs:
                                    if getattr(out, 'type', '') == 'text':
                                        final_report_text += getattr(out, 'text', '')

                            class MockInteraction:
                                pass
                            mock_int = MockInteraction()

                            class MockOutput:
                                def __init__(self, text):
                                    self.text = text
                            mock_int.outputs = [MockOutput(final_report_text)]

                            return self._parse_output_to_catalog(mock_int)

                        elif status in ["FAILED", "CANCELLED"]:
                            print(f"\n    ❌ [Deep Research] Job instantly failed ({status}) on attempt {attempt}.")
                            attempt_failed = True
                            break

                        time.sleep(poll_interval)

                    except Exception as poll_err:
                        print("?", end="", flush=True)
                        time.sleep(poll_interval)

            if attempt_failed and attempt < max_retries:
                print(f"    🔄 Brute-force retry in 10 seconds...")
                time.sleep(60)

        print(f"\n    🛑 Exhausted all {max_retries} Deep Research attempts. Giving up.")
        return []

    def _execute_local_research(self, prompt: str) -> List[Dict[str, Any]]:
        print(f"\n🧠 [Local Deep Research] Initiating Gemma-based Agentic Research Loop...")
        query_prompt = (
            "You are the strategic planner for a data cataloging agent. "
            "Based on the task below, generate exactly 3 highly specific Google Search queries "
            "to find the official datasets and repositories.\n\n"
            f"TASK:\n{prompt}\n\n"
            "Return ONLY a JSON list of 3 strings. Do not use markdown blocks."
        )
        print("    🧠 Generating search strategy...")
        try:
            response = self.tools.generate_content_standard("LOCAL", query_prompt)
            if response.text == "[missing]":
                print("    ⚠️ Search strategy failed due to local LLM context limits.")
                queries = []
            else:
                queries = self.tools._extract_json_robustly(response.text)
        except Exception as e:
            if self.verbosity >= 2: print(f"    ⚠️ Strategy generation failed: {e}")
            queries = []

        if not isinstance(queries, list) or len(queries) == 0:
            context = getattr(self.config, 'PROJECT_CONTEXT', 'Target Project')
            queries = [f"{context} official dataset time series", f"{context} benchmark datasets", f"{context} repository github data"]

        aggregated_context = ""
        index_docs = []
        visited_urls = set()

        for q in queries[:3]:
            if not isinstance(q, str): continue
            print(f"    🔎 Searching: {q}")
            results = self.tools.tool_search_and_fetch(q, num_results=3)
            for res in results:
                url = res.get("url", "")
                if not url or url in visited_urls or "youtube.com" in url.lower(): continue
                visited_urls.add(url)
                content = res.get("content", "")
                title = res.get("title", "")
                print(f"    🕸️ Fetching: {url}")
                page_data = self.tools.tool_load_url(url)
                if page_data and len(page_data) > 0: content = page_data[0].get("content", content)
                aggregated_context += f"--- Source: {url} ---\n{content[:4000]}\n\n"
                index_docs.append({"title": title, "url": url, "content": content, "type": "Local Deep Research Grounding", "is_index_doc": True})

        if not aggregated_context.strip(): print("    ⚠️ No web context gathered. Falling back to zero-shot extraction.")
        print("    🧠 Synthesizing findings into Catalog format...")

        is_pro = self.config.LLM_BACKEND == "LOCAL_PRO"
        safe_context_len = 10000 if is_pro else 6000
        safe_context = aggregated_context[:safe_context_len]

        extraction_prompt = (
            f"TASK: {prompt}\n\n"
            "INSTRUCTIONS:\n"
            "You are extracting discovered_datasets. "
            "Based strictly on the gathered context below, extract the datasets into a JSON format. "
            "Format EXACTLY as: {\"discovered_datasets\": [{\"dataset_name\": \"Name\", \"Type\": \"Dataset\", \"Domain\": \"Finance\", \"Total Variables\": \"5\", \"Number of Time Points\": \"100\", \"Time interval between points\": \"1 hour\", \"Primary Source Repository\": \"Github\", \"Primary URL\": \"http...\", \"Link to Data (Actual Source)\": \"http...\", \"Other URL\": \"\", \"Detailed Description\": \"Desc\"}]} "
            "Output ONLY valid JSON.\n\n"
            f"=== GATHERED WEB CONTEXT ===\n"
            f"{safe_context}\n"
            f"============================\n"
        )

        try:
            final_response = self.tools.generate_content_standard("LOCAL", extraction_prompt, max_new_tokens=2048)
            if final_response.text == "[missing]":
                print("    ⚠️ Local Deep Research Extraction aborted due to LLM Context OOM or Safety Halt.")
                return index_docs

            data = self.tools._extract_json_robustly(final_response.text)
            parsed_items = []
            datasets = data.get("discovered_datasets", []) if isinstance(data, dict) else []
            if not datasets and isinstance(data, list): datasets = data

            valid_items = 0
            for d in datasets:
                if not isinstance(d, dict): continue
                raw_name = d.get("dataset_name", d.get("Dataset Name", ""))
                if not raw_name: continue

                item = {"Dataset Name": {"value": raw_name, "confidence": 0.80, "telemetry_context": "Local Deep Research"}}
                for col in ["Type", "Domain", "Total Variables", "Number of Time Points", "Time interval between points", "Primary Source Repository", "Primary URL", "Link to Data (Actual Source)", "Other URL", "Detailed Description"]:
                    val = d.get(col, "")
                    if val: item[col] = {"value": str(val), "confidence": 0.80, "telemetry_context": "Local Deep Research"}

                item["Assignment Confidence"] = {"value": "0.85", "confidence": 1.0}
                item["Assignment Rationale"] = {"value": "Identified by Local Gemma Deep Research Loop", "confidence": 1.0}

                parsed_items.append(item)
                valid_items += 1

            print(f"    🧠 Successfully parsed {valid_items} datasets from Local Deep Research.")
            if index_docs:
                for idx_doc in reversed(index_docs): parsed_items.insert(0, idx_doc)
            return parsed_items
        except Exception as e:
            print(f"    ⚠️ Local Deep Research Extraction Failed: {e}")
            return index_docs

    def _clean_text(self, text: str) -> str:
        if not text: return ""
        clean = text.replace("**", "").replace("__", "").strip("*").strip("_").strip()
        patterns = [
            r'\s*\((?:.*\bUTSD\b.*)\)', r'\s*\((?:.*\bTimer\b.*)\)',
            r'\s*\(Component\)', r'\s*\(Subset\)', r'\s*\(Collection\)'
        ]
        for p in patterns:
            clean = re.sub(p, '', clean, flags=re.IGNORECASE).strip()
        return clean

    def _clean_url_field(self, text: str) -> str:
        if not text: return ""
        import re
        md_links = re.findall(r'\[.*?\]\((https?://.*?)\)', text)
        raw_links = re.findall(r'(?<!\()(https?://[^\s>\"\'\)]+)', text)

        all_links = []
        if md_links:
            all_links.extend([m.strip() for m in md_links])

        for link in raw_links:
            clean_link = link.rstrip('.,;:')
            if clean_link not in all_links:
                all_links.append(clean_link)

        return ", ".join(all_links) if all_links else text.strip()

    def _parse_output_to_catalog(self, interaction: Any) -> List[Dict[str, Any]]:
        if not interaction.outputs: return []
        text = interaction.outputs[-1].text

        meta_item = {
            "title": "Deep Research Report", "url": "DEEP_RESEARCH_API",
            "content": text, "type": "Deep Research Report", "is_index_doc": True
        }
        parsed_items = [meta_item]

        try:
            lines = text.split('\n')
            table_lines = []
            capture = False

            for line in lines:
                if "Dataset Name" in line and "|" in line:
                    capture = True
                    line = line.replace("|||", "|")

                if capture:
                    if not line.strip():
                        capture = False
                        continue
                    if "---" in line: continue

                    line = line.replace("|||", "|")
                    line = line.strip().strip('|')
                    table_lines.append(line)

            if table_lines:
                reader = csv.DictReader(table_lines, delimiter='|')
                reader.fieldnames = [f.strip() for f in reader.fieldnames] if reader.fieldnames else []

                col_map = {
                    "Dataset Name": "Dataset Name", "Entity Type": "Type", "Type": "Type",
                    "Domain": "Domain", "Number of Variables": "Total Variables",
                    "Number of Time Points": "Number of Time Points",
                    "Time interval": "Time interval between points",
                    "Primary Source": "Primary Source Repository",
                    "Primary Home Page URL": "Primary URL",
                    "Link to Data (Actual Source)": "Link to Data (Actual Source)",
                    "Link to Data": "Link to Data (Actual Source)",
                    "Other URLs": "Other URL",
                    "Detailed Description": "Detailed Description"
                }

                valid_items = 0
                for row in reader:
                    row = {k: v.strip() for k, v in row.items() if v}
                    raw_name = self._clean_text(row.get("Dataset Name", ""))
                    if not raw_name or "---" in raw_name: continue

                    item = {"Dataset Name": {"value": raw_name, "confidence": 0.70, "telemetry_context": "Deep Research"}}

                    for dr_col, schema_col in col_map.items():
                        matched_key = next((k for k in row.keys() if dr_col in k), None)
                        if matched_key:
                            if "URL" in schema_col or "Link" in schema_col:
                                val = self._clean_url_field(str(row[matched_key]))
                            else:
                                val = self._clean_text(str(row[matched_key]))

                            if val and val.lower() != "nan":
                                if schema_col == "Type":
                                    if "collection" in val.lower() or "archive" in val.lower(): val = "Collection"
                                    elif "provider" in val.lower(): val = "Provider"
                                    else: val = "Dataset"
                                item[schema_col] = {"value": val, "confidence": 0.70, "telemetry_context": "Deep Research"}

                    item["Assignment Confidence"] = {"value": "0.80", "confidence": 1.0}
                    item["Assignment Rationale"] = {"value": "Identified by Deep Research Agent", "confidence": 1.0}
                    if "Type" not in item or not item["Type"].get("value"):
                        item["Type"] = {"value": "Dataset", "confidence": 0.8}

                    parsed_items.append(item)
                    valid_items += 1

                print(f"    🧠 Successfully parsed {valid_items} datasets from Deep Research.")

        except Exception as e:
            print(f"    ⚠️ Error parsing Deep Research table: {e}")

        return parsed_items

print("✅ deepcollector/tools/deep_research_runner.py written (Configurable Agent Model).")