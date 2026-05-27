# =============================================================================
# V183: RAG Engine (JSON Markdown Parse Fix Natively Baked In - Fully Uncompressed)
# =============================================================================
import json
import re
import asyncio
import time
import sys
import gc
import traceback
import pandas as pd
import math
import ast
from typing import List, Dict, Optional, Tuple, Any, TYPE_CHECKING
from datetime import datetime

try:
    from google.genai import types
except ImportError:
    types = None

try:
    from deepcollector.utils.profiler import profiler
except ImportError:
    class DummyProfiler:
        def track(self, c):
            return lambda f: f
        def update_stats(self, *a, **k):
            pass
    profiler = DummyProfiler()

try:
    from deepcollector.core.state import RAGResult, CatalogState, CellData
    try:
        from llama_index.core.schema import NodeWithScore
    except ImportError:
        from llama_index.core import NodeWithScore
except ImportError:
    CatalogState = object
    CellData = dict

if TYPE_CHECKING:
    from deepcollector.tools.research import ResearchTools
    from deepcollector.config.settings import AppConfig

# Safe markdown constants to prevent UI truncation issues
MD_JSON = chr(96) * 3 + "json\n"
MD_END = "\n" + chr(96) * 3


class RAGEngine:
    def __init__(self, config: 'AppConfig', tools: 'ResearchTools'):
        self.config = config
        self.tools = tools
        self.verbosity = getattr(config, 'VERBOSITY_LEVEL', 1)

        self.CATALOG_SCHEMA = getattr(config, 'CATALOG_SCHEMA', {})
        self.PLAUSIBILITY_THRESHOLDS = getattr(config, 'PLAUSIBILITY_THRESHOLDS', {})
        self.MISSING_DATA_PLACEHOLDERS = getattr(config, 'MISSING_DATA_PLACEHOLDERS', set())
        self.CELLULAR_RAG_BATCH_SIZE = getattr(config, 'CELLULAR_RAG_BATCH_SIZE', 50)
        self.CELLULAR_RAG_THROTTLE_DELAY = getattr(config, 'CELLULAR_RAG_THROTTLE_DELAY', 0.5)
        self.PARALLEL_CONCURRENCY_LIMIT = getattr(config, 'PARALLEL_CONCURRENCY_LIMIT', 4)

        # Configurable RAG Limits pulled dynamically from settings
        self.RAG_DISCOVERY_TOP_K = getattr(config, 'RAG_DISCOVERY_TOP_K', 15)
        self.RAG_DISCOVERY_MAX_CHARS = getattr(config, 'RAG_DISCOVERY_MAX_CHARS', 45000)
        self.RAG_CELLULAR_TOP_K = getattr(config, 'RAG_CELLULAR_TOP_K', 10)
        self.RAG_CELLULAR_MAX_CHARS = getattr(config, 'RAG_CELLULAR_MAX_CHARS', 35000)
        self.RAG_CELLULAR_FALLBACK_CHARS = getattr(config, 'RAG_CELLULAR_FALLBACK_CHARS', 15000)

    @profiler.track("RAGEngine: Discover Datasets")
    def discover_datasets_from_index(self, state: CatalogState) -> int:
        if not state.index:
            if self.verbosity >= 2:
                print("    ⚠️ [Discovery] No active vector index found. Skipping discovery.")
            return 0

        is_local = getattr(self.config, 'LLM_BACKEND', '') in ["LOCAL_PRO", "LOCAL_CLASSROOM"]
        retriever = state.get_retriever(similarity_top_k=self.RAG_DISCOVERY_TOP_K, mode="HYBRID")

        if not retriever:
            return 0

        query = f"List ALL datasets, archives, benchmarks, and repositories directly related to: '{state.context}'."

        try:
            nodes = retriever.retrieve(query)
            if not nodes:
                return 0

            context = self._format_context(nodes)[:self.RAG_DISCOVERY_MAX_CHARS]

            if getattr(self.config, 'ENABLE_VARIANT_MAPPING', False):
                variant_instruction = "2. **VARIANT MAPPING (Parent-Child):** If you find a dataset collection, list specific variants as SEPARATE dataset entries.\n"
            else:
                variant_instruction = "2. **EXTRACT LEAF DATASETS:** Your primary goal is to find specific individual datasets.\n"

            json_example = (
                MD_JSON +
                '{ "discovered_datasets": [{"dataset_name": "Name", "type": "Real-World Dataset", "confidence": 0.95, "rationale": "Is a leaf dataset in..."}] }' +
                MD_END
            )

            prompt = (
                f"Analyze the target project: '{state.context}'.\n\n"
                "--- Instructions ---\n"
                "1. **RUTHLESS SCOPE RULE:** ONLY extract 'Time-Series' datasets that are the PRIMARY FOCUS of the target project. Absolutely IGNORE datasets that are mentioned as 'related work', or static image datasets (e.g. Pascal VOC, ImageNet), NLP corpora, tabular ML sets (e.g. Titanic), or python scripts.\n"
                f"{variant_instruction}"
                "3. **UNPACK COLLECTIONS:** If a collection/archive is mentioned, list the specific datasets contained within it.\n"
                "4. **CLASSIFY TYPE / ENTITY TAXONOMY:** Explicitly classify each entry strictly as one of: [Real-World Dataset | Synthetic Dataset | Collection | Provider | Synthetic Generator | Augmentation Tool | Evaluation Script].\n"
                "5. **Format:** You MUST respond ONLY with a valid JSON block. Do NOT add extra conversational text.\n"
                f"{json_example}\n"
                "6. **Default Confidence:** 0.95.\n\n"
                f"--- Context ---\n{context}\n"
            )

            model = self.tools.models.MODEL_SYNTHESIZER
            max_new = 1536 if is_local else 1536
            response = self.tools.generate_content_synthesizer(model, prompt, max_new_tokens=max_new)

            if response.text == "[missing]" or response.text == "{}" or response.text == "[]":
                return 0

            return self._parse_discovery_response(state, response.text)

        except Exception as e:
            if getattr(self.config, '_CUDA_OOM_ABORT', False):
                raise e
            if self.verbosity >= 1:
                print(f"    ❌ [Discovery Error] Failed to extract from index: {str(e)[:150]}")
            return 0

    def _parse_discovery_response(self, state: CatalogState, text: str) -> int:
        added = 0
        try:
            data = self.tools._extract_json_robustly(text)

            if isinstance(data, dict):
                datasets = data.get("discovered_datasets", [])
            elif isinstance(data, list):
                datasets = data
            else:
                datasets = []

            for d in datasets:
                if not isinstance(d, dict):
                    continue

                name = d.get("dataset_name", "").strip()
                if not name or len(name) < 2 or name.lower() in ["varies", "n/a", "unknown", "dataset", "time series"] or name.endswith(('.py', '.xlsx')):
                    continue

                raw_conf = float(d.get("confidence", 0.5))
                conf = max(raw_conf, 0.95)
                raw_type = str(d.get("type", "Real-World Dataset")).strip().title()

                if getattr(self.config, 'ENABLE_ENTITY_TAXONOMY', False):
                    invalid_types = ["Synthetic Generator", "Augmentation Tool", "Evaluation Script"]
                    if any(inv.lower() in raw_type.lower() for inv in invalid_types):
                        continue

                    if "Real" in raw_type or "Synthetic" in raw_type or "Dataset" in raw_type:
                        raw_type = "Dataset"
                    elif "Collection" in raw_type:
                        raw_type = "Collection"
                    elif "Provider" in raw_type:
                        raw_type = "Provider"
                    else:
                        raw_type = "Dataset"
                else:
                    if raw_type not in ["Dataset", "Collection", "Provider"]:
                        raw_type = "Dataset"

                existing = state.find_item_by_name(name)

                if not existing:
                    new_item = state._initialize_new_item(name)
                    new_item["Assignment Confidence"] = {
                        "value": str(conf),
                        "confidence": 1.0,
                        "telemetry_context": "Discovery RAG",
                        "anchor_ref_id": None
                    }
                    new_item["Assignment Rationale"] = {
                        "value": d.get("rationale", "Extracted from context"),
                        "confidence": 1.0,
                        "telemetry_context": "Discovery RAG",
                        "anchor_ref_id": None
                    }
                    new_item["Type"] = {
                        "value": raw_type,
                        "confidence": 0.95,
                        "telemetry_context": "Discovery RAG",
                        "anchor_ref_id": None
                    }
                    state.catalog.append(new_item)
                    added += 1
                else:
                    try:
                        curr_val = float(state.get_cell_data(name, "Assignment Confidence")['value'])
                    except ValueError:
                        curr_val = 0.0

                    if conf > curr_val:
                        state.update_cell_data(
                            name,
                            "Assignment Confidence",
                            {"value": str(conf), "confidence": 1.0}
                        )

                    curr_type = state.get_cell_data(name, "Type").get("value", "[missing]")
                    if curr_type == "[missing]":
                        state.update_cell_data(
                            name,
                            "Type",
                            {"value": raw_type, "confidence": 0.95}
                        )

        except Exception as e:
            if getattr(self.config, '_CUDA_OOM_ABORT', False):
                raise e
            pass

        return added

    @profiler.track("RAGEngine: Plan Discovery")
    def plan_discovery_search(self, state: CatalogState) -> List[Dict[str, str]]:
        if not state.index:
            return []

        known_datasets = [i["Dataset Name"]["value"] for i in state.catalog]
        known_str = ", ".join(known_datasets[:20])

        json_example_queries = MD_JSON + '["query 1", "query 2"]' + MD_END

        prompt = (
            f"You are the strategic planner for a data cataloging agent. Target Project Context: {state.context}\n"
            f"Currently Discovered Datasets: {known_str}\n\nInstructions:\n"
            "1. Analyze the context and identified datasets.\n"
            "2. Determine what is missing. Are there foundational datasets likely used but not yet found?\n"
            "3. **RUTHLESS SCOPE RULE:** Do NOT generate searches for 'related' datasets, later versions, or competitors.\n"
            "4. Generate 3 targeted Google Search queries to find these missing data sources.\n"
            "5. Format STRICTLY as a JSON list of strings ONLY inside a markdown block:\n"
            f"{json_example_queries}\n"
        )

        try:
            model = self.tools.models.MODEL_PLANNER
            is_local = getattr(self.config, 'LLM_BACKEND', '') in ["LOCAL_PRO", "LOCAL_CLASSROOM"]
            response = self.tools.generate_content_planner(model, prompt, max_new_tokens=256 if is_local else 512)

            if response.text in ["[]", "[missing]"]:
                return []

            data = self.tools._extract_json_robustly(response.text)

            if isinstance(data, list):
                return [{"type": "search", "query": q} for q in data if isinstance(q, str)]
            else:
                return []

        except Exception as e:
            if getattr(self.config, '_CUDA_OOM_ABORT', False):
                raise e
            pass
            return []

    @profiler.track("RAGEngine: Execute Cellular RAG")
    def execute_cellular_rag(self, state: CatalogState, target_fields: List[str], retrieval_mode="HYBRID") -> Tuple[int, int, int]:
        is_local = getattr(self.config, 'LLM_BACKEND', '') in ["LOCAL_PRO", "LOCAL_CLASSROOM"]

        if not self.tools or (not getattr(self.tools.models, 'CLIENT', None) and not is_local):
            return 0, 0, 0

        candidate_cells = self._identify_candidate_cells(state, target_fields)
        if not candidate_cells:
            return 0, 0, 0

        retriever = state.get_retriever(similarity_top_k=self.RAG_CELLULAR_TOP_K, mode=retrieval_mode)
        if not retriever:
            return 0, 0, 0

        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(self._run_rag_batches(state, candidate_cells, retriever))
        fills, refinements, confirmed = self._process_rag_results(state, results)

        if self.verbosity >= 1:
            print(f"    📊 [Cellular RAG] Execution complete. Updates applied: {fills + refinements}")

        return fills, refinements, confirmed

    async def _run_rag_batches(self, state, candidate_cells, retriever):
        results = []
        total_cells = len(candidate_cells)
        is_local = getattr(self.config, 'LLM_BACKEND', '') in ["LOCAL_PRO", "LOCAL_CLASSROOM"]
        batch_size = 1 if is_local else self.CELLULAR_RAG_BATCH_SIZE

        for i in range(0, total_cells, batch_size):
            batch = candidate_cells[i:i+batch_size]
            start_time = time.time()

            if self.verbosity >= 1:
                print(f"    🔄 [Cellular RAG Batch] Processing batch {i//batch_size + 1} of {math.ceil(total_cells/batch_size)} ({len(batch)} cells)...")

            tasks = []
            valid_batch = []

            for cell_info in batch:
                dataset_name = cell_info["dataset_name"]
                field_name = cell_info["field_name"]

                query_template = self.CATALOG_SCHEMA.get(field_name, {}).get("query")
                if not query_template:
                    continue

                verified_url = self._get_cell_value(cell_info['item'], "Link to Data (Actual Source)")
                if verified_url == "[missing]":
                    verified_url = None

                task = self._extract_cell_data_rag(
                    dataset_name,
                    effective_name=state.get_effective_name(cell_info['item']),
                    field_name=field_name,
                    query_template=query_template,
                    verified_url=verified_url,
                    retriever=retriever
                )
                tasks.append(task)
                valid_batch.append(cell_info)

            batch_results = []
            if is_local:
                import torch
                for t in tasks:
                    try:
                        res = await t
                        batch_results.append(res)
                    except Exception as e:
                        batch_results.append(RuntimeError(str(e)))
                    gc.collect()
                    if torch is not None and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
            else:
                batch_results = await self._run_async_tasks(tasks)

            for cell_info, rag_result in zip(valid_batch, batch_results):
                if getattr(self.config, '_CUDA_OOM_ABORT', False):
                    raise RuntimeError("CUDA OOM Abort")

                if isinstance(rag_result, Exception):
                    continue

                if rag_result:
                    results.append({
                        "dataset_name": cell_info["dataset_name"],
                        "field_name": cell_info["field_name"],
                        "rag_result": rag_result
                    })

            duration = time.time() - start_time
            profiler.update_stats("LLM: Cellular RAG", duration, count=len(batch))

            if i + batch_size < total_cells:
                await asyncio.sleep(self.CELLULAR_RAG_THROTTLE_DELAY)

        return results

    async def _run_async_tasks(self, tasks):
        semaphore = asyncio.Semaphore(self.PARALLEL_CONCURRENCY_LIMIT)

        async def semaphore_wrapper(task):
            async with semaphore:
                try:
                    return await task
                except Exception as e:
                    return RuntimeError(str(e))

        wrapped_tasks = [semaphore_wrapper(task) for task in tasks]
        return await asyncio.gather(*wrapped_tasks, return_exceptions=True)

    async def _extract_cell_data_rag(self, dataset_name, effective_name, field_name, query_template, verified_url, retriever) -> Optional[RAGResult]:
        is_local = getattr(self.config, 'LLM_BACKEND', '') in ["LOCAL_PRO", "LOCAL_CLASSROOM"]
        base_query = query_template.format(name=effective_name)
        queries = [base_query]

        numeric_instruction = ""
        f_lower = field_name.lower()

        if "time points" in f_lower or "variables" in f_lower or "locations" in f_lower:
            numeric_instruction = " WARNING: The value MUST be ONLY the raw integer number."
        elif "frequency" in f_lower or "interval" in f_lower:
            numeric_instruction = " WARNING: The value MUST be a concise phrase."
        elif "domain" in f_lower:
            numeric_instruction = " WARNING: The value MUST be a concise category."

        if getattr(self.config, 'ENABLE_MULTI_QUERY_RAG', False):
            if "Variables" in field_name:
                queries.append(f"How many features, dimensions, or variables does the {effective_name} dataset have?")
            elif "Time Points" in field_name:
                queries.append(f"What is the total length, number of rows, or number of time points for the {effective_name} dataset?")
            elif "Frequency" in field_name or "interval" in field_name.lower():
                queries.append(f"What is the sampling rate, frequency, or time interval for the {effective_name} dataset?")
            elif "url" in field_name.lower() or "link" in field_name.lower():
                queries.append(f"What is the official website or download link for the {effective_name} dataset?")

        nodes = []
        seen_node_ids = set()

        try:
            for q in queries:
                if hasattr(retriever, 'aretrieve'):
                    q_nodes = await retriever.aretrieve(q)
                else:
                    q_nodes = retriever.retrieve(q)

                for n in q_nodes:
                    nid = getattr(n.node, 'node_id', getattr(n, 'node_id', None))
                    if nid not in seen_node_ids:
                        seen_node_ids.add(nid)
                        nodes.append(n)
        except Exception as e:
            if getattr(self.config, '_CUDA_OOM_ABORT', False):
                raise e
            del e

        if nodes and hasattr(nodes, 'score'):
            nodes.sort(key=lambda x: getattr(x, 'score', 0.0), reverse=True)

        nodes = nodes[:self.RAG_CELLULAR_TOP_K]

        context_snippets = []
        for node in nodes:
            meta = node.node.metadata if hasattr(node, 'node') else getattr(node, 'metadata', {})
            url = meta.get("url", "N/A")

            if verified_url and url == verified_url:
                prefix = "Verified Source"
            else:
                prefix = f"Source ({meta.get('type', 'Unknown')})"

            if hasattr(node, 'get_content'):
                content = node.get_content()
            else:
                content = str(node)

            context_snippets.append(f"--- [{prefix}: {meta.get('title', 'N/A')}] ---\n{content}")

        context = self._sanitize_context("\n\n".join(context_snippets))

        arbitration_instruction = ""
        if getattr(self.config, 'ENABLE_ARBITRATION_PROMPT', False):
            arbitration_instruction = "**DISCREPANCY ARBITRATION:** Review the context chunks carefully. PRIORITIZE repository file structures and technical appendices over high-level abstract mentions."

        kwargs = {}
        if types and not is_local:
            try:
                kwargs["config"] = types.GenerateContentConfig(response_mime_type="application/json")
            except Exception:
                pass

        for attempt in range(2):
            try:
                if attempt == 0:
                    max_len = self.RAG_CELLULAR_MAX_CHARS
                else:
                    max_len = self.RAG_CELLULAR_FALLBACK_CHARS

                curr_context = context[:max_len]
                if len(context) > max_len:
                    curr_context += "... [TRUNCATED]"

                citation_instruction = ""
                if "citation" in field_name.lower():
                    citation_instruction = "IMPORTANT: Return the FULL academic citation if available."

                json_example = '{"value": "Extracted Text", "confidence": 0.95, "rationale": "Found in context"}'
                json_instructions = f"\nFormat EXACTLY as raw JSON. Do NOT wrap in markdown blocks. Example:\n{json_example}\n"

                prompt = (
                    f"Query: {base_query}\nTarget Field: {field_name}\nContext:\n{curr_context}\n\n"
                    f"Instructions: Answer strictly based on context. {numeric_instruction} {citation_instruction} {arbitration_instruction} "
                    f"If not found, set value='[missing]'. {json_instructions}"
                )

                resp = await self.tools.generate_content_rag_async(prompt, max_new_tokens=256, **kwargs)

                if not resp or not resp.text:
                    raise ValueError("Empty response")

                if resp.text in ["[missing]", "{}", "[]"]:
                    if attempt == 1:
                        return None
                    continue

                val, conf, rat = self._parse_response(resp.text, True, field_name)

                is_plausible, reason = self._validate_plausibility(field_name, val)

                if not is_plausible:
                    val = "[implausible]"
                    conf = 0.2
                    rat = rat + f" [VETO: {reason}]"

                return Auditor(self.config).format_result(val, conf, rat, nodes)

            except Exception as e:
                if getattr(self.config, '_CUDA_OOM_ABORT', False):
                    raise e
                if attempt == 1:
                    return None

        return None


    def _process_rag_results(self, state, results) -> Tuple[int, int, int]:
        fills = 0
        refinements = 0
        confirmed = 0

        for res in results:
            d_name = res["dataset_name"]
            f_name = res["field_name"]
            new_data = res["rag_result"]["cell_data"]

            old_data = state.get_cell_data(d_name, f_name)
            old_val = old_data.get("value", "[missing]")
            old_conf = old_data.get("confidence", 0.0)

            new_val = new_data.get("value", "[missing]")
            new_conf = new_data.get("confidence", 0.0)

            is_fill = False
            if old_val in self.MISSING_DATA_PLACEHOLDERS and new_val not in self.MISSING_DATA_PLACEHOLDERS:
                is_fill = True

            is_same = False
            if old_val.lower().strip() == str(new_val).lower().strip():
                is_same = True

            if is_fill:
                if state.update_cell_data(d_name, f_name, new_data):
                    fills += 1
            elif is_same:
                 if new_conf > old_conf:
                     state.update_cell_data(d_name, f_name, new_data)
                 confirmed += 1
            elif not is_same and new_conf >= old_conf:
                 if state.update_cell_data(d_name, f_name, new_data):
                     refinements += 1

        return fills, refinements, confirmed


    def _parse_response(self, text, is_json, field_name=""):
        text = self._clean_json_text(text)
        val = "[missing]"
        conf = 0.0
        rat = "JSON Parse Error"

        if is_json:
            data = self.tools._extract_json_robustly(text)
            if isinstance(data, dict) and "value" in data:
                val = data.get('value', '[missing]')

                if isinstance(val, list):
                    val = ", ".join([str(v).strip() for v in val if v])
                elif isinstance(val, dict):
                    val = str(val)

                val = re.sub(r'[\r\n]+', ' ', str(val).strip())

                conf = float(data.get('confidence', 0.0))
                if conf > 10.0:
                    conf = conf / 100.0
                elif conf > 1.0:
                    conf = conf / 10.0

                conf = min(conf, 1.0)
                rat = data.get('rationale', '')
            else:
                val_match = re.search(r'"value"\s*:\s*"([^"]+)"', text, re.I)
                c_match = re.search(r'"confidence"\s*:\s*([\d.]+)', text, re.IGNORECASE)

                if val_match:
                    val = val_match.group(1).strip()
                if c_match:
                    try:
                        conf = min(float(c_match.group(1)), 1.0)
                    except:
                        pass

        val = str(val).strip()
        if val.startswith("['") and val.endswith("']"):
            val = val[2:-2]
        if val.startswith('["') and val.endswith('"]'):
            val = val[2:-2]

        f_lower = field_name.lower()
        if "url" in f_lower or "link" in f_lower:
            m = re.search(r'(https?://[^\s\'"\]\)\>\,]+)', val)
            if m:
                val = m.group(1)

        return val, conf, rat


    def _clean_json_text(self, text):
        tb = chr(96) * 3
        if text.strip().startswith(tb):
            try:
                text = text.strip()
                if text.startswith(tb + "json"):
                    text = text[7:]
                elif text.startswith(tb):
                    text = text[3:]

                if text.endswith(tb):
                    text = text[:-3]
                return text.strip()
            except Exception:
                return text
        return text


    def _sanitize_context(self, text):
        text = re.sub(r'[ \t]+', ' ', text)
        return re.sub(r'\n{3,}', '\n\n', text)


    def _get_cell_value(self, item, field):
        return item.get(field, {}).get("value", "[missing]")


    def _validate_plausibility(self, field, val) -> Tuple[bool, str]:
        if val == "[missing]":
            return True, ""

        thresholds = self.PLAUSIBILITY_THRESHOLDS.get(field)
        if not thresholds:
            return True, ""

        v_lower = str(val).lower()
        multiplier = 1

        if re.search(r'(?:\b|\d)(billion|b)(?:\b|$)', v_lower):
            multiplier = 1_000_000_000
        elif re.search(r'(?:\b|\d)(million|m)(?:\b|$)', v_lower):
            multiplier = 1_000_000
        elif re.search(r'(?:\b|\d)(thousand|k)(?:\b|$)', v_lower):
            multiplier = 1_000

        nums = re.findall(r'(\d+\.?\d*)', str(val).replace(',', ''))
        if not nums:
            return True, ""

        max_val = max([float(n) for n in nums]) * multiplier

        if "min" in thresholds and max_val < thresholds["min"]:
            return False, "Below Min"
        if "max" in thresholds and max_val > thresholds["max"]:
            return False, "Above Max"

        return True, ""


    def _identify_candidate_cells(self, state, fields):
        cands = []
        for item in state.catalog:
            name = item["Dataset Name"]["value"]
            for f in fields:
                if state.get_cell_data(name, f)["confidence"] < 0.95:
                    cands.append({"dataset_name": name, "field_name": f, "item": item})
        return cands


    def _format_context(self, nodes):
        formatted_nodes = []
        for n in nodes:
            url = getattr(n, 'metadata', {}).get('url','')
            if hasattr(n, 'get_content'):
                content = n.get_content()[:2000]
            else:
                content = str(n)[:2000]
            formatted_nodes.append(f"--- Source: {url} ---\n{content}")
        return "\n\n".join(formatted_nodes)


class Auditor:
    def __init__(self, config):
        self.config = config
        self.MISSING = getattr(config, 'MISSING_DATA_PLACEHOLDERS', set())

    def format_result(self, val, conf, rat, nodes) -> RAGResult:
        if val.lower() in self.MISSING:
            val = "[missing]"
            conf = 0.0

        srcs = [getattr(n, 'metadata', {}).get('url') for n in nodes if getattr(n, 'metadata', {}).get('url')]

        if nodes and isinstance(nodes, list) and hasattr(nodes[0], 'score'):
            score = nodes[0].score
        else:
            score = 0.0

        if nodes and isinstance(nodes, list):
            anchor_id = getattr(nodes[0], 'node_id', None)
        else:
            anchor_id = None

        data = {
            "value": val,
            "confidence": conf,
            "telemetry_context": f"Rationale: {rat}\nTop Score: {score:.2f}\nSources: {srcs[:3]}",
            "anchor_ref_id": anchor_id
        }

        if CellData != dict:
            data = CellData(**data)

        return RAGResult(cell_data=data, potential_sources=srcs)

print("✅ deepcollector/core/rag_engine.py written (100% Fully Expanded + Native JSON Prompt Fix).")