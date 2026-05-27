# =============================================================================
# V33: CatalogState (LlamaIndex Metadata Crash Fix + Search Memory Loop Killer)
# =============================================================================
import pandas as pd
import re
import copy
import math
import time
import asyncio
import os
from typing import Dict, List, TypedDict, Optional, Any, Union

try:
    from llama_index.core import VectorStoreIndex, Document, Settings
    from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever
    from llama_index.core.schema import NodeWithScore
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.retrievers.bm25 import BM25Retriever
except ImportError:
    VectorStoreIndex = None; Document = None; VectorIndexRetriever = None; BaseRetriever = object
    NodeWithScore = object; BM25Retriever = None; Settings = None; SentenceSplitter = None
    print("⚠️ [State] LlamaIndex components missing.")

try:
    from deepcollector.utils.profiler import profiler
except ImportError:
    class DummyProfiler:
        def track(self, c):
            return lambda f: f
        def update_stats(self, *a, **k):
            pass
    profiler = DummyProfiler()

class CellData(TypedDict):
    value: str
    confidence: float
    telemetry_context: Optional[str]
    anchor_ref_id: Optional[str]

CatalogItem = Dict[str, CellData]

class RAGResult(TypedDict):
    cell_data: CellData
    potential_sources: List[str]

if VectorIndexRetriever and BM25Retriever:
    class HybridRetriever(BaseRetriever):
        def __init__(self, vr, br):
            self.vr = vr
            self.br = br
            super().__init__()

        def _retrieve(self, q, **kwargs):
            bn = self.br.retrieve(q, **kwargs)
            vn = self.vr.retrieve(q, **kwargs)
            seen = set()
            final = []
            for n in vn + bn:
                if n.node.node_id not in seen:
                    final.append(n)
                    seen.add(n.node.node_id)
            return final[:self.vr.similarity_top_k]

class CatalogState:
    def __init__(self, config: Any, context: str, project_id: str):
        self.config = config
        self.context = context
        self.project_id = project_id
        self.verbosity = getattr(config, 'VERBOSITY_LEVEL', 1)
        self.catalog: List[CatalogItem] = []
        self.history: List[str] = []
        self.current_phase: str = "INITIALIZATION"
        self.iteration: int = 0
        self.confidence_history: List[Dict[str, Any]] = []

        self.discovered_citations: List[Dict[str, Any]] = []
        self.discovered_weblinks: List[Dict[str, Any]] = []
        self.discovered_composites: List[Dict[str, Any]] = []

        self.index: Optional[VectorStoreIndex] = None
        self.documents: List[Document] = []
        self.all_nodes: List[Any] = []
        self.indexed_urls: set = set()

        # 🧠 FIX: Search Memory prevents infinite RAG extraction loops
        self.past_searches: set = set()

        self.bm25_retriever: Optional[Any] = None
        self.BM25RetrieverClass = getattr(config, 'BM25Retriever', None)

        if VectorStoreIndex and getattr(config, 'LLAMA_INDEX_AVAILABLE', False):
            try:
                self.index = VectorStoreIndex.from_documents([])
            except Exception as e:
                if self.verbosity >= 2: print(f"⚠️ [State] LlamaIndex Init Failed: {e}")
                self.index = None

        self.CATALOG_SCHEMA = getattr(config, 'CATALOG_SCHEMA', {})
        self.MISSING_DATA_PLACEHOLDERS = getattr(config, 'MISSING_DATA_PLACEHOLDERS', set(["[missing]"]))
        self.CONFIDENCE_LOCK_THRESHOLD = getattr(config, 'CONFIDENCE_LOCK_THRESHOLD', 0.95)
        self.INDEXING_BATCH_SIZE = getattr(config, 'INDEXING_BATCH_SIZE', 10)
        self.EXTRACTED_FIELDS = getattr(config, 'EXTRACTED_FIELDS', [])
        self.GROUNDING_FIELDS = getattr(config, 'GROUNDING_FIELDS', [])
        self.MIN_ASSIGNMENT_CONFIDENCE_GATE = getattr(config, 'MIN_ASSIGNMENT_CONFIDENCE_GATE', 0.70)

        self.INVALID_NAMES = {
            "", "[missing]", "unknown", "n/a", "nan", "none",
            "dataset", "time series", "time series dataset", "various", "varies", "null"
        }

    def _is_valid_dataset_scope(self, name: str) -> bool:
        if not name: return False
        n = name.lower()
        if n.endswith(('.py', '.xlsx', '.csv', '.md', '.txt')): return False
        invalid_static = ['pascal voc', 'titanic', 'shopee', 'imagenet', 'cifar', 'mnist']
        if any(inv in n for inv in invalid_static): return False
        return True

    def add_history(self, message: str):
        self.history.append(f"[{self.current_phase} Iter {self.iteration}] {message}")

    @profiler.track("State: Add Data and Index")
    def add_data_and_index(self, data: List[Dict[str, str]]):
        if not self.index: return

        new_documents = []
        for item in data:
            url = str(item.get("url", "N/A"))[:150]
            title = str(item.get("title", "N/A"))[:150]
            m_type = str(item.get("type", "Web"))[:50]
            content = item.get("content", "")

            ref_id = item.get("ref_id", url)
            if not content or ref_id in self.indexed_urls: continue

            text = f"Title: {title}\nURL: {url}\nContext: {self.context}\nContent:\n{content}"
            if Document:
                doc = Document(
                    text=text,
                    metadata={"url": url, "title": title, "type": m_type},
                    id_=ref_id
                )
                new_documents.append(doc)
                self.indexed_urls.add(ref_id)

        if not new_documents: return
        self.documents.extend(new_documents)

        try:
            node_parser = Settings.node_parser if Settings and hasattr(Settings, 'node_parser') else SentenceSplitter(chunk_size=1024)
            new_nodes = node_parser.get_nodes_from_documents(new_documents)
            self.all_nodes.extend(new_nodes)
            self.index.insert_nodes(new_nodes)
            if self.verbosity >= 1: print(f"    ✅ [Vector Index] Inserted {len(new_documents)} docs ({len(new_nodes)} nodes).")
        except Exception as e:
            if self.verbosity >= 2: print(f"    ❌ [Vector Index Error] {e}")

        if self.BM25RetrieverClass and self.all_nodes:
            try:
                self.bm25_retriever = self.BM25RetrieverClass.from_defaults(nodes=self.all_nodes, similarity_top_k=10)
                if self.verbosity >= 1: print(f"    ✅ [BM25 Index] Rebuilt with {len(self.all_nodes)} nodes.")
            except Exception as e:
                if self.verbosity >= 2: print(f"    ❌ [BM25 Index Error] {e}")

    def get_retriever(self, similarity_top_k=8, mode="HYBRID") -> Any:
        if not self.index or not VectorIndexRetriever: return None

        actual_k = min(similarity_top_k, len(self.all_nodes))
        if actual_k <= 0: actual_k = 1

        vr = VectorIndexRetriever(index=self.index, similarity_top_k=actual_k)
        if mode == "VECTOR": return vr

        if self.bm25_retriever and mode in ["BM25", "HYBRID"]:
            self.bm25_retriever.similarity_top_k = actual_k
            if mode == "BM25": return self.bm25_retriever
            if 'HybridRetriever' in globals():
                return HybridRetriever(vr, self.bm25_retriever)
        return vr

    def _initialize_new_item(self, name):
        item = {k: {"value": "[missing]", "confidence": 0.0, "telemetry_context": None, "anchor_ref_id": None} for k in self.CATALOG_SCHEMA}
        item["Dataset Name"] = {"value": name, "confidence": 0.5, "telemetry_context": "Init", "anchor_ref_id": None}
        return item

    def find_item_by_name(self, name):
        return next((i for i in self.catalog if i["Dataset Name"]["value"] == name), None)

    def get_cell_data(self, dataset_name, field, item_override=None):
        item = item_override or self.find_item_by_name(dataset_name)
        return item.get(field, {"value": "[missing]", "confidence": 0.0}) if item else {"value": "[missing]", "confidence": 0.0}

    def update_cell_data(self, dataset_name, field, new_data, allow_new_datasets=False, item_override=None):
        if not self._is_valid_dataset_scope(dataset_name):
            return False

        item = item_override or self.find_item_by_name(dataset_name)
        if not item:
            if not allow_new_datasets: return False
            item = self._initialize_new_item(dataset_name); self.catalog.append(item)

        if field not in item: return False
        curr = item[field]
        curr_val = curr.get("value", "[missing]")
        is_curr_missing = curr_val in self.MISSING_DATA_PLACEHOLDERS

        try: new_conf = float(new_data.get("confidence", 0.0))
        except: new_conf = 0.0

        if curr["confidence"] >= self.CONFIDENCE_LOCK_THRESHOLD and not is_curr_missing:
             return False

        new_val = new_data.get("value", "[missing]")
        is_new_missing = new_val in self.MISSING_DATA_PLACEHOLDERS

        should_update = False
        if is_curr_missing and not is_new_missing: should_update = True
        elif not is_curr_missing and new_conf > curr["confidence"]: should_update = True

        if should_update:
            curr.update({k: v for k, v in new_data.items() if k in curr})
            return True
        return False

    def update_catalog_batch(self, updates: List[CatalogItem], allow_new_datasets: bool = True):
        count = 0
        for up in updates:
            name = up.get("Dataset Name", {}).get("value")
            if not name or not self._is_valid_dataset_scope(name):
                continue
            for field, data in up.items():
                if self.update_cell_data(name, field, data, allow_new_datasets): count += 1
        if self.verbosity >= 1: print(f"    📊 [Batch Update] Updated {count} fields.")

    def capture_confidence_metrics(self, stage_name: str = "Unknown Stage") -> Dict[str, float]:
        if not self.catalog: return {"avg_conf": 0.0, "completeness": 0.0, "count": 0}
        total_conf = 0.0; total_cells = 0; filled_cells = 0
        for item in self.catalog:
            for field in self.EXTRACTED_FIELDS:
                data = self.get_cell_data(None, field, item)
                val = data.get("value", "[missing]")
                total_conf += data.get("confidence", 0.0)
                total_cells += 1
                if val not in self.MISSING_DATA_PLACEHOLDERS: filled_cells += 1
        avg_conf = (total_conf / total_cells) if total_cells > 0 else 0.0
        completeness = (filled_cells / total_cells) if total_cells > 0 else 0.0
        stats = {"Stage": stage_name, "Iteration": self.iteration, "Avg Confidence": avg_conf, "Completeness": completeness, "Catalog Size": len(self.catalog)}
        self.confidence_history.append(stats)
        if self.verbosity >= 1: print(f"    📊 [Metrics: {stage_name}] Catalog Size: {len(self.catalog)} | Avg Conf: {avg_conf:.2f} | Completeness: {completeness:.1%}")
        return {"avg_conf": avg_conf, "completeness": completeness, "count": len(self.catalog)}

    def get_average_confidence(self, fields=None):
        if not self.catalog: return 0.0
        fields = fields or self.CATALOG_SCHEMA.keys()
        total = 0.0; count = 0
        for item in self.catalog:
            for f in fields: total += self.get_cell_data(None, f, item)["confidence"]; count += 1
        return total / count if count else 0.0

    def inject_structured_knowledge(self, data): pass

    def update_project_resources(self, citations: List[Dict[str, Any]], weblinks: List[Dict[str, Any]]):
        if self.verbosity >= 1: print(f"    🔄 [State] Updating project resources (Citations: {len(citations)}, WebLinks: {len(weblinks)})...")
        self.discovered_citations.extend(citations)
        self.discovered_weblinks.extend(weblinks)

    def get_effective_name(self, item_or_name: Union[Dict, str]) -> str:
        item = item_or_name if isinstance(item_or_name, dict) else self.find_item_by_name(item_or_name)
        if not item: return str(item_or_name)

        def is_valid(val):
            s = str(val).strip().lower()
            if not s or s in self.MISSING_DATA_PLACEHOLDERS or s in self.INVALID_NAMES or s == "unknown": return False
            if s.startswith("ds_") and len(s) < 10: return False
            return True

        variant = item.get("Variant Name", {}).get("value", "")
        if is_valid(variant): return str(variant)

        ds_name = item.get("Dataset Name", {}).get("value", "")
        if is_valid(ds_name): return str(ds_name)

        aliases = item.get("Aliases", {}).get("value", "")
        if aliases and isinstance(aliases, str):
            for alias in aliases.split(','):
                clean_alias = alias.strip()
                if is_valid(clean_alias):
                    return clean_alias

        canon = item.get("Canonical Name", {}).get("value", "")
        if is_valid(canon): return str(canon)

        return "Unknown Dataset"

print("✅ deepcollector/core/state.py written (Added Search Memory Set).")