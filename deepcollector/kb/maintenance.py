# =============================================================================
# V166: Dataset Doctor (SyntaxWarning Fix & Aggressive Artifact Cleanser)
# =============================================================================
import pandas as pd
import time
import re
import json
from typing import List, Dict, Any

class DatasetDoctor:
    def __init__(self, kb_manager, research_tools, verbosity=1):
        self.kb_manager = kb_manager
        self.tools = research_tools
        self.verbosity = verbosity

        self.INVALID_NAMES = {
            "", "[missing]", "missing", "unknown", "n/a", "nan", "none",
            "dataset", "time series", "time series dataset", "various", "varies", "null"
        }

    def _get_effective_name(self, item):
        def clean(s): return str(s).strip()
        def is_valid(s): return s.lower() not in self.INVALID_NAMES and not s.startswith("DS_")

        variant = clean(item.get("Dataset Name", {}).get("value", ""))
        if is_valid(variant): return variant

        variant_alt = clean(item.get("Variant Name", {}).get("value", ""))
        if is_valid(variant_alt): return variant_alt

        aliases = clean(item.get("Aliases", {}).get("value", ""))
        if aliases:
            for a in aliases.split(','):
                if is_valid(a.strip()): return a.strip()

        canon = clean(item.get("Canonical Name", {}).get("value", ""))
        if is_valid(canon): return canon

        return variant

    def execute_maintenance(self, catalog: List[Dict]) -> List[Dict]:
        print("\n=== PHASE 3.5: MAINTENANCE & REPAIR (V166) ===")
        if not catalog: return []

        catalog = self._clean_names_regex(catalog)
        catalog = self._clean_urls(catalog)
        catalog = self._exact_deduplicate(catalog)
        catalog = self._repair_zombies(catalog)
        catalog = self._sanitize_uci_links(catalog)

        return catalog

    def _clean_names_regex(self, catalog: List[Dict]) -> List[Dict]:
        print("    🧹 [Doctor] Standardizing names...")

        patterns = [
            r'^thuml/', r'^google/', r'^microsoft/', r'^facebook/', r'^ucr/',
            r'\s*\((?:.*\bUTSD\b.*)\)', r'\s*\((?:.*\bTimer\b.*)\)',
            r'\s*\(Component\)', r'\s*\(Subset\)', r'\s*\(Collection\)',
            r'[-_ ](small|medium|large|tiny|full|subset|complete|12g|1g|4g)(?=[-_ ]|$)'
        ]

        count = 0
        for item in catalog:
            name_obj = item.get("Dataset Name", {})
            old_name = name_obj.get("value", "")
            new_name = old_name

            for p in patterns:
                new_name = re.sub(p, '', new_name, flags=re.IGNORECASE).strip()

            new_name = new_name.strip("-_ ")

            if new_name != old_name and len(new_name) > 2:
                item["Dataset Name"]["value"] = new_name
                item["Variant Name"] = {"value": old_name, "confidence": 1.0}
                count += 1

        if count > 0 and self.verbosity >= 1:
            print(f"    ✨ Standardized {count} names.")

        return catalog

    def _clean_urls(self, catalog: List[Dict]) -> List[Dict]:
        print("    🧹 [Doctor] Cleaning URL formatting and purging banned domains...")
        url_cols = ["Link to Data (Actual Source)", "Primary URL", "Other URL", "URL"]
        banned_domains = ['vertexaisearch.cloud.google.com', 'grounding-api-redirect', 'scholar.google']
        cleaned_count = 0
        purged_count = 0

        for item in catalog:
            for col in url_cols:
                if col in item:
                    val = str(item[col].get("value", "")).strip()
                    if not val or val in self.INVALID_NAMES: continue

                    if any(banned in val.lower() for banned in banned_domains):
                        item[col] = {"value": "[missing]", "confidence": 0.0}
                        purged_count += 1
                        continue

                    old_val = val

                    # 1. Strip exact outer match brackets and quotes
                    val = re.sub(r"^\[\]\'\"]+|\[\]\'\"]+$", "", val)

                    # 2. Re-format if it gave a comma-separated list of links inside a string
                    val = val.replace("', '", ", ").replace('", "', ', ')

                    # 3. Pull strictly the URL out if it got wrapped in markdown after all
                    match = re.search(r'\[.*?\]\((https?://.*?)\)', val)
                    if match:
                        val = match.group(1)
                    elif val.startswith("http") and " " in val:
                        match = re.search(r'(https?://[^\s]+)', val)
                        if match:
                            val = match.group(1)

                    # 4. Final safety strip of trailing punctuation
                    # CRITICAL FIX: Removed the backslash completely. rstrip takes a set of characters, so "]", ">" etc are perfectly valid
                    val = val.rstrip(".,;:'\"]>)")

                    if val != old_val:
                        item[col]["value"] = val
                        cleaned_count += 1

        if cleaned_count > 0 or purged_count > 0:
            if self.verbosity >= 1:
                print(f"    ✨ Cleaned {cleaned_count} URLs, Purged {purged_count} banned links.")

        return catalog

    def _exact_deduplicate(self, catalog: List[Dict]) -> List[Dict]:
        print(f"    🧠 [Doctor] Running STRICT Exact Deduplication on {len(catalog)} items...")
        seen = {}
        cleaned = []
        for item in catalog:
            name = self._get_effective_name(item).lower().strip()
            if not name or name in self.INVALID_NAMES: continue

            if name in seen: self._merge_data(seen[name], item)
            else:
                seen[name] = item
                cleaned.append(item)

        dropped = len(catalog) - len(cleaned)
        if dropped > 0: print(f"    📉 Deduplication finished. Safely merged {dropped} exact duplicates.")
        else: print("    ✅ No exact duplicates found.")
        return cleaned

    def _merge_data(self, target: Dict, source: Dict):
        for field, source_cell in source.items():
            if field.startswith("_"): continue
            target_cell = target.get(field)
            source_val = source_cell.get("value", "")
            target_val = target_cell.get("value", "") if target_cell else "[missing]"

            if target_val in ["[missing]", "", None] and source_val not in ["[missing]", "", None]:
                target[field] = source_cell.copy()
            if isinstance(target_cell, dict) and isinstance(source_cell, dict):
                if source_cell.get("confidence", 0) > target_cell.get("confidence", 0):
                    target[field] = source_cell.copy()

    def _sanitize_uci_links(self, catalog):
        print("    🧹 [Sanitizer] Checking for broken UCI zip links...")
        fixed_count = 0
        for item in catalog:
            url_obj = item.get("Link to Data (Actual Source)", {})
            url = url_obj.get("value", "")
            if "archive.ics.uci.edu" in url and url.endswith(".zip"):
                name = self._get_effective_name(item)
                try:
                    query = f'site:archive.ics.uci.edu/dataset/ "{name}"'
                    results = self.tools.tool_search_and_fetch(query, num_results=1)
                    if results and "/dataset/" in results.get('url', ''):
                        new_url = results['url']
                        item["Link to Data (Actual Source)"] = {"value": new_url, "confidence": 0.98}
                        fixed_count += 1
                        if self.verbosity >= 1: print(f"      ✨ Fixed Link: {new_url}")
                except Exception: pass

        print(f"    🧹 Sanitization complete. Fixed {fixed_count} bad links.")
        return catalog

    def _repair_zombies(self, catalog):
        print(f"    🩺 [Doctor] Scanning {len(catalog)} items for missing or redirect URLs...")
        fixed_count = 0
        url_cols = ["Link to Data (Actual Source)", "Primary URL", "URL"]
        invalid_data_domains = [
            'arxiv.org', 'doi.org', 'researchgate.net', 'ieeexplore.ieee.org',
            'sciencedirect.com', 'springer.com', 'nature.com', 'acm.org', 'semanticscholar.org',
            'vertexaisearch.cloud.google.com', 'grounding-api-redirect'
        ]

        for item in catalog:
            name = self._get_effective_name(item)
            if not name or name.lower().strip() in self.INVALID_NAMES: continue
            if item.get("Type", {}).get("value") == "Provider": continue

            has_valid_url = False
            for col in url_cols:
                val = item.get(col, {}).get("value", "")
                if val and val not in ["[missing]", "n/a", ""] and "http" in val:
                    if not any(bad_dom in val.lower() for bad_dom in invalid_data_domains):
                        has_valid_url = True
                        break

            if not has_valid_url:
                if self.verbosity >= 1: print(f"    🧟 Zombie / Paper URL Found (Needs real URL): {name}")
                found_url = None
                try:
                    query = f"official download url repository github dataset '{name}'"
                    res = self.tools.tool_search_and_fetch(query, num_results=2)
                    if res:
                        for r in res:
                            candidate_url = r.get('url', '')
                            if candidate_url and not any(bad_dom in candidate_url.lower() for bad_dom in invalid_data_domains):
                                found_url = candidate_url
                                break
                except Exception: pass

                if found_url:
                    item["Link to Data (Actual Source)"] = {"value": found_url, "confidence": 0.95}
                    print(f"      ✨ Healed with Real URL: {found_url}")
                    fixed_count += 1
                else:
                    if self.verbosity >= 1: print("      ⚠️ Could not recover true data URL. Retaining original fallback link.")

        print(f"    🩺 Repair complete. Fixed {fixed_count} items.")
        return catalog

print("✅ deepcollector/kb/maintenance.py written (SyntaxWarning Fix).")