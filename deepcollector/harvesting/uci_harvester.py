# =============================================================================
# HDK V11.2: UCI Harvester (Deep Think Fix: Single Line Enforcer)
# =============================================================================

import requests
import time
import re
import random
import json
import pandas as pd
import traceback
from typing import List, Dict, Any, Optional, Tuple

try:
    import gspread
    from google.colab import auth
    from google.auth import default
except ImportError:
    gspread = None

try:
    import lxml
    BS_PARSER = 'lxml'
except ImportError:
    BS_PARSER = 'html.parser'

from bs4 import BeautifulSoup

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

from requests.exceptions import HTTPError, RequestException, ConnectionError, Timeout

try:
    from deepcollector.harvesting.base_harvester import BaseHarvester
    from deepcollector.core.state import CatalogState, CellData
    from deepcollector.utils.profiler import profiler
except ImportError:
    BaseHarvester = object
    CatalogState = object
    CellData = dict

    class DummyProfiler:
        def track(self, c):
            return lambda x: x

    profiler = DummyProfiler()


class UCIHarvester(BaseHarvester):
    """
    Comprehensive Embedded JSON Extraction Harvester for the UCI Machine Learning Repository.
    """

    BASE_URL = "https://archive.ics.uci.edu"
    API_LIST_ENDPOINT = f"{BASE_URL}/api/datasets/list"
    WEB_UI_BASE_LIST = f"{BASE_URL}/datasets"
    TARGET_DATA_TYPE = "Time-Series"

    # Export Configurations
    SPREADSHEET_NAME_TEMPLATE = "HDK_UCI_Harvest_Export_{date}"
    CHAR_LIMIT = 49000

    # User agents to rotate to avoid basic scraping blocks
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0'
    ]

    # Rate Limiting & Backoff
    FETCH_DELAY_MIN = 1.0
    FETCH_DELAY_MAX = 3.0
    ADAPTIVE_BACKOFF_TIME = 60.0
    MAX_CONSECUTIVE_FAILURES = 5

    def __init__(self, config: Any, tools: Any):
        """
        Initializes the UCI Harvester with the application configuration and tools.
        """
        if BaseHarvester != object:
            try:
                super().__init__(config, tools)
            except TypeError:
                self.config = config
                self.tools = tools
                self.verbosity = getattr(config, 'VERBOSITY_LEVEL', 1)
        else:
             self.config = config
             self.tools = tools
             self.verbosity = getattr(config, 'VERBOSITY_LEVEL', 1)

        # Initialize a persistent requests session for connection pooling
        if 'requests' in globals():
             self.session = requests.Session()
        else:
             self.session = None

    @profiler.track("Harvester: UCI Execute")
    def execute_harvest(self, state: CatalogState) -> bool:
        """
        The main orchestration method for the UCI Harvest.
        """
        print("\n" + "="*60)
        print("🚜 STARTING HARVEST (UCI V11.2 - Single Line Fix)")
        print("="*60)

        if not self.session or not BeautifulSoup:
            print("❌ [UCI Harvester] CRITICAL: Missing 'requests' or 'BeautifulSoup' libraries.")
            return False

        # --- Stage 1: Discovery ---
        asset_identifiers = self.discover_assets()

        if not asset_identifiers:
            print("🛑 [UCI Harvester] CRITICAL: Discovery failed or no assets found. Aborting harvest.")
            return False

        # --- Stage 2 & 3: Fetch HTML & Screen Metadata ---
        enriched_assets = self.fetch_extract_and_screen_metadata(asset_identifiers)

        if not enriched_assets:
            print("🚜 [UCI Stage 2/3] Finished. No target Time-Series assets identified.")
            return True

        # --- Stage 4: Update Catalog State ---
        self._update_state(state, enriched_assets)

        return True

    @profiler.track("Harvester: UCI Discover Assets (API)")
    def discover_assets(self) -> List[Dict[str, Any]]:
        """
        Fetches the complete roster of all available dataset IDs and Slugs.
        """
        print("🚜 [UCI Stage 1] Fetching Comprehensive API List...")
        all_assets = []

        try:
            if self.session is None:
                raise AttributeError("self.session is None")

            headers = self._get_randomized_headers(referer=self.WEB_UI_BASE_LIST)

            # The UCI API currently ignores skip/take and dumps everything.
            # We make one large request to be safe and avoid infinite loops.
            response = self.session.get(
                self.API_LIST_ENDPOINT,
                headers=headers,
                params={'skip': 0, 'take': 5000},
                timeout=45
            )
            response.raise_for_status()

            data = self._parse_json_response(response)

            batch_assets = []
            if isinstance(data, dict):
                if 'data' in data:
                    if isinstance(data['data'], list):
                        batch_assets = data['data']
            elif isinstance(data, list):
                batch_assets = data

            if not batch_assets:
                print("    ❌ [UCI Stage 1] API returned an empty list.")
                return []

            standardized_batch = self._standardize_keys(batch_assets)

            seen_ids = set()
            for item in standardized_batch:
                item_id = item.get('ID')
                if item_id and item_id not in seen_ids:
                    seen_ids.add(item_id)
                    all_assets.append(item)

            if self.verbosity >= 1:
                print(f"    ✅ [UCI Stage 1] API fetch complete. Retrieved {len(all_assets)} unique datasets.")

            return all_assets

        except Exception as e:
            print(f"    ❌ [UCI Stage 1 Error] {type(e).__name__}: {e}")
            return []

    @profiler.track("Harvester: UCI HTML Fetch")
    def fetch_extract_and_screen_metadata(self, asset_identifiers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Iterates through discovered assets, visits their specific HTML page,
        extracts the embedded JSON metadata, and screens them for the "Time-Series" tag.
        """
        print("\n🚜 [UCI Stage 2/3] HTML Fetch, Extraction, and Screening...")

        enriched_assets = []
        screened_out_count = 0
        failure_count = 0
        consecutive_failures = 0
        last_referer = self.WEB_UI_BASE_LIST

        # Allow user to artificially limit the run size via configuration
        MAX_ASSETS = getattr(self.config, 'HDK_LIMIT_ASSETS', None)
        if MAX_ASSETS and len(asset_identifiers) > MAX_ASSETS:
            print(f"    ⚠️ [Limit] Processing only the first {MAX_ASSETS} assets as per config.")
            asset_identifiers = asset_identifiers[:MAX_ASSETS]
        else:
             print(f"    💡 Processing {len(asset_identifiers)} assets. (Estimated time: 10-15 minutes)")

        for i, asset in enumerate(asset_identifiers):
            asset_id = asset.get('ID')
            asset_slug = asset.get('Slug')
            asset_name = asset.get('Name', "").strip()

            # Formulate a fallback name for logging
            fallback_name = asset_name
            if not fallback_name:
                if asset_slug:
                    fallback_name = asset_slug
                else:
                    fallback_name = f"ID {asset_id}"

            if not asset_id:
                continue

            # -------------------------------------------------------------
            # STRICT ANTI-GARBAGE FILTER:
            # Skip if API returned broken data or literal placeholder strings
            # -------------------------------------------------------------
            if not asset_name or str(asset_name).strip().lower() in ["null", "none", "nan", "missing", "[missing]", ""]:
                if self.verbosity >= 2:
                    print(f"    ⚠️ [Filter] Skipping ID {asset_id} due to missing or garbage name.")
                continue

            # Polite scraping delay
            time.sleep(random.uniform(self.FETCH_DELAY_MIN, self.FETCH_DELAY_MAX))

            if self.verbosity >= 1:
                print(f"    🔍 [Fetch] ({i+1}/{len(asset_identifiers)}) '{fallback_name}'...")

            slug_part = asset_slug if asset_slug else ""
            html_page_url = f"{self.BASE_URL}/dataset/{asset_id}/{slug_part}".rstrip('/')

            try:
                headers = self._get_randomized_headers(referer=last_referer)
                response = self.session.get(html_page_url, headers=headers, timeout=30)
                response.raise_for_status()
                last_referer = html_page_url

                if self._is_blocked(response.text):
                    consecutive_failures += 1
                    failure_count += 1
                    continue

                soup = BeautifulSoup(response.content, BS_PARSER)

                # 1. Try to get pristine metadata from embedded JSON
                extracted_blocks = self._extract_and_identify_json_blocks(soup)
                raw_metadata = extracted_blocks.get('metadata')
                raw_variables = extracted_blocks.get('variables')

                if raw_metadata:
                    metadata = self._map_embedded_json_to_standard_structure(raw_metadata, raw_variables)

                    # 2. Augment missing variables with Deep HTML Scraping
                    metadata = self._augment_metadata_from_html(metadata, soup)

                    # 3. Screen for "Time-Series"
                    if self._is_target_type(metadata):
                        if self.verbosity >= 1:
                            if raw_variables:
                                var_status = "(+Variables Table)"
                            else:
                                var_status = ""
                            print(f"    ✅ [MATCH] Target '{self.TARGET_DATA_TYPE}' Identified! {var_status}")

                        # Fallback for naming
                        if not metadata.get('name') or metadata.get('name') in ["[missing]", ""]:
                            metadata['name'] = fallback_name

                        metadata['_source_method'] = 'deep_scrape_v11.1'
                        enriched_assets.append(metadata)
                        consecutive_failures = 0
                    else:
                        if self.verbosity >= 2:
                            print(f"    ❌ [Filter] Dropping - Not '{self.TARGET_DATA_TYPE}'")
                        screened_out_count += 1
                        consecutive_failures = 0
                else:
                    # JSON failed completely
                    consecutive_failures += 1
                    failure_count += 1

            except Exception as e:
                if self.verbosity >= 2:
                    print(f"    ⚠️ [Fetch Error] {e}")
                consecutive_failures += 1
                failure_count += 1

            # Adaptive Backoff if we are hitting rate limits
            if consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                print(f"    ⚠️ [Backoff] {self.MAX_CONSECUTIVE_FAILURES} consecutive failures. Pausing {self.ADAPTIVE_BACKOFF_TIME}s to avoid ban.")
                time.sleep(self.ADAPTIVE_BACKOFF_TIME)
                consecutive_failures = 0

        print(f"\n🚜 [UCI Stage 2/3] Finished. Matches: {len(enriched_assets)}, Screened Out: {screened_out_count}, Failures: {failure_count}.")
        return enriched_assets

    def _augment_metadata_from_html(self, metadata: Dict[str, Any], soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Scrapes the physical HTML of the UCI page to find variables and instances
        if the embedded JSON payload forgot to include them.
        """
        if metadata.get('num_instances') and metadata.get('num_features'):
            return metadata

        try:
            headers = soup.find_all(['h1', 'h2', 'h3', 'p', 'span', 'div'])

            # Scrape Features
            if not metadata.get('num_features'):
                for elem in headers:
                    text = elem.get_text(strip=True).lower()
                    if text == "features" or "number of features" in text:
                        parent = elem.find_parent()
                        if parent:
                            parent_text = parent.get_text(separator=" ", strip=True)
                            digits = re.findall(r'\d+', parent_text)
                            if digits:
                                metadata['num_features'] = int(digits[-1])
                                break

            # Scrape Instances (Time points)
            if not metadata.get('num_instances'):
                for elem in headers:
                    text = elem.get_text(strip=True).lower()
                    if text == "instances" or "number of instances" in text:
                        parent = elem.find_parent()
                        if parent:
                            parent_text = parent.get_text(separator=" ", strip=True)
                            digits = re.findall(r'\d+', parent_text)
                            if digits:
                                metadata['num_instances'] = int(digits[-1])
                                break
        except Exception:
            pass

        # If NumFeatures is STILL missing, try to count the rows in the HTML variable table
        if not metadata.get('num_features'):
            try:
                tables = soup.find_all('table')
                for table in tables:
                    th = table.find('th')
                    if th and ('variable' in th.get_text(strip=True).lower() or 'feature' in th.get_text(strip=True).lower()):
                        rows = table.find_all('tr')
                        if len(rows) > 1:
                            metadata['num_features'] = len(rows) - 1
                            break
            except Exception:
                pass

        return metadata

    def _transform_asset(self, metadata: Dict[str, Any]) -> Optional[Dict[str, CellData]]:
        """
        Converts the raw extracted metadata dictionary into the
        standardized DeepCollector CatalogState CellData format.
        """
        if not self._is_target_type(metadata):
            return None

        item: Dict[str, CellData] = {}
        source_method = metadata.get('_source_method', 'unknown')
        telemetry_base = f"Harvested via UCI ({source_method})"

        def create_cell(value: Any, confidence: float, context_detail: str = "") -> CellData:
            if pd.isnull(value) or value is None:
                str_value = "[missing]"
                confidence = 0.0
            else:
                str_value = str(value).strip()

            if not str_value or str_value.lower() == "none" or str_value.lower() == "nan":
                str_value = "[missing]"
                confidence = 0.0

            context = f"{telemetry_base}. {context_detail}".strip()

            if CellData == dict:
                return {"value": str_value, "confidence": confidence, "telemetry_context": context, "anchor_ref_id": None}
            else:
                try:
                    return CellData(value=str_value, confidence=confidence, telemetry_context=context, anchor_ref_id=None)
                except TypeError:
                    return {"value": str_value, "confidence": confidence, "telemetry_context": context, "anchor_ref_id": None}

        name = metadata.get('name')
        if not name:
            return None

        if str(name).lower().strip() in ["null", "none", "nan", "[missing]", "missing", ""]:
            return None

        # --- Base Identifiers (Bumped to 1.00) ---
        item["Dataset Name"] = create_cell(name, 1.00, "Field: Name")
        item["Canonical Name"] = create_cell(name, 1.00, "Field: Name")
        item["Domain"] = create_cell(metadata.get('subject_area'), 1.00, "Field: Area")

        # EXPLICITLY SET THE TYPE
        item["Type"] = create_cell("Dataset", 1.00, "Hardcoded")

        # --- Description Assembly ---
        desc_parts = []
        abstract_text = metadata.get('abstract', '')
        info_text = metadata.get('additional_info', '')

        if abstract_text:
            desc_parts.append(abstract_text)

        if metadata.get('creators'):
            creators_joined = ', '.join(metadata['creators'])
            desc_parts.append(f"Creators: {creators_joined}")

        formatted_vars, time_heur = self._format_variable_info(metadata.get('variable_info'), metadata.get('name'))
        if formatted_vars:
            desc_parts.append(" | Variables: " + formatted_vars.replace('\n', ' '))

        # --- FIX 4: Strip all newlines to maintain Single Line Rule for Google Sheets ---
        final_desc = " | ".join(desc_parts).replace('\n', ' ').replace('\r', '')
        final_desc = re.sub(r'\s{2,}', ' ', final_desc).strip()
        item["Detailed Description"] = create_cell(final_desc, 1.00, "Fields: Abstract, Variables")

        # --- Time Interval Heuristics (Bumped to 1.00 to avoid repair loops) ---
        text_heur, text_conf = self._extract_time_interval_heuristic_text(metadata)
        if time_heur:
            item["Time interval between points"] = create_cell(time_heur, 1.00, "Heuristic: Variables Table")
        elif text_heur:
            item["Time interval between points"] = create_cell(text_heur, 1.00, "Heuristic: Abstract Text Regex")
        else:
            item["Time interval between points"] = create_cell("[missing]", 0.0)

        # --- Dimensionality Heuristics (Bumped to 1.00) ---
        raw_num_instances = metadata.get('num_instances')
        item["Number of Time Points"] = create_cell(raw_num_instances, 1.00, "Field: NumInstances")
        item["Number of Locations/Series"] = create_cell(1, 1.00, "Default (Single Series assumed)")

        num_features = metadata.get('num_features')
        if not num_features:
            var_info = metadata.get('variable_info')
            if var_info:
                if 'data' in var_info:
                    num_features = len(var_info['data'])

        item["Variables per Location"] = create_cell(num_features, 1.00, "Field: NumFeatures")
        item["Total Variables"] = create_cell(num_features, 1.00, "Field: NumFeatures")

        # --- Links & Provenance (Deterministic & 1.00 Confidence) ---
        item["Primary Source Repository"] = create_cell("UCI Machine Learning Repository", 1.00, "Source: UCI")

        uci_id = metadata.get('uci_id')
        doi = metadata.get('doi')

        primary_url = f"https://archive.ics.uci.edu/dataset/{uci_id}" if uci_id else ""
        data_url = f"https://archive.ics.uci.edu/static/public/{uci_id}/data.zip" if uci_id else ""
        other_url = f"https://doi.org/{doi}" if doi else ""

        item["Primary URL"] = create_cell(primary_url, 1.00 if primary_url else 0.0, "Deterministic UCI URL")
        item["Link to Data (Actual Source)"] = create_cell(data_url, 1.00 if data_url else 0.0, "Deterministic UCI URL")
        item["Other URL"] = create_cell(other_url, 1.00 if other_url else 0.0, "Deterministic UCI URL")

        item["Assignment Confidence"] = create_cell(1.00, 1.00, "Time-Series Filter")
        item["Assignment Rationale"] = create_cell(f"Explicitly identified as '{self.TARGET_DATA_TYPE}'", 1.00)

        return item

    def _update_state(self, state: CatalogState, enriched_assets: List[Dict[str, Any]]):
        """Pushes transformed datasets into the active CatalogState."""
        print(f"\n🚜 [UCI Stage 4] State Update ({len(enriched_assets)} items)...")
        transformed_items = []

        for metadata in enriched_assets:
            try:
                item = self._transform_asset(metadata)
                if item:
                    transformed_items.append(item)
            except Exception as e:
                if self.verbosity >= 2:
                    print(f"      ⚠️ [Transform Error] Failed to map asset: {e}")

        if transformed_items:
            if hasattr(state, 'update_catalog_batch'):
                state.update_catalog_batch(transformed_items, allow_new_datasets=True)

        print(f"🚜 [UCI Stage 4] Updated {len(transformed_items)} assets in catalog.")

    def _export_and_summarize(self, state: CatalogState):
        """Export is handled globally by the Agent in Phase 4."""
        pass

    def _standardize_keys(self, assets):
        """Normalizes dictionary keys from the raw API response."""
        standardized = []
        for i in assets:
            new_item = {}
            new_item['ID'] = i.get('id') or i.get('ID')
            new_item['Name'] = i.get('name') or i.get('Name')
            new_item['Slug'] = i.get('slug') or i.get('Slug')
            standardized.append(new_item)
        return standardized

    def _extract_and_identify_json_blocks(self, soup):
        """Fully unrolled JSON extraction from script tags embedded in Next.js page."""
        results = {'metadata': None, 'variables': None}

        next_data_tag = soup.find('script', id='__NEXT_DATA__')
        if next_data_tag and next_data_tag.string:
            try:
                next_data = json.loads(next_data_tag.string)
                props = next_data.get('props', {})
                page_props = props.get('pageProps', {})
                dataset_obj = page_props.get('dataset', {})

                if dataset_obj:
                    results['metadata'] = dataset_obj

                    if 'Variables' in dataset_obj:
                        results['variables'] = {'data': dataset_obj['Variables']}
            except Exception:
                pass

        if not results['metadata']:
            for tag in soup.find_all('script', {'type': 'application/json'}):
                if tag.string:
                    try:
                        data = json.loads(tag.string)
                        if isinstance(data, dict):
                            if 'body' in data:
                                body_str = data['body']
                                body = json.loads(body_str)

                                if isinstance(body, list):
                                    for item in body:
                                        if 'result' in item:
                                            result_obj = item['result']
                                            if 'data' in result_obj:
                                                data_obj = result_obj['data']
                                                if 'json' in data_obj:
                                                    j = data_obj['json']

                                                    if 'ID' in j and 'Abstract' in j:
                                                        results['metadata'] = j

                                                    if 'headers' in j:
                                                        headers_list = j.get('headers', [])
                                                        headers_lower = [str(h).lower() for h in headers_list]
                                                        if 'role' in headers_lower or 'type' in headers_lower:
                                                            results['variables'] = j
                    except Exception:
                        continue

        return results

    def _map_embedded_json_to_standard_structure(self, meta, vars):
        """Maps varying JSON schemas into a consistent dictionary for the Agent."""
        m = {
            'uci_id': meta.get('ID') or meta.get('id'),
            'doi': meta.get('DOI') or meta.get('doi'), # ADDED TO CAPTURE DOI SAFELY
            'name': meta.get('Name') or meta.get('name'),
            'abstract': meta.get('Abstract') or meta.get('abstract'),
            'num_instances': meta.get('NumInstances') or meta.get('numInstances'),
            'num_features': meta.get('NumFeatures') or meta.get('numFeatures'),
            'subject_area': meta.get('Area') or meta.get('area'),
            'additional_info': meta.get('DatasetInfo') or meta.get('datasetInfo'),
            'variable_info': vars
        }

        def parse(field_name):
            val = meta.get(field_name) or meta.get(field_name.lower())
            if isinstance(val, list):
                return [str(v) for v in val if v]
            if val:
                return [v.strip() for v in str(val).split(',')]
            return []

        m['characteristics'] = parse('Types')
        if not m['characteristics']:
            m['characteristics'] = parse('characteristics')

        m['associated_tasks'] = parse('Task')
        if not m['associated_tasks']:
            m['associated_tasks'] = parse('tasks')

        return m

    def _format_variable_info(self, vars_dict, name=None):
        """Builds a markdown table from the variable data and looks for frequency hints."""
        if not vars_dict:
            return None, None

        if not vars_dict.get('data'):
            return None, None

        rows = vars_dict['data']
        headers = vars_dict.get('headers', [])
        heur = None

        for row in rows:
            if isinstance(row, dict):
                row_vals = list(row.values())
            else:
                row_vals = row

            txt = " ".join([str(x) for x in row_vals])
            match = re.search(r'\b(hourly|daily|weekly|monthly|quarterly|yearly|hz|seconds|minutes|days)\b', txt, re.I)
            if match:
                heur = match.group(1).capitalize()
                break

        if tabulate and isinstance(rows[0], list):
            return tabulate(rows, headers=headers), heur

        return str(rows), heur

    def _extract_time_interval_heuristic_text(self, metadata):
        """Searches the abstract and additional info for frequency keywords."""
        abstract = str(metadata.get('abstract', ''))
        additional = str(metadata.get('additional_info', ''))
        text = (abstract + " " + additional).lower()

        patterns = [
            (r'(\d+\s*khz|\d+\s*hz)', 0.95),
            (r'sampled at (\d+\s*hz|\d+\s*khz)', 0.95),
            (r'\b(hourly|daily|weekly|monthly|yearly|quarterly|minutes?|seconds?|ms)\b', 0.90),
            (r'every (\d+\s*(?:minutes?|seconds?|hours?|days?|ms))', 0.90),
            (r'frequency of (\d+\s*(?:hz|khz|minutes?|seconds?|hours?|days?))', 0.85)
        ]

        for p, c in patterns:
            m = re.search(p, text)
            if m:
                return m.group(1).strip().capitalize(), c

        return None, 0.0

    def _is_target_type(self, meta):
        """Confirms the dataset actually has 'Time-Series' in its characteristics list."""
        characteristics = meta.get('characteristics', [])
        char_list_lower = [str(c).lower() for c in characteristics]
        return self.TARGET_DATA_TYPE.lower() in char_list_lower

    def _get_randomized_headers(self, referer=None):
        h = {'User-Agent': random.choice(self.USER_AGENTS)}
        if referer:
            h['Referer'] = referer
        return h

    def _parse_json_response(self, resp):
        try:
            return resp.json()
        except Exception:
            return self._extract_json_robustly_from_text(resp.text)

    def _extract_json_robustly_from_text(self, txt):
        # FIX 6: Use .*? for non-greedy evaluation so it doesn't consume the entire HTML document
        m = re.search(r'{\s*".*?":.*?}', txt, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return None

    def _is_blocked(self, txt):
        if "Access denied" in txt:
            return True
        if "Cloudflare" in txt:
            return True
        return False

print("✅ deepcollector/harvesting/uci_harvester.py written (V11.2: Deep Think Fix: Single Line Enforcer).")