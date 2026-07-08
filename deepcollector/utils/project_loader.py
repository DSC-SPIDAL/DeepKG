# =============================================================================
# V32: Project Loader (Full Restoration + Gspread V5/V6 API Safe Fetch)
# =============================================================================
import sys
import re
import pandas as pd
import traceback
from tenacity import retry, stop_after_attempt, wait_exponential

class ExternalKnowledge:
    def __init__(self, config, gc_client, verbosity=1):
        self.config = config
        self.gc = gc_client
        self.verbosity = verbosity
        self.sheet_id = getattr(config, 'GOOGLE_SHEET_PROJECT_LIST_ID', None)
        self.projects = []
        self.headers = []
        self.worksheet = None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10))
    def _fetch_sheet_data(self):
        try:
            sh = self.gc.open_by_key(self.sheet_id)

            # Safely grab the first worksheet regardless of gspread version
            try:
                self.worksheet = sh.get_worksheet(0)
                if self.worksheet is None:
                    self.worksheet = sh.worksheets()
            except Exception:
                self.worksheet = sh.worksheets()

            # CRITICAL FIX: Bulletproof cascade to support both Gspread v5 and v6+
            try:
                if hasattr(self.worksheet, 'get_values'):
                    return self.worksheet.get_values(value_render_option='FORMULA')
                else:
                    try:
                        return self.worksheet.get_all_values(value_render_option='FORMULA')
                    except TypeError:
                        return self.worksheet.get_all_values()
            except Exception:
                # Ultimate fallback
                return self.worksheet.get_all_values()

        except Exception as e:
            if self.verbosity >= 1:
                print(f"      ⚠️ [Google Sheets API] Fetch failed: {type(e).__name__} - {e}")
            raise e

    def load(self):
        if self.verbosity >= 1:
            print("🌐 [ExternalKnowledge] Initialization complete. Ready to load sources.")
        if not self.gc or not self.sheet_id:
            if self.verbosity >= 1:
                print("    ⚠️ [ExternalKnowledge] Google credentials or Sheet ID missing.")
            return

        if self.verbosity >= 1:
            print(f"💡 [ExternalKnowledge] Loading Canonical Projects from Sheet ID: {self.sheet_id[:8]}...{self.sheet_id[-4:]}")

        try:
            data = self._fetch_sheet_data()

            if data and len(data) > 1:
                self.headers = [str(h).strip() for h in data]
                self.projects = [dict(zip(self.headers, row)) for row in data[1:]]
                if self.verbosity >= 1:
                    print(f"    ✅ [ExternalKnowledge] Loaded {len(self.projects)} records.")
                    if self.verbosity >= 2:
                        print(f"    📋 Detected Headers: {self.headers}")

                self._ensure_proj_lost()
            else:
                if self.verbosity >= 1: print("    ⚠️ [ExternalKnowledge] Canonical Sheet is empty.")
        except Exception as e:
            if self.verbosity >= 1:
                print(f"    ❌ [ExternalKnowledge] Failed to load from canonical sheet after retries: {e}")
            raise e

    def _ensure_proj_lost(self):
        has_lost = False
        for p in self.projects:
            pid = ""
            for k, v in p.items():
                if any(term in str(k).lower() for term in ["canonical", "projectid", "id"]):
                    pid = str(v).strip().upper()
                    break

            if pid == "PROJ_LOST" or "LOST" in pid:
                has_lost = True
                break

        if not has_lost and self.worksheet and self.headers:
            if self.verbosity >= 1:
                print("    🗂️ 'PROJ_LOST' not found in canonical list. Appending it now...")
            try:
                new_row = []
                for h in self.headers:
                    h_lower = str(h).lower()
                    if "canonical" in h_lower or "id" in h_lower: new_row.append("PROJ_LOST")
                    elif "name" in h_lower: new_row.append("Lost / Incidental Datasets")
                    elif "link" in h_lower or "url" in h_lower: new_row.append("")
                    elif "comment" in h_lower or "desc" in h_lower: new_row.append("Auto-generated sink for orphans and hallucinations.")
                    elif "method" in h_lower or "type" in h_lower: new_row.append("System")
                    else: new_row.append("")

                self.worksheet.append_row(new_row, value_input_option='USER_ENTERED')
                self.projects.append(dict(zip(self.headers, new_row)))
                if self.verbosity >= 1:
                    print("    ✅ 'PROJ_LOST' successfully added to external project list.")
            except Exception as e:
                if self.verbosity >= 1:
                    print(f"    ⚠️ Could not append PROJ_LOST to canonical sheet: {e}")

class ProjectLoader:
    def __init__(self, config, authenticated_gc=None, verbosity=1):
        self.config = config
        self.gc = authenticated_gc
        self.verbosity = verbosity

    def _clean_formula(self, text):
        """Extracts display text from =HYPERLINK formulas so Names/Comments aren't messy."""
        text = str(text).strip()
        m = re.search(r'=HYPERLINK\("([^"]+)"\s*,\s*"([^"]+)"\)', text, re.IGNORECASE)
        if m: return m.group(2)

        m2 = re.search(r'=HYPERLINK\("([^"]+)"\)', text, re.IGNORECASE)
        if m2: return m2.group(1)

        return text

    def resolve_configuration(self):
        raw_id = str(getattr(self.config, 'CURRENT_PROJECT_ID', '')).strip().upper()

        if not raw_id:
            sys.exit("🛑 ABORTING: CURRENT_PROJECT_ID is missing or empty in configuration.")

        if self.verbosity >= 1:
            print(f"🔄 [ProjectLoader] Dynamically loading configuration for ID: {raw_id}...")

        target_id = raw_id if raw_id.startswith("PROJ_") else f"PROJ_{raw_id}"
        self.config.CURRENT_PROJECT_ID = target_id

        ek = ExternalKnowledge(self.config, self.gc, self.verbosity)
        ek.load()

        if not ek.projects:
            sys.exit("🛑 ABORTING: Could not load any projects from the canonical sheet.")

        matched_p = None
        available_ids = []

        for p in ek.projects:
            pid_raw = ""
            for k, v in p.items():
                k_lower = str(k).lower()
                if any(term in k_lower for term in ["canonical", "projectid", "id"]):
                    pid_raw = self._clean_formula(v).upper()
                    break

            if not pid_raw and p:
                pid_raw = self._clean_formula(list(p.values())).upper()

            pname_raw = ""
            for k, v in p.items():
                k_lower = str(k).lower()
                if "name" in k_lower and "canonical" not in k_lower:
                    pname_raw = self._clean_formula(v).upper()
                    break

            pid = f"PROJ_{pid_raw}" if pid_raw and not pid_raw.startswith("PROJ_") else pid_raw
            pname_as_id = f"PROJ_{pname_raw}" if pname_raw else ""

            if pid: available_ids.append(pid)
            elif pname_as_id: available_ids.append(pname_as_id)

            if target_id == pid or target_id == pname_as_id:
                matched_p = p
                break

        if not matched_p:
            print(f"\n📋 Detected Headers in Sheet: {ek.headers}")
            print(f"📋 Available Projects in Sheet: {', '.join(available_ids[:10])}... (Showing first 10)")
            sys.exit(f"\n🛑 ABORTING: Target project '{target_id}' was NOT FOUND in the Canonical Projects Sheet.\n"
                     f"Please ensure it is added to the Google Sheet.")

        # =====================================================================
        # OMNI-URL EXTRACTOR: Scan ALL columns in the row for http/https links OR /content/drive/ paths
        # This completely ignores headers, immune to shifted columns or typos.
        # =====================================================================
        extracted_urls = []
        for k, v in matched_p.items():
            val_str = str(v)
            found_urls = re.findall(r'(https?://[^\s,\">|\]\)]+|/content/drive/[^\s,\">|\]\)]+)', val_str)
            for u in found_urls:
                clean_u = u.strip(';,"\'()[]')
                if clean_u and clean_u not in extracted_urls:
                    extracted_urls.append(clean_u)

        if not extracted_urls and target_id != "PROJ_LOST":
            row_data_str = " | ".join([f"{k}: {v}" for k, v in matched_p.items()])
            print(f"\n⚠️ WARNING: Project '{target_id}' was found, but NO valid http/https URLs or /content/drive/ paths could be extracted from ANY column.\n"
                  f"   Row Data Seen by Agent: {row_data_str}\n"
                  f"   Please ensure URLs start with http://, https://, or /content/drive/")

        self.config.INITIAL_URLS = extracted_urls

        pname_raw = ""
        for k, v in matched_p.items():
            k_lower = str(k).lower()
            if "name" in k_lower and "canonical" not in k_lower:
                pname_raw = self._clean_formula(v)
                break

        self.config.CURRENT_PROJECT_NAME = pname_raw or target_id.replace("PROJ_", "")

        method_str = ""
        for k, v in matched_p.items():
            if "method" in str(k).lower():
                method_str = self._clean_formula(v)
                break

        try:
            clean_val = re.sub(r'[^0-9]', '', method_str)
            method_val = int(clean_val) if clean_val else 1
        except ValueError:
            method_val = 1

        self.config.PROJECT_METHOD = method_val
        self.config.SCRAPING_METHOD = method_val

        desc = ""
        for k, v in matched_p.items():
            if "comment" in str(k).lower() or "description" in str(k).lower() or "summary" in str(k).lower():
                desc = self._clean_formula(v)
                break

        self.config.PROJECT_CONTEXT = f"{self.config.CURRENT_PROJECT_NAME}: {desc}" if desc else self.config.CURRENT_PROJECT_NAME

        if self.verbosity >= 1:
            print(f"    ✅ [ProjectLoader] Configuration loaded. Method: {method_val}, URLs: {len(self.config.INITIAL_URLS)}.")
            for i, u in enumerate(self.config.INITIAL_URLS):
                print(f"    🔗 Source URL [{i+1}]: {u}")

        return True

print("✅ deepcollector/utils/project_loader.py written (Full Restoration + Gspread V5/V6 API Safe Fetch).")