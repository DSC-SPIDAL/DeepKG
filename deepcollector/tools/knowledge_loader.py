# V18.5 (Based on V18.2): Improved Google Sheets Error Clarity
import pandas as pd
import re
from typing import Any, Dict, List, Optional
try:
    import gspread
    # V18.5: Import specific exceptions for better error handling
    from gspread.exceptions import SpreadsheetNotFound, APIError
except ImportError:
    gspread = None
    SpreadsheetNotFound = Exception
    APIError = Exception

# Robust profiler import
try:
    from deepcollector.utils.profiler import profiler
except ImportError:
    class DummyProfiler:
        def track(self, category):
            def decorator(func): return func
            return decorator
    profiler = DummyProfiler()


class ExternalKnowledgeManager:
    """
    Manages the loading and parsing of external knowledge sources
    (Expert Hints and Canonical Projects) from Google Sheets.
    """
    def __init__(self, config: Any):
        self.config = config
        self.verbosity = getattr(config, 'VERBOSITY_LEVEL', 1)
        # Robust access to GSPREAD_AVAILABLE
        self.GSPREAD_AVAILABLE = getattr(config, 'GSPREAD_AVAILABLE', False)
        self.gc = None
        self.is_initialized = False

        # Configuration IDs
        self.hints_sheet_id = getattr(config, 'GOOGLE_SHEET_HINTS_ID', None)
        self.projects_sheet_id = getattr(config, 'GOOGLE_SHEET_PROJECT_LIST_ID', None)

        # Data storage
        self.expert_hints_data: List[Dict[str, str]] = []
        self.canonical_projects_data: List[Dict[str, Any]] = []

    # (initialize, load_all remain the same)
    def initialize(self, authenticated_gc: Any):
        """Initializes the connection using a pre-authenticated gspread client."""
        if not self.GSPREAD_AVAILABLE or gspread is None:
            # Suppress warning if verbosity is low, as this might be expected
            if self.verbosity >= 1:
                print("⚠️ [ExternalKnowledge] Google Sheets libraries not available. Cannot load external knowledge.")
            return False

        # Check if any sheets are configured
        if not self.hints_sheet_id and not self.projects_sheet_id:
            # This is common if only KB is used, so suppress the log message.
            return True # Not a failure, just not configured

        self.gc = authenticated_gc
        self.is_initialized = True
        if self.verbosity >= 1:
            print(f"🌐 [ExternalKnowledge] Initialization complete. Ready to load sources.")
        return True

    @profiler.track("ExternalKnowledge: Load All")
    def load_all(self):
        """Loads data from all configured external sources."""
        if not self.is_initialized or not self.gc:
            return

        if self.hints_sheet_id:
            self._load_expert_hints()

        if self.projects_sheet_id:
            self._load_canonical_projects()

    # V18.5: Updated error handling for clarity
    def _load_spreadsheet_values(self, sheet_id: str, source_name: str) -> Optional[List[List[str]]]:
        """Helper to safely load spreadsheet values using gspread."""
        # V18.5: Check explicitly for placeholder text before attempting API call.
        if not sheet_id or "YOUR_GOOGLE_SHEET_ID_HERE" in sheet_id:
             # Increased verbosity level (>=0) for this critical configuration check
             if self.verbosity >= 0:
                print(f"⚠️ [ExternalKnowledge] Skipping {source_name}. Sheet ID is missing or placeholder text is present.")
             return None

        if self.verbosity >= 1:
            # Display only the first few characters of the ID for security/brevity
            display_id = f"{sheet_id[:8]}...{sheet_id[-4:]}" if len(sheet_id) > 12 else sheet_id
            print(f"💡 [ExternalKnowledge] Loading {source_name} from Sheet ID: {display_id}...")

        try:
            spreadsheet = self.gc.open_by_key(sheet_id)
            # Assumption: Data is in the first worksheet (index 0)
            worksheet = spreadsheet.get_worksheet(0)
            data = worksheet.get_all_values() # Get as list of lists (strings)

            if not data:
                 if self.verbosity >= 1:
                    print(f"    ⚠️ [ExternalKnowledge] {source_name} sheet is empty.")
                 return None

            if self.verbosity >= 1:
                # Handle case where data might only contain a header
                record_count = max(0, len(data)-1)
                print(f"    ✅ [ExternalKnowledge] Loaded {record_count} records from {source_name} (excluding header).")
            return data

        # V18.5: Specific exception handling
        except SpreadsheetNotFound:
            if self.verbosity >= 0:
                # This error specifically means the ID is invalid OR the user lacks permission.
                display_id = f"{sheet_id[:8]}...{sheet_id[-4:]}" if len(sheet_id) > 12 else sheet_id
                print(f"❌ [ExternalKnowledge] CRITICAL: {source_name} spreadsheet not found (ID: {display_id}).")
                print(f"   -> Check if the ID is correct AND that the authenticated Google account has access rights.")
            return None
        except APIError as e:
             if self.verbosity >= 0:
                # Handle other potential API errors (e.g., quota exceeded)
                print(f"❌ [ExternalKnowledge] Google Sheets API Error while loading {source_name}: {e}")
             return None
        except Exception as e:
            if self.verbosity >= 0:
                print(f"❌ [ExternalKnowledge] Unexpected error loading {source_name}: {e}")
            return None

    # (_load_expert_hints remains the same)
    def _load_expert_hints(self):
        """Loads and parses the Expert Hints spreadsheet."""
        data = self._load_spreadsheet_values(self.hints_sheet_id, "Expert Hints")
        if data is None: return

        # Expected format: Hint Type | Keyword | Hint 1 | Hint 2 | ...

        for index, row in enumerate(data[1:]): # Skip header
            if len(row) < 3:
                continue # Need at least Type, Keyword, and one Hint

            hint_type = row[0].strip()
            keyword = row[1].strip()

            if not hint_type or not keyword:
                continue

            # Remaining columns are hints
            hints = [h.strip() for h in row[2:] if h.strip()]

            # We store each hint individually for better RAG granularity
            for i, hint_text in enumerate(hints):
                # Store data in a format suitable for RAG indexing
                self.expert_hints_data.append({
                    'type': hint_type,
                    'keyword': keyword,
                    'hint_text': hint_text,
                    # Create a unique ref_id for indexing (using row index+1 because we skipped header)
                    'ref_id': f"HINT:{keyword[:30]}_{index+1}_{i}"
                })

    # (_load_canonical_projects remains the same as V18.2)
    def _load_canonical_projects(self):
        """Loads and parses the Canonical Projects spreadsheet."""
        data = self._load_spreadsheet_values(self.projects_sheet_id, "Canonical Projects")
        if data is None: return

        # V18.2 Expected format:
        # A: Canonical Project ID
        # B: Project Name (Informal Names)
        # C: Links (comma-separated URLs)
        # D: Project Context Description (Optional)
        # E: Project Method (Optional, defaults to 1)

        EXPECTED_COLS = 5

        for index, row in enumerate(data[1:]): # Skip header
            # Ensure we have 5 columns, padding if necessary
            if len(row) < EXPECTED_COLS:
                 # Pad the row if columns are missing entirely
                 row += [''] * (EXPECTED_COLS - len(row))

            # 1. Parse Columns
            canonical_id = row[0].strip().upper()
            project_name = row[1].strip()
            links_str = row[2].strip()
            context_desc = row[3].strip()
            method_str = row[4].strip()

            if not canonical_id:
                continue

            # 2. Process Data
            # Parse Method
            try:
                method = int(method_str) if method_str else 1
            except ValueError:
                method = 1

            # Parse Comma-Separated URLs
            initial_urls_list = [url.strip() for url in links_str.split(',') if url.strip() and re.match(r'^https?://', url.strip())]

            # Construct Full Context (Name: Description)
            if project_name and context_desc:
                full_project_context = f"{project_name}: {context_desc}"
            else:
                # Fallback if name or description is missing
                full_project_context = project_name or context_desc or canonical_id

            # 3. Store Data (used by ProjectLoader and RAG indexing)
            self.canonical_projects_data.append({
                'project_id': canonical_id,
                'project_name': project_name,
                'initial_urls': initial_urls_list, # Stored as list for ProjectLoader
                'project_context': full_project_context,
                'method': method,
                'ref_id': f"PROJ_DEF:{canonical_id}",
                # Fields specifically for RAG indexing format (backward compatibility)
                'informal_names': project_name,
                'links': links_str, # Kept as string for RAG indexing display
            })

print("✅ deepcollector/tools/knowledge_loader.py written.")