import pandas as pd
import io
import zipfile
import requests
# Import necessary typing hints
from typing import Dict, Any, Optional, Tuple, List
from urllib.parse import urljoin
import re
import textwrap
import sys
import time

# Attempt to import BeautifulSoup
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

# Import utilities
# Robust imports to handle potential failures if previous cells didn't run
try:
    from deepcollector.utils.profiler import profiler
    from deepcollector.utils.initialization import get_network_retry_strategy, HEADERS
except ImportError as e:
    print(f"❌ CRITICAL: Failed to import utilities in ddi.py: {e}.")
    # Define placeholders
    class DummyProfiler:
        def track(self, category):
            def decorator(func): return func
            return decorator
    profiler = DummyProfiler()
    get_network_retry_strategy = lambda x: lambda f: f
    HEADERS = {}


class DataInspectionTools:
    """Handles direct inspection of data files and navigation of directory listings."""

    # (Class attributes identical to previous versions)
    PARSABLE_EXTENSIONS = ['.csv', '.tsv', '.txt', '.data']
    ARCHIVE_EXTENSIONS = ['.zip']
    COMPRESSION_EXTENSIONS = ['.gz', '.bz2']

    DIRECTORY_LISTING_PRIORITY = {
        '.zip': 100, '.tar.gz': 95, '.tgz': 95, '.rar': 90,
        '.data': 80, '.csv': 75, '.txt': 70, '.arff': 60,
        '.names': 10, '.info': 10,
    }

    IGNORE_PATTERNS = [
        r'^/$', r'^\\?C=', r'parent directory', r'name', r'last modified', r'size', r'description',
        r'\\.pdf$', r'\\.md$', r'readme', r'index\\.html', r'license',
        r'\\.jpg$', r'\\.png$', r'\\.gif$'
    ]

    class SizeLimitExceeded(Exception):
        pass

    def __init__(self, config: Any):
        self.config = config
        self.verbosity = getattr(config, 'VERBOSITY_LEVEL', 1)
        # Initialize the retry strategy dynamically based on config
        self.NETWORK_RETRY_STRATEGY = get_network_retry_strategy(self.verbosity)

        # Apply the retry decorator dynamically in __init__.
        # We wrap the methods that perform network operations.
        self._fetch_resource_head = self.NETWORK_RETRY_STRATEGY(self._fetch_resource_head)
        self._fetch_file_preview = self.NETWORK_RETRY_STRATEGY(self._fetch_file_preview)
        self._fetch_full_file = self.NETWORK_RETRY_STRATEGY(self._fetch_full_file)

    # (The implementation of the methods below remains the same as the previous stable version)

    # --- Main Entry Point and Dispatcher ---

    @profiler.track("Tool: Data File Inspection")
    def inspect_file(self, url: str, timeout=45, recursion_depth=0) -> Dict[str, Any]:
        """Main entry point: Resolves the URL (handling directories) and executes inspection strategy."""

        MAX_RECURSION = 3
        start_time = time.time()

        # Use injected configuration instead of globals
        if not getattr(self.config, 'DATA_INSPECTION_ENABLED', True):
            return {"status": "disabled", "error": "Data inspection is disabled in configuration."}

        if recursion_depth > MAX_RECURSION:
            if self.verbosity >= 1:
                print(f"    ⚠️ [Data Inspector] Maximum recursion depth exceeded for URL: {url}")
            return {"status": "error", "error": "Maximum redirection/recursion depth exceeded."}

        if self.verbosity >= 1:
            prefix = "    " * recursion_depth + ("" if recursion_depth == 0 else "↪️ ")
            print(f"{prefix}🔬 [Data Inspector] Analyzing resource: {textwrap.shorten(url, width=80)}")

        # Basic URL validation (Access MISSING_DATA_PLACEHOLDERS via config)
        MISSING_DATA = getattr(self.config, 'MISSING_DATA_PLACEHOLDERS', set())
        if not url or url.lower() in MISSING_DATA:
             return {"status": "error", "error": "Invalid or missing URL provided."}

        # Heuristic: Handle GitHub URLs
        original_url = url
        if "github.com" in url and "/blob/" in url:
            url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
            if self.verbosity >= 2:
                 print(f"    🔧 [Data Inspector] Converted GitHub URL to raw: {url}")

        # --- Step 1: Determine Resource Type ---
        try:
            # Call the wrapped method
            headers, initial_content = self._fetch_resource_head(url, timeout)
            content_type = headers.get('Content-Type', '').lower()

        except Exception as e:
            if self.verbosity >= 1:
                print(f"    ⚠️ [Data Inspector] Failed to fetch resource head: {type(e).__name__} - {e}")

            # Handle HTTP errors specifically for feedback mechanism
            is_http_error = False
            try:
                if isinstance(e, requests.exceptions.HTTPError):
                    is_http_error = True
            except (AttributeError, TypeError):
                 pass

            if is_http_error:
                 return {"status": "error", "error": f"HTTP Error during fetch: {e}"}

            return {"status": "error", "error": f"Failed to analyze resource: {e}"}


        # --- Step 2: Handle Directory Listings ---
        is_html = 'text/html' in content_type

        if is_html and BeautifulSoup and urljoin:
            content_str = initial_content.decode(errors='ignore')
            # Basic heuristic to detect directory listings
            is_directory_listing = ("Index of" in content_str[:500] or "Directory Listing" in content_str[:500])

            if is_directory_listing:
                if self.verbosity >= 1:
                    print(f"    📂 [Data Inspector] Directory listing detected. Scraping for data files...")

                data_url = self._scrape_directory_listing(content_str, url)

                if data_url:
                    if self.verbosity >= 1:
                        print(f"    💡 [Data Inspector] Found candidate data file: {textwrap.shorten(data_url, width=80)}. Redirecting inspection.")
                    # Recursively call inspect_file
                    return self.inspect_file(data_url, timeout, recursion_depth + 1)
                else:
                    if self.verbosity >= 1:
                        print(f"    ⚠️ [Data Inspector] Could not find suitable data files in the directory listing.")
                    return {"status": "unsupported", "error": "Directory listing found, but no suitable data files detected."}

            else:
                # It's an HTML page but not a directory listing
                if self.verbosity >= 1:
                    print(f"    ℹ️ [Data Inspector] HTML page detected (not a directory listing). Skipping inspection.")
                return {"status": "unsupported", "error": "URL points to an HTML page (e.g., homepage), not a direct data file or directory listing."}


        # --- Step 3: Handle Direct Files (ZIP or Delimited) ---
        is_zip = 'application/zip' in content_type or url.lower().endswith('.zip')

        if is_zip:
            result = self._inspect_zip_archive(url, timeout)
        else:
            result = self._inspect_direct_file(url, timeout)

        # Add timing information
        if result:
            result["duration_s"] = round(time.time() - start_time, 3)

        return result


    # --- Directory Listing Navigation ---

    @staticmethod
    def _scrape_directory_listing(html_content: str, base_url: str) -> Optional[str]:
        """Parses HTML directory listing and uses heuristics to find the best data file URL."""
        if not BeautifulSoup or not urljoin:
            return None

        # Determine parser
        try:
            import lxml
            parser = 'lxml'
        except ImportError:
            parser = 'html.parser'

        soup = BeautifulSoup(html_content, parser)
        candidates = []

        # Compile the ignore patterns regex
        ignore_regex = re.compile("|".join(DataInspectionTools.IGNORE_PATTERNS), re.IGNORECASE)

        for link in soup.find_all('a'):
            href = link.get('href')
            text = link.get_text(strip=True)

            if not href or not text:
                continue

            # Filter out irrelevant links
            if ignore_regex and (ignore_regex.search(href) or ignore_regex.search(text)):
                continue

            # Determine the priority based on extension
            priority = 0
            file_extension = None
            for ext, prio in DataInspectionTools.DIRECTORY_LISTING_PRIORITY.items():
                if href.lower().endswith(ext):
                    if prio > priority:
                        priority = prio
                        file_extension = ext

            if priority == 0:
                # Ignore subdirectories
                if href.endswith('/'):
                    continue
                # Default priority for unrecognized extensions
                priority = 5

            candidates.append({
                'url': urljoin(base_url, href),
                'priority': priority,
                'extension': file_extension
            })

        if not candidates:
            return None

        # Sort by priority and return the top candidate
        candidates.sort(key=lambda x: x['priority'], reverse=True)
        return candidates[0]['url']


    # --- Strategy Implementations (ZIP and Direct) ---

    def _inspect_direct_file(self, url: str, timeout: int) -> Dict[str, Any]:
        """Strategy: Inspects direct (non-ZIP) files (CSV, GZ, TXT) using efficient preview."""
        try:
            # Access configuration via self.config
            MAX_DOWNLOAD_PREVIEW_BYTES = getattr(self.config, 'MAX_DOWNLOAD_PREVIEW_BYTES', 1024*1024)
            # Call the wrapped method
            file_content = self._fetch_file_preview(url, MAX_DOWNLOAD_PREVIEW_BYTES, timeout)

        except DataInspectionTools.SizeLimitExceeded as e:
            return {"status": "error", "error": str(e)}
        except Exception as e:
            if self.verbosity >= 1:
                print(f"    ⚠️ [Data Inspector] Failed to fetch file preview: {type(e).__name__}")
            return {"status": "error", "error": f"Failed to download preview: {e}"}

        if not file_content:
            return {"status": "error", "error": "Downloaded content is empty."}

        return self._process_delimited_content(file_content, url)


    def _inspect_zip_archive(self, url: str, timeout: int) -> Dict[str, Any]:
        """Strategy: Inspects ZIP archives."""
        MAX_DOWNLOAD_ARCHIVE_BYTES = getattr(self.config, 'MAX_DOWNLOAD_ARCHIVE_BYTES', 50*1024*1024)

        if self.verbosity >= 1:
            print(f"    📦 [Data Inspector ZIP] Starting download (Limit: {MAX_DOWNLOAD_ARCHIVE_BYTES/(1024*1024):.1f} MB)...")

        try:
            # Call the wrapped method
            archive_content = self._fetch_full_file(url, MAX_DOWNLOAD_ARCHIVE_BYTES, timeout)
        except DataInspectionTools.SizeLimitExceeded as e:
             return {"status": "error", "error": str(e)}
        except Exception as e:
            if self.verbosity >= 1:
                print(f"    ⚠️ [Data Inspector ZIP] Failed to download archive: {e}")
            return {"status": "error", "error": f"Failed to download archive: {e}"}

        if not archive_content:
             return {"status": "error", "error": "Downloaded archive is empty."}

        # Analyze the archive contents
        try:
            with zipfile.ZipFile(io.BytesIO(archive_content)) as zf:
                file_list = zf.infolist()

                if not file_list:
                    return {"status": "error", "error": "ZIP archive is empty."}

                candidates = []
                for f_info in file_list:
                    filename = f_info.filename
                    # Skip directories and hidden/macOS artifacts
                    if f_info.is_dir() or filename.startswith('__MACOSX/') or filename.split('/')[-1].startswith('.'):
                        continue

                    # Check if the file extension is parsable
                    if any(filename.lower().endswith(ext) for ext in DataInspectionTools.PARSABLE_EXTENSIONS):
                        candidates.append(f_info)

                if not candidates:
                    # Check for compressed files within the ZIP
                    if any(f.filename.lower().endswith(tuple(DataInspectionTools.COMPRESSION_EXTENSIONS)) for f in file_list):
                         return {"status": "unsupported", "error": f"ZIP contains compressed files (e.g., .gz) which require specialized handling."}
                    return {"status": "unsupported", "error": f"ZIP archive does not contain parsable files ({', '.join(DataInspectionTools.PARSABLE_EXTENSIONS)})."}

                # Heuristic: Select the largest candidate file
                candidates.sort(key=lambda x: x.file_size, reverse=True)
                target_file_info = candidates[0]

                if self.verbosity >= 1:
                    print(f"    📄 [Data Inspector ZIP] Inspecting largest file: '{target_file_info.filename}'...")

                # Extract the target file content
                with zf.open(target_file_info.filename) as f:
                    extracted_content = f.read()

                # Process the extracted content
                results = self._process_delimited_content(extracted_content, target_file_info.filename)

                # Add context to the results
                if results and results.get("status") == "success":
                    results["file_type"] += " (via ZIP)"
                    results["extracted_file_name"] = target_file_info.filename

                return results

        except zipfile.BadZipFile:
            return {"status": "error", "error": "Invalid or corrupted ZIP file."}
        except Exception as e:
            return {"status": "error", "error": f"Error during ZIP processing: {e}"}

    # --- Fetch Helpers ---
    # Note: These are the methods that the retry strategy wraps (applied in __init__)

    def _fetch_resource_head(self, url: str, timeout: int, preview_bytes=1024*5) -> Tuple[Dict[str, str], bytes]:
        """Fetches headers and a small preview of the content."""
        response = requests.get(url, headers=HEADERS, timeout=timeout, stream=True, allow_redirects=True)
        response.raise_for_status()

        response_headers = response.headers
        content_preview = b""
        try:
            # Iterate over content to get a preview
            for chunk in response.iter_content(chunk_size=1024):
                content_preview += chunk
                if len(content_preview) >= preview_bytes:
                    break
        finally:
            # Ensure the connection is closed
            response.close()

        return response_headers, content_preview[:preview_bytes]

    def _fetch_file_preview(self, url: str, max_bytes: int, timeout: int) -> bytes:
        """Downloads only the beginning segment of a file using HTTP Range headers."""
        headers = HEADERS.copy()
        # Request a specific range of bytes
        headers['Range'] = f'bytes=0-{max_bytes-1}'
        response = requests.get(url, headers=headers, timeout=timeout, stream=True, allow_redirects=True)

        # 200 OK (if server ignores Range) or 206 Partial Content (if server respects Range)
        if response.status_code not in [200, 206]:
            response.raise_for_status()

        content = b""
        try:
            for chunk in response.iter_content(chunk_size=8192):
                content += chunk
                if len(content) >= max_bytes:
                    break
        finally:
            response.close()

        return content[:max_bytes]

    def _fetch_full_file(self, url: str, max_bytes: int, timeout: int) -> bytes:
        """Downloads the full file content, enforcing a maximum size limit."""

        # Step 1: Check Content-Length using a HEAD request (Optimization)
        try:
            # Short timeout for HEAD request
            head_response = requests.head(url, headers=HEADERS, timeout=5, allow_redirects=True)
            if head_response.status_code == 200 and 'Content-Length' in head_response.headers:
                try:
                    file_size = int(head_response.headers['Content-Length'])
                    if file_size > max_bytes:
                        raise DataInspectionTools.SizeLimitExceeded(f"File size ({file_size / (1024*1024):.2f} MB) exceeds limit based on HEAD request.")
                except ValueError:
                     pass # Ignore if Content-Length is not a valid integer
        except Exception as e:
             if isinstance(e, DataInspectionTools.SizeLimitExceeded):
                  raise e
             # If HEAD fails (e.g., times out or server doesn't support it), proceed to streaming download

        # Step 2: Stream the download
        response = requests.get(url, headers=HEADERS, timeout=timeout, stream=True, allow_redirects=True)
        response.raise_for_status()

        content = b""
        try:
            for chunk in response.iter_content(chunk_size=8192):
                content += chunk
                if len(content) > max_bytes:
                    # Abort the download if the limit is exceeded
                    raise DataInspectionTools.SizeLimitExceeded(f"Download exceeded the size limit ({max_bytes/(1024*1024):.1f} MB) during transfer. Aborted.")
        finally:
            response.close()

        return content

    # --- Parsing Helpers ---

    def _process_delimited_content(self, file_content: bytes, filename_for_context: str) -> Dict[str, Any]:
        """Inspects CSV/TSV like content using Pandas and formats the results."""
        try:
            # Pass verbosity to the helper
            inspection_results = self._inspect_delimited_with_pandas(file_content, filename_for_context, self.verbosity)

            if inspection_results:
                inspection_results["status"] = "success"
                return inspection_results
            else:
                return {"status": "error", "error": "Failed to parse delimited file structure."}

        except Exception as e:
            if self.verbosity >= 1:
                print(f"    ⚠️ [Data Inspector] Error during file parsing: {type(e).__name__}")
            return {"status": "error", "error": f"Error during parsing: {e}"}

    # Static method is appropriate here as it's a pure utility function.
    @staticmethod
    def _inspect_delimited_with_pandas(file_content: bytes, filename_for_context: str, verbosity_level: int) -> Optional[Dict[str, Any]]:
        """The core logic for inspecting delimited content using Pandas."""

        # Handle potential compression based on filename context
        compression = 'gzip' if filename_for_context.lower().endswith('.gz') else 'infer'

        # Define robust parsing parameters
        parsing_params = {
            'sep': None,          # Auto-detect separator
            'engine': 'python',   # Use Python engine for better separator detection
            'nrows': 1000,        # Limit inspection to the first 1000 rows
            'compression': compression,
        }
        # Handle 'on_bad_lines' parameter compatibility across Pandas versions
        try:
            if pd.__version__ >= '1.3.0':
                parsing_params['on_bad_lines'] = 'skip'
            else:
                # Older versions use error_bad_lines/warn_bad_lines
                parsing_params['error_bad_lines'] = False
                parsing_params['warn_bad_lines'] = True
        except Exception:
             # Handle potential import issues with pd version check
             pass

        try:
            file_like_object = io.BytesIO(file_content)

            # Attempt 1: Standard read (assuming headers)
            try:
                df_preview = pd.read_csv(file_like_object, **parsing_params)

            except Exception as e:
                # Check if the exception is a ParserError
                is_parser_error = False
                # Check exception type by name (robust across environments)
                if type(e).__name__ == 'ParserError': is_parser_error = True
                # Check exception type by instance (if pd.errors is available)
                elif hasattr(pd.errors, 'ParserError'):
                     try:
                         if isinstance(e, pd.errors.ParserError): is_parser_error = True
                     except TypeError: pass # Handle potential type errors during isinstance check

                if is_parser_error:
                    # Attempt 2: Handle files without headers if parsing failed
                    file_like_object.seek(0) # Reset buffer position
                    parsing_params['header'] = None
                    df_preview = pd.read_csv(file_like_object, **parsing_params)
                    # Assign generic column names
                    df_preview.columns = [f"Column_{i+1}" for i in range(len(df_preview.columns))]
                else:
                    # If it's another error, re-raise it
                    raise e

        except Exception as e:
            if verbosity_level >= 1:
                print(f"    ⚠️ [Data Inspector] Pandas failed to parse the file preview. Error: {e}")
            return None

        if df_preview.empty and len(df_preview.columns) == 0:
            return None

        # (Result generation logic)
        column_count = len(df_preview.columns)
        headers_sample = df_preview.columns.tolist()[:10] # Limit sample size

        # Heuristic for data type estimation
        if column_count > 2: data_type = "Multivariate"
        elif column_count == 1: data_type = "Univariate"
        else: data_type = "Univariate (Potentially)" # Often time index + value

        results = {
            "file_type": "CSV/Delimited",
            "column_count": column_count,
            "headers_sample": headers_sample,
            "data_type_estimate": data_type,
            "preview_rows_parsed": len(df_preview)
        }
        return results

print("✅ deepcollector/tools/ddi.py written.")