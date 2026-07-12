import os
import io

print("☁️ 1. Connecting to Google Drive via local DGX token...")
try:
    from googleapiclient.discovery import build
    from google.oauth2.credentials import Credentials
    from googleapiclient.http import MediaIoBaseDownload
    
    TOKEN_FILE = os.path.expanduser("~/Desktop/DeepKG/token.json")
    DEST_DIR = os.path.expanduser("~/Desktop/DeepKG/PDFGems")

    os.makedirs(DEST_DIR, exist_ok=True)

    creds = Credentials.from_authorized_user_file(TOKEN_FILE)
    service = build('drive', 'v3', credentials=creds)

    print("🔍 Searching for 'PDFGems' folder...")
    results = service.files().list(
        q="mimeType='application/vnd.google-apps.folder' and name='PDFGems'", 
        spaces='drive', 
        fields='files(id, name)'
    ).execute()

    folders = results.get('files', [])

    if not folders:
        print("❌ Could not find 'PDFGems' in your Google Drive.")
        print("Please copy the PDFs manually to ~/Desktop/DeepKG/PDFGems/")
    else:
        folder_id = folders[0]['id']
        print(f"✅ Found 'PDFGems' (ID: {folder_id}). Fetching file list...")

        results = service.files().list(
            q=f"'{folder_id}' in parents", 
            spaces='drive', 
            fields='files(id, name)'
        ).execute()

        files = results.get('files', [])
        if not files:
            print("⚠️ Folder is empty.")
        else:
            print(f"📥 Found {len(files)} files. Starting download to {DEST_DIR}...")
            for f in files:
                file_path = os.path.join(DEST_DIR, f['name'])
                if os.path.exists(file_path):
                    print(f"   ⏭️ Skipping {f['name']} (already exists)")
                    continue
                print(f"   -> Downloading {f['name']}...")
                request = service.files().get_media(fileId=f['id'])
                fh = io.FileIO(file_path, 'wb')
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
            print("🎉 All PDFs downloaded successfully!")

except Exception as e:
    print(f"❌ Drive API Error: {e}")
    print("Please copy the PDFs manually using SCP/MobaXterm to ~/Desktop/DeepKG/PDFGems/")

print("\n🔧 2. Patching research.py to route Colab paths to the local DGX folder...")
research_paths = [
    os.path.expanduser("~/Desktop/DeepKG/deepcollector/tools/research.py"),
    os.path.expanduser("~/Desktop/DeepKG/deepcollector/localdgxfiles/research.py")
]

old_code = '''        if url.startswith('/content/drive/'):
            try:
                if self.verbosity >= 2: print(f"    📁 [Local Fetch] Reading {url} directly from Drive...")
                with open(url, 'rb') as f:
                    content = f.read()'''

new_code = '''        if url.startswith('/content/drive/'):
            try:
                import os
                # Map Colab path to local DGX path
                if "PDFGems" in url:
                    filename = os.path.basename(url)
                    url = os.path.expanduser(f"~/Desktop/DeepKG/PDFGems/{filename}")
                    
                if self.verbosity >= 2: print(f"    📁 [Local Fetch] Reading {url} directly from local disk...")
                with open(url, 'rb') as f:
                    content = f.read()'''

for path in research_paths:
    if os.path.exists(path):
        with open(path, "r") as f:
            code = f.read()
            
        if old_code in code:
            code = code.replace(old_code, new_code)
            with open(path, "w") as f:
                f.write(code)
            print(f"  ✅ Successfully patched {path}")
        elif "PDFGems" in code:
            print(f"  ✅ {path} already patched.")

print("\n🚀 Setup complete! You can now restart your run.")
