import os
import requests
import hashlib
import json
from pathlib import Path
from api_headers import api_headers

BASE_DIR = Path(__file__).resolve().parent
UPLOADED_STATE_FILE = BASE_DIR / "uploaded_state.json"

def get_file_md5(file_path: str) -> str:
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def load_uploaded_state() -> dict:
    if os.path.exists(UPLOADED_STATE_FILE):
        with open(UPLOADED_STATE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

doc_folder = BASE_DIR / "my_knowledge"
state = load_uploaded_state()

for filename in os.listdir(doc_folder):
    if not (filename.endswith(".txt") or filename.endswith(".docx") or filename.endswith(".pdf")):
        continue
    
    file_path = os.path.join(doc_folder, filename)
    md5 = get_file_md5(file_path)
    

    if filename in state and state[filename] == md5:
        print(f"⏭️ {filename}: 已上传且内容未变，跳过")
        continue
    

    with open(file_path, "rb") as f:
        files = {"file": (filename, f, "application/octet-stream")}
        try:
            resp = requests.post(
                os.environ.get("RAG_BASE_URL", "http://localhost:8001") + "/upload",
                files=files,
                headers=api_headers(),
            )
            print(f"{filename}: {resp.json()}")
        except Exception as e:
            print(f"❌ {filename}: 上传失败 - {e}")
