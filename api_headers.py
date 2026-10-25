
import os


def api_headers() -> dict:
    token = os.environ.get("RAG_API_TOKEN", "").strip()
    if not token:
        return {}
    return {"Authorization": "Bearer " + token}
