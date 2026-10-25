
import os


def api_headers(admin: bool = False) -> dict:
    token = ""
    if admin:
        token = os.environ.get("RAG_ADMIN_TOKEN", "").strip()
    if not token:
        token = os.environ.get("RAG_API_TOKEN", "").strip()
    if not token:
        return {}
    return {"Authorization": "Bearer " + token}


def admin_headers() -> dict:
    """返回知识库管理接口使用的高权限请求头。"""
    return api_headers(admin=True)
