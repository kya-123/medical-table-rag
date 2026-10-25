import os
import uvicorn

from api import app
from security import (
    is_local_client,
    validate_network_security,
)

if __name__ == "__main__":


    HOST = os.environ.get("RAG_HOST", "127.0.0.1")
    PORT = int(os.environ.get("RAG_PORT", "8001"))
    api_token = os.environ.get("RAG_API_TOKEN", "").strip()
    admin_token = os.environ.get("RAG_ADMIN_TOKEN", "").strip()
    validate_network_security(HOST, api_token, admin_token)
    external = not is_local_client(HOST)
    print("=" * 64)
    print(f"  接口地址：  http://127.0.0.1:{PORT}/ask")
    print(f"  接口文档：  http://127.0.0.1:{PORT}/docs   （浏览器直接打开，能点着测）")
    if external:
        print(f"  局域网访问：http://<本机IP>:{PORT}/ask      （手机 / 小程序用这个）")
        print("  鉴权      ：已开启（普通 Token 与管理员 Token 分离）")
    else:
        print(f"  监听地址：  {HOST}（仅本机；要给小程序连，设 RAG_HOST=0.0.0.0）")
    print("=" * 64)
    uvicorn.run(app, host=HOST, port=PORT)
