import os
import uvicorn

from api import app

if __name__ == "__main__":


    HOST = os.environ.get("RAG_HOST", "127.0.0.1")
    PORT = int(os.environ.get("RAG_PORT", "8001"))
    token_on = bool(os.environ.get("RAG_API_TOKEN", "").strip())
    print("=" * 64)
    print(f"  接口地址：  http://127.0.0.1:{PORT}/ask")
    print(f"  接口文档：  http://127.0.0.1:{PORT}/docs   （浏览器直接打开，能点着测）")
    if HOST in ("0.0.0.0", "::"):
        print(f"  局域网访问：http://<本机IP>:{PORT}/ask      （手机 / 小程序用这个）")
        print(f"  鉴权      ：{'已开启（Bearer token）' if token_on else '⚠ 未开启 —— 同局域网任何人可调用'}")
    else:
        print(f"  监听地址：  {HOST}（仅本机；要给小程序连，设 RAG_HOST=0.0.0.0）")
    print("=" * 64)
    uvicorn.run(app, host=HOST, port=PORT)
