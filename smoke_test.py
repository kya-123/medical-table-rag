import io, os, time
import requests
from api_headers import api_headers

B = os.environ.get("RAG_BASE_URL", "http://127.0.0.1:8001").rstrip("/")
HEADERS = api_headers()
SMOKE_QUESTION = os.environ.get(
    "RAG_SMOKE_QUESTION",
    "MedRAG 的切片长度是多少，为什么是这个数",
)
ok = fail = 0

def check(name, cond, extra=""):
    global ok, fail
    if cond:
        ok += 1; print(f"  ✔ {name} {extra}")
    else:
        fail += 1; print(f"  ✘ {name} {extra}")

print("=== 1. 健康检查 ===")
r = requests.get(f"{B}/health", timeout=30, headers=HEADERS)
check("GET /health", r.status_code == 200 and r.json().get("code") == 200)

print("=== 2. 问答 ===")
r = requests.post(
    f"{B}/ask",
    json={"question": SMOKE_QUESTION, "top_k": 5},
    timeout=240,
    headers=HEADERS,
)
d = r.json()
check("POST /ask", r.status_code == 200 and len(d.get("answer", "")) > 20, f"({len(d.get('answer',''))} 字)")
check("/ask 带引用", "【资料" in d.get("answer", ""))
check("/ask 返回 sources", len(d.get("sources", [])) > 0, f"({len(d.get('sources', []))} 条)")

print("=== 3. 知识库清单 ===")
r = requests.get(f"{B}/sources", timeout=60, headers=HEADERS)
d = r.json()
check("GET /sources", r.status_code == 200 and d.get("total_chunks", 0) > 0,
      f"({d.get('total_sources')} 文件 / {d.get('total_chunks')} 片段)")

print("=== 4. 用户档案 ===")
r = requests.post(
    f"{B}/profile",
    json={"user_id": "_selftest", "age": 58, "diabetes_type": "2型"},
    timeout=30,
    headers=HEADERS,
)
check("POST /profile", r.status_code == 200 and "档案已保存" in r.json().get("message", ""))
r = requests.get(f"{B}/profile/_selftest", timeout=30, headers=HEADERS)
check("GET /profile", r.status_code == 200 and r.json().get("data", {}).get("age") == 58)
r = requests.get(f"{B}/profiles", timeout=30, headers=HEADERS)
allow_profile_list = os.environ.get("RAG_ALLOW_PROFILE_LIST", "").strip().lower() in ("1", "true", "yes")
expected_profile_status = 200 if allow_profile_list else 403
check("GET /profiles", r.status_code == expected_profile_status,
      f"(status={r.status_code}, expected={expected_profile_status})")

print("=== 5. 上传（建临时 txt）===")
tmp = "_selftest_upload.txt"
open(tmp, "w", encoding="utf-8").write("糖尿病自我测试文档。" * 40)
with open(tmp, "rb") as f:
    r = requests.post(
        f"{B}/upload",
        files={"file": (tmp, f, "text/plain")},
        timeout=240,
        headers=HEADERS,
    )
msg = r.json().get("message", "")
check("POST /upload", r.status_code == 200 and "上传成功" in msg, f"({msg[:30]})")
os.remove(tmp)

print("=== 6. 删除刚才上传的文件 ===")
r = requests.post(
    f"{B}/delete",
    json={"source": "_selftest_upload.txt"},
    timeout=60,
    headers=HEADERS,
)
check("POST /delete 预览", r.status_code == 200 and r.json().get("data", {}).get("deleted") is False)
r = requests.post(
    f"{B}/delete",
    json={"source": "_selftest_upload.txt", "confirm": True},
    timeout=120,
    headers=HEADERS,
)
check("POST /delete 执行", r.status_code == 200 and r.json().get("data", {}).get("deleted") is True)

print("=== 7. 拒答（知识库外）===")
r = requests.post(
    f"{B}/ask",
    json={"question": "今天天气怎么样", "top_k": 5},
    timeout=240,
    headers=HEADERS,
)
check("无关问题被拒答", r.json().get("answer", "").startswith("未找到"))

r = requests.delete(f"{B}/profile/_selftest", timeout=30, headers=HEADERS)
check("DELETE /profile 清理", r.status_code == 200 and r.json().get("data", {}).get("deleted") is True)

print(f"\n合计：通过 {ok}，失败 {fail}")
