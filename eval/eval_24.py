import json, os, re, time
import sys
import requests
from docx import Document
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api_headers import api_headers


_here = os.path.dirname(os.path.abspath(__file__))
DOCX = os.path.join(_here, "测试问题.docx")
if not os.path.exists(DOCX):
    DOCX = os.path.join(_here, "..", "测试问题.docx")
URL = os.environ.get("RAG_BASE_URL", "http://127.0.0.1:8001").rstrip("/") + "/ask"

doc = Document(DOCX)
paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]


questions, group = [], ""
for p in paras:
    if re.match(r"^\d+[.．、]", p):
        q = re.sub(r"^\d+[.．、]\s*", "", p).strip()
        questions.append({"group": group, "question": q})
    elif p != "测试问题":
        group = p

print(f"从 docx 解析出 {len(questions)} 道题\n")


gold = json.load(open(os.path.join(_here, "gold_standard.json"), encoding="utf-8"))
gold_qs = {(g["question"] if isinstance(g, dict) else g) for g in gold}

def norm(s):
    return re.sub(r"[\s，。？?！!、；;,.]", "", s)

gold_norm = {norm(q) for q in gold_qs}

rows = []
for i, item in enumerate(questions, 1):
    q = item["question"]
    in_gold = norm(q) in gold_norm
    t0 = time.time()
    r = requests.post(
        URL,
        json={"question": q, "top_k": 5},
        timeout=240,
        headers=api_headers(),
    ).json()
    dt = time.time() - t0
    ans = (r.get("answer") or "").strip()
    refused = bool(r.get("refused", ans.startswith("未找到")))
    rows.append({
        "no": i, "group": item["group"], "question": q, "in_gold": in_gold,
        "seconds": round(dt, 1), "sources": len(r.get("sources") or []),
        "refused": refused, "answer_len": len(ans), "answer_head": ans[:80],
        "rewritten": r.get("rewritten_question", ""),
    })
    print(f"{i:2d}. [{'gold' if in_gold else '    '}] {dt:4.1f}s 起 {len(r.get('sources') or []):2d} "
          f"{'拒答' if refused else '作答'}  {q[:34]}")

json.dump(rows, open(os.path.join(_here, "eval_24.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

ok = [r for r in rows if not r["refused"]]
print(f"\n合计 {len(rows)} 题：作答 {len(ok)}，拒答 {len(rows)-len(ok)}")
print(f"平均耗时 {sum(r['seconds'] for r in rows)/len(rows):.1f}s")
print(f"其中带标准答案的 {sum(1 for r in rows if r['in_gold'])} 题")
print("\n被拒答的题：")
for r in rows:
    if r["refused"]:
        print(f"  {r['no']}. [{r['group']}] {r['question'][:50]}")
