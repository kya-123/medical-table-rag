import re, os, requests
from sentence_transformers import CrossEncoder

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
MODEL = os.path.join(_ROOT, "model", "models", "BAAI--bge-reranker-base", "snapshots", "master")
ce = CrossEncoder(MODEL)

GROUPS = {
    "库内-正常": [
        "糖尿病患者能吃西瓜吗",
        "糖尿病患者能吃苹果吗",
        "糖尿病患者的血糖控制目标是多少",
        "我国9次全国性糖尿病流行病学调查分别在哪些年份",
        "二甲双胍应该怎么服用",
        "糖尿病患者的运动建议是什么",
        "低血糖该怎么应急处理",
        "妊娠期糖尿病的诊断标准是什么",
    ],
    "相邻科室": [
        "高血压应该吃什么药",
        "贫血吃什么好",
    ],
    "完全无关": [
        "红烧肉怎么做",
        "量子纠缠的原理是什么",
        "今天天气怎么样",
    ],
}


def split_chunks(ctx):
    parts = re.split(r"【资料\d+·[^】]*】\n", ctx)
    return [p.strip() for p in parts if p.strip()]


print(f"{'问题':28s} {'片段数':>4s} {'top1':>6s} {'≥.50':>4s} {'≥.45':>4s} {'≥.40':>4s} {'≥.30':>4s}")
print("-" * 78)
rows = []
for group, qs in GROUPS.items():
    for q in qs:
        r = requests.post("http://127.0.0.1:8001/ask", json={"question": q, "top_k": 5}, timeout=240).json()
        chunks = split_chunks(r.get("context", ""))
        if not chunks:
            print(f"{q[:26]:28s} — 无资料")
            continue
        sc = sorted([float(x) for x in ce.predict([(q, c) for c in chunks])], reverse=True)
        row = (group, q, len(sc), sc[0],
               sum(1 for s in sc if s >= 0.50),
               sum(1 for s in sc if s >= 0.45),
               sum(1 for s in sc if s >= 0.40),
               sum(1 for s in sc if s >= 0.30))
        rows.append(row)
        print(f"{q[:26]:28s} {row[2]:4d} {row[3]:6.3f} {row[4]:4d} {row[5]:4d} {row[6]:4d} {row[7]:4d}")

print("\n" + "=" * 78)
print("按组统计 top1 范围")
for g in GROUPS:
    vals = [r[3] for r in rows if r[0] == g]
    if vals:
        print(f"  {g:8s} top1: min={min(vals):.3f} max={max(vals):.3f}  n={len(vals)}")
print("按组统计 片段中 ≥0.45 的条数")
for g in GROUPS:
    vals = [r[5] for r in rows if r[0] == g]
    if vals:
        print(f"  {g:8s} ≥0.45 条数: min={min(vals)} max={max(vals)}")
