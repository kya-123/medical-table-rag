import sqlite3, time, torch, os
from sentence_transformers import SentenceTransformer, CrossEncoder

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
EMB = os.path.join(_ROOT, "model", "models", "Xorbits--bge-large-zh-v1.5", "snapshots", "master")
RRK = os.path.join(_ROOT, "model", "models", "BAAI--bge-reranker-base", "snapshots", "master")

con = sqlite3.connect('file:' + os.path.join(_ROOT, 'chroma_db', 'chroma.sqlite3') + '?mode=ro', uri=True)
rows = [r[0] for r in con.execute(
    'SELECT string_value FROM embedding_metadata WHERE key="chroma:document" LIMIT 300')]
con.close()
print(f"取到真实片段 {len(rows)} 条，平均长度 {sum(len(x) for x in rows)/len(rows):.0f} 字")

TOTAL = 4573

for dev in ["cpu", "cuda"]:
    if dev == "cuda" and not torch.cuda.is_available():
        print("cuda 不可用，跳过")
        continue
    m = SentenceTransformer(EMB, device=dev)
    m.encode(rows[:8], batch_size=8, show_progress_bar=False)
    t0 = time.time()
    m.encode(rows[:200], batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    dt = time.time() - t0
    print(f"[{dev:4s}] 向量化 200 段 / {dt:5.1f}s = {200/dt:5.1f} 段/秒 "
          f"→ 全量 {TOTAL} 段约 {TOTAL/(200/dt)/60:5.1f} 分钟")
    del m
    if dev == "cuda":
        torch.cuda.empty_cache()

ce_cpu = CrossEncoder(RRK, max_length=512, device="cpu")
pairs = [("糖尿病患者空腹血糖控制目标是多少", t) for t in rows[:20]]
ce_cpu.predict(pairs, batch_size=20)
t0 = time.time(); ce_cpu.predict(pairs, batch_size=20); print(f"[cpu ] 重排 20 对 / {time.time()-t0:4.1f}s")
del ce_cpu

if torch.cuda.is_available():
    ce_gpu = CrossEncoder(RRK, max_length=512, device="cuda")
    ce_gpu.predict(pairs, batch_size=20)
    t0 = time.time(); ce_gpu.predict(pairs, batch_size=20); print(f"[cuda] 重排 20 对 / {time.time()-t0:4.1f}s")
