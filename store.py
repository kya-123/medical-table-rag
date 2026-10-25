
import os
import sqlite3
import jieba
import numpy as np
from contextlib import closing


os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

from config import (
    PROFILE_DB, PROFILE_FIELDS, RERANK_MODEL_PATH, EMBEDDING_MODEL_PATH,
    CHROMA_DB_DIR, DEVICE,
)
from parsing import chinese_rag_slice_advanced, text_hash_of, extract_text

client = chromadb.PersistentClient(
    path=CHROMA_DB_DIR,
    settings=Settings(anonymized_telemetry=False),
)
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL_PATH,
    device=DEVICE
)
collection = client.get_or_create_collection(
    name="knowledge_base",
    embedding_function=ef
)

def add_text_to_knowledge_base(text: str, source_name: str) -> int:
    chunks = chinese_rag_slice_advanced(text, chunk_size=400, overlap_ratio=0.15, min_slice_len=20)
    if not chunks:
        return 0

    stem = os.path.splitext(source_name)[0]
    digest = text_hash_of(text)
    ids = [f"{stem}_{digest}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {'source': source_name, 'chunk_index': i, 'content_hash': digest}
        for i in range(len(chunks))
    ]


    UPSERT_BATCH = 2000
    for start in range(0, len(chunks), UPSERT_BATCH):
        end = start + UPSERT_BATCH
        collection.upsert(
            documents=chunks[start:end],
            ids=ids[start:end],
            metadatas=metadatas[start:end],
        )



    try:
        got = collection.get(where={"source": source_name}, include=[])
        fresh = set(ids)
        stale = [x for x in (got.get("ids") or []) if x not in fresh]
        if stale:
            collection.delete(ids=stale)
            print("【入库】%s 的旧版本片段已清理：%d 个" % (source_name, len(stale)))
    except Exception as e:
        print("【警告】清理旧片段失败（不影响新内容，但可能残留旧版）：%s: %s"
              % (type(e).__name__, e))

    return len(chunks)

def build_knowledge_base(file_path: str) -> int:
    return add_text_to_knowledge_base(extract_text(file_path), os.path.basename(file_path))

_TYPE_SQL = {"int": "INTEGER", "float": "REAL", "str": "TEXT"}

def _init_profile_db():
    cols = ",\n                ".join(
        "%s %s" % (key, _TYPE_SQL.get(t, "TEXT")) for key, _label, t in PROFILE_FIELDS)
    with closing(sqlite3.connect(PROFILE_DB)) as conn:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS profiles (
                    user_id        TEXT PRIMARY KEY,
                    %s,
                    updated_at     TEXT
                )
            """ % cols)

            have = {r[1] for r in conn.execute("PRAGMA table_info(profiles)")}
            for key, _label, t in PROFILE_FIELDS:
                if key not in have:
                    conn.execute("ALTER TABLE profiles ADD COLUMN %s %s"
                                 % (key, _TYPE_SQL.get(t, "TEXT")))

def _load_profile(user_id):
    if not user_id:
        return {}
    try:
        with closing(sqlite3.connect(PROFILE_DB)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM profiles WHERE user_id = ?",
                               (str(user_id),)).fetchone()
        if not row:
            return {}
        return {k: row[k] for k in row.keys() if k not in ("user_id", "updated_at")}
    except Exception as e:

        print("【提示】读取健康档案失败（%s），本次按无档案处理。" % e)
        return {}

_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            if os.path.isdir(RERANK_MODEL_PATH):
                print("正在加载重排模型（bge-reranker-base）...")
                _reranker = CrossEncoder(RERANK_MODEL_PATH, max_length=512, device=DEVICE)
                print("重排模型加载完成：%s" % RERANK_MODEL_PATH)
            else:
                print("【提示】没找到重排模型（%s），检索层将 fail-closed 拒答。" % RERANK_MODEL_PATH)
                _reranker = False
        except Exception as e:
            print("【提示】重排模型加载失败（%s），检索层将 fail-closed 拒答。" % e)
            _reranker = False
    return _reranker or None

_bm25_index = None
_bm25_docs = []
_bm25_ids = []
_bm25_metas = []

def _build_bm25_index():
    global _bm25_index, _bm25_docs, _bm25_ids, _bm25_metas
    
    print("正在构建 BM25 索引...")

    all_data = collection.get(include=["documents", "metadatas"])
    _bm25_docs = all_data.get("documents", [])
    _bm25_ids = all_data.get("ids", [])
    _bm25_metas = all_data.get("metadatas", [])
    
    if not _bm25_docs:
        print("【警告】向量库为空，无法构建 BM25 索引")
        return
    

    tokenized_docs = [list(jieba.cut(doc)) for doc in _bm25_docs]
    _bm25_index = BM25Okapi(tokenized_docs)
    print(f"BM25 索引构建完成，共 {len(_bm25_docs)} 个文档")

def _bm25_search(question: str, top_k: int = 20):
    if _bm25_index is None:
        return []
    

    tokenized_query = list(jieba.cut(question))
    scores = _bm25_index.get_scores(tokenized_query)
    

    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append((_bm25_docs[idx], scores[idx], _bm25_ids[idx]))
    
    return results







_init_profile_db()
