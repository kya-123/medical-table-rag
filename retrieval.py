
import re
import os

import store
from store import collection, get_reranker, _bm25_search

from config import (
    RERANK_POOL, HYBRID_ON,
    KEYWORD_MAX_TERMS, KEYWORD_MAX_HITS, KEYWORD_MAX_CHUNKS,
    KEYWORD_RARE_MAX, KEYWORD_KEEP, _JUNK_CHARS, DEDUP_ADJACENT,
)
from evidence_sufficiency import assess_evidence

def _contains_count(term: str) -> int:
    try:
        got = collection.get(where_document={"$contains": term}, include=[],
                             limit=KEYWORD_MAX_HITS + 1)
        return len(got.get("ids") or [])
    except Exception:
        return -1

def _keyword_terms(question: str):
    q = re.sub(r"[^\w\u4e00-\u9fff]", "", question or "")
    if len(q) < 2:
        return []

    _QUOTA = {4: 10, 3: 8, 2: min(16, max(6, len(q) // 4))}

    def _sample(layer, k):
        if k <= 0 or not layer:
            return []
        if len(layer) <= k:
            return list(layer)
        step = len(layer) / float(k)
        return [layer[int(i * step)] for i in range(k)]

    cands = []
    for n in (4, 3, 2):
        layer = []
        for i in range(len(q) - n + 1):
            t = q[i:i + n]

            if t in layer or sum(1 for c in t if c in _JUNK_CHARS) * 2 >= len(t):
                continue
            layer.append(t)
        k = _QUOTA[n]
        head_k = (k + 1) // 2
        tail_k = k - head_k
        head = _sample(layer, head_k)
        tail = list(reversed(_sample(list(reversed(layer)), tail_k)))
        for t in head + tail:
            if t not in cands:
                cands.append(t)
    scored = []
    for t in cands:
        c = _contains_count(t)
        if 1 <= c <= KEYWORD_MAX_HITS:
            scored.append((c, -len(t), t))
    if not scored:
        return []


    if min(c for c, _l, _t in scored) > KEYWORD_RARE_MAX:
        return []
    scored.sort()


    used = [False] * len(q)
    out = []
    for _c, _l, t in scored:
        pos = q.find(t)
        if pos < 0 or any(used[pos:pos + len(t)]):
            continue
        out.append(t)
        for i in range(pos, min(pos + len(t), len(q))):
            used[i] = True
        if len(out) >= KEYWORD_MAX_TERMS:
            break
    return out

def _keyword_search(question: str):
    if not HYBRID_ON:
        return []
    terms = _keyword_terms(question)
    if not terms:
        return []

    if len(terms) == 1:
        wd = {"$contains": terms[0]}
    else:
        wd = {"$or": [{"$contains": t} for t in terms]}
    try:
        got = collection.get(
            where_document=wd,
            include=["documents", "metadatas"], limit=KEYWORD_MAX_CHUNKS)
    except Exception as e:
        print("【提示】关键词检索失败（%s），本次只用向量检索。" % type(e).__name__)
        return []
    pairs = list(zip(got.get("metadatas") or [], got.get("documents") or []))

    pairs.sort(key=lambda x: -sum(1 for t in terms if t in x[1]))
    print("【混合检索】关键词 %d 个 → 命中 %d 条" % (len(terms), len(pairs)))

    return [(d, (m or {}).get("source", "")) for m, d in pairs]

def _rrf_fusion(vector_results, bm25_results, k=60):
    fused_scores = {}
    

    for rank, (doc, score, doc_id) in enumerate(vector_results):
        if doc_id not in fused_scores:
            fused_scores[doc_id] = {'doc': doc, 'score': 0.0, 'id': doc_id}
        fused_scores[doc_id]['score'] += 1.0 / (k + rank + 1)
    

    for rank, (doc, score, doc_id) in enumerate(bm25_results):
        if doc_id not in fused_scores:
            fused_scores[doc_id] = {'doc': doc, 'score': 0.0, 'id': doc_id}
        fused_scores[doc_id]['score'] += 1.0 / (k + rank + 1)
    

    sorted_results = sorted(fused_scores.values(), key=lambda x: x['score'], reverse=True)
    return [(r['doc'], r['score'], r['id']) for r in sorted_results]

def _retrieve_with_assessment(
    question: str,
    top_k: int,
    pool_mult: int = 5,
    include_trace: bool = False,
):
    reranker = get_reranker()



    if reranker is None:
        assessment = assess_evidence(question, [])
        assessment.reason = "重排模型不可用，无法执行证据充分性判定"
        print("【安全边界】重排模型不可用，本次拒绝回答")
        return [], assessment.to_dict()

    pool = max(int(top_k) * int(pool_mult), RERANK_POOL)
    keep = pool


    res = collection.query(query_texts=[question], n_results=pool, include=["documents", "metadatas"])
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    ids = (res.get("ids") or [[]])[0]

    if not docs:
        return [], assess_evidence(question, []).to_dict()


    vector_results = [(doc, 1.0, doc_id) for doc, doc_id in zip(docs, ids)]


    bm25_results = _bm25_search(question, top_k=pool)
    

    if bm25_results:
        fused_results = _rrf_fusion(vector_results, bm25_results, k=60)
        print(f"【RRF融合】向量检索 {len(vector_results)} 条 + BM25 {len(bm25_results)} 条 → 融合后 {len(fused_results)} 条")
    else:

        fused_results = vector_results


    picked, taken, seen = [], [], set()
    doc_source = {}
    for doc, score, doc_id in fused_results:

        try:
            idx_in_results = ids.index(doc_id)
            meta = metas[idx_in_results] or {}
        except (ValueError, IndexError):

            meta = {}
            try:
                b_idx = store._bm25_ids.index(doc_id)
                meta = (store._bm25_metas[b_idx] if b_idx < len(store._bm25_metas) else None) or {}
            except (ValueError, IndexError):
                meta = {}
        
        src = meta.get("source", "")
        idx = meta.get("chunk_index", -1)

        if doc in seen:
            continue
        if DEDUP_ADJACENT > 0 and any(
               src == s and isinstance(idx, int) and isinstance(i, int)
               and abs(idx - i) <= DEDUP_ADJACENT
               for s, i in taken):
            continue

        picked.append(doc)
        doc_source[doc] = src
        taken.append((src, idx))
        seen.add(doc)
        if len(picked) >= keep:
            break


    kw_pairs = _keyword_search(question)
    kw_top = [d for d, _s in kw_pairs[:KEYWORD_KEEP]]
    if kw_pairs:
        have = set(picked)
        for d, s in kw_pairs:
            if d not in have:
                have.add(d)
                picked.append(d)
                if s:
                    doc_source[d] = s




    for d in kw_top:
        if d not in picked:
            picked.append(d)


    try:
        scores = reranker.predict([(question, d) for d in picked], batch_size=16)
        scored_picked = list(zip(picked, scores))
        scored_picked.sort(key=lambda x: -float(x[1]))
    except Exception as e:
        print("【警告】重排失败，无法执行安全判定：%s" % type(e).__name__)
        assessment = assess_evidence(question, [])
        assessment.reason = "重排模型推理失败，无法安全判断证据是否充分"
        return [], assessment.to_dict()


    selected_scored = list(scored_picked[:int(top_k)])
    selected_docs = {doc for doc, _score in selected_scored}
    score_by_doc = {doc: float(score) for doc, score in scored_picked}
    for doc in kw_top:
        if doc not in selected_docs:
            selected_scored.append((doc, score_by_doc.get(doc, 0.0)))
            selected_docs.add(doc)

    def with_trace(
        docs,
        result,
        after_retrieval_docs,
        after_table_docs,
    ):
        if include_trace:
            result = {
                **result,
                "trace": {
                    "after_retrieval": [
                        {
                            "text": doc,
                            "source": doc_source.get(doc, ""),
                        }
                        for doc in after_retrieval_docs
                    ],
                    "after_table_expansion": [
                        {
                            "text": doc,
                            "source": doc_source.get(doc, ""),
                        }
                        for doc in after_table_docs
                    ],
                    "after_gate": [
                        {
                            "text": doc,
                            "source": source,
                        }
                        for doc, source in docs
                    ],
                },
            }
        return docs, result

    assessment = assess_evidence(question, selected_scored)
    after_retrieval_docs = [doc for doc, _score in selected_scored]
    after_table_docs = list(after_retrieval_docs)
    if assessment.decision == "refuse":


        preview_docs = [doc for doc, _score in selected_scored]
        expanded_preview = _expand_tables(preview_docs, doc_source=doc_source)
        if expanded_preview != preview_docs:
            after_table_docs = expanded_preview
            top_score = float(scored_picked[0][1]) if scored_picked else 0.0
            expanded_scored = [
                (doc, score_by_doc.get(doc, top_score * 0.99))
                for doc in expanded_preview
            ]
            expanded_scored.sort(key=lambda item: -float(item[1]))
            expanded_assessment = assess_evidence(question, expanded_scored)
            if expanded_assessment.decision != "refuse":
                print("【证据判定】表格补全后重新判定为 %s" % expanded_assessment.decision)
                selected_scored = expanded_scored
                assessment = expanded_assessment

    if assessment.decision == "refuse":
        print(
            "【证据判定】拒答：%s（意图 %.3f / 分面覆盖 %.3f / 风险 %s）"
            % (
                assessment.reason,
                assessment.intent_score,
                assessment.mandatory_coverage,
                assessment.risk_level,
            )
        )
        return with_trace(
            [],
            assessment.to_dict(),
            after_retrieval_docs,
            after_table_docs,
        )

    print(
        "【证据判定】%s：%s（意图 %.3f / 分面覆盖 %.3f / 风险 %s）"
        % (
            assessment.decision,
            assessment.reason,
            assessment.intent_score,
            assessment.mandatory_coverage,
            assessment.risk_level,
        )
    )
    picked = [doc for doc, _score in selected_scored]
    after_table_docs = list(picked)


    picked = _expand_tables(picked, doc_source=doc_source)
    after_table_docs = list(picked)
    delivered = [(d, doc_source.get(d, "")) for d in picked]

    return with_trace(
        delivered,
        assessment.to_dict(),
        after_retrieval_docs,
        after_table_docs,
    )


def _retrieve_distinct(question: str, top_k: int, pool_mult: int = 5):
    docs, _assessment = _retrieve_with_assessment(question, top_k, pool_mult)
    return docs

def _expand_tables(picked, max_tables: int = 3, max_chunks: int = 60, doc_source=None):
    table_names = []
    for d in picked:

        for name in re.findall(r"【表(?:·([^】]{1,60}))?】", d):
            if name and name not in table_names:
                table_names.append(name)
    if not table_names:
        return picked

    extra = []
    for name in table_names[:max_tables]:
        got = None

        src = ""
        if doc_source:
            src = next((doc_source.get(d, "") for d in picked if name in d), "")
        scope = {"source": src} if src else None

        for key in (name, name[:12]):
            try:
                got = collection.get(where_document={"$contains": key},
                                     where=scope,
                                     include=["documents", "metadatas"])
                if got and got.get("documents"):
                    break
            except Exception:
                got = None
        if not (got and got.get("documents")) and scope is not None:


            for key in (name, name[:12]):
                try:
                    got = collection.get(where_document={"$contains": key},
                                         include=["documents", "metadatas"])
                    if got and got.get("documents"):
                        print("【表格补全】限定来源没找到，已退回全库")
                        break
                except Exception:
                    got = None
        if not got:
            continue
        pairs = list(zip(got.get("metadatas") or [], got.get("documents") or []))

        pairs.sort(key=lambda x: (x[0] or {}).get("chunk_index") or 0)
        for _m, d in pairs:
            extra.append(d)
            if doc_source is not None and _m:
                doc_source[d] = _m.get("source", "")

    if not extra:
        return picked


    seen, merged = set(), []
    for d in extra + picked:
        if d not in seen:
            seen.add(d)
            merged.append(d)
    return merged[:max_chunks]
