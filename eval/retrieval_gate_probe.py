
import json
import hashlib
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from config import CHROMA_DB_DIR, DEVICE
from retrieval import _retrieve_with_assessment
from store import _build_bm25_index, collection


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(result.stdout.strip())
    except Exception:
        return True


def _context_metrics(item: dict, context: str) -> tuple:
    keys = item.get("key_info", [])
    chunks = item.get("must_have_chunks", [])
    found_keys = [key for key in keys if key in context]
    found_chunks = [chunk for chunk in chunks if chunk in context]
    recall = len(found_keys) / len(keys) if keys else 0.0
    hit = len(found_chunks) / len(chunks) if chunks else 0.0
    return recall, hit


def run() -> int:
    if collection.count() == 0:
        raise RuntimeError(
            "索引为空，未生成探针结果。请先设置 RAG_CHROMA_DIR 指向已构建的索引目录。"
        )
    gold = json.loads((_HERE / "gold_standard.json").read_text(encoding="utf-8"))
    _build_bm25_index()
    index_metas = collection.get(include=["metadatas"]).get("metadatas") or []
    index_sources = len({
        (metadata or {}).get("source")
        for metadata in index_metas
        if (metadata or {}).get("source")
    })
    index_chunks = collection.count()
    source_names = sorted({
        (metadata or {}).get("source")
        for metadata in index_metas
        if (metadata or {}).get("source")
    })
    index_fingerprint = hashlib.sha256(
        ("|".join(source_names) + ":" + str(index_chunks)).encode("utf-8")
    ).hexdigest()[:12]
    details = []
    for item in gold:
        docs, assessment = _retrieve_with_assessment(
            item["question"],
            top_k=5,
            include_trace=True,
        )
        trace = assessment.get("trace") or {}
        pre_docs = trace.get("after_retrieval", [])
        table_docs = trace.get("after_table_expansion", pre_docs)
        pre_context = "\n".join(doc.get("text", "") for doc in pre_docs)
        table_context = "\n".join(doc.get("text", "") for doc in table_docs)
        post_context = "\n".join(doc for doc, _source in docs)
        pre_recall, pre_hit = _context_metrics(item, pre_context)
        table_recall, table_hit = _context_metrics(item, table_context)
        post_recall, post_hit = _context_metrics(item, post_context)
        details.append({
            "question": item["question"],
            "decision": assessment["decision"],
            "retrieval_recall_before_gate": pre_recall,
            "retrieval_hit_before_gate": pre_hit,
            "recall_after_table_expansion": table_recall,
            "hit_after_table_expansion": table_hit,
            "context_recall_after_gate": post_recall,
            "context_hit_after_gate": post_hit,
            "after_retrieval_chunks": len(pre_docs),
            "after_table_expansion_chunks": len(table_docs),
            "after_gate_chunks": len(docs),
        })
        print(details[-1], flush=True)

    total = len(details)
    summary = {
        "total": total,
        "retrieval_recall_before_gate": sum(
            item["retrieval_recall_before_gate"] for item in details
        ) / total,
        "retrieval_hit_before_gate": sum(
            item["retrieval_hit_before_gate"] for item in details
        ) / total,
        "recall_after_table_expansion": sum(
            item["recall_after_table_expansion"] for item in details
        ) / total,
        "hit_after_table_expansion": sum(
            item["hit_after_table_expansion"] for item in details
        ) / total,
        "context_recall_after_gate": sum(
            item["context_recall_after_gate"] for item in details
        ) / total,
        "context_hit_after_gate": sum(
            item["context_hit_after_gate"] for item in details
        ) / total,
        "answer": sum(item["decision"] == "answer" for item in details),
        "partial": sum(item["decision"] == "partial" for item in details),
        "refuse": sum(item["decision"] == "refuse" for item in details),
    }
    output = {
        "run_info": {
            "status": "current_retrieval_only",
            "date": datetime.now().astimezone().isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
            "git_dirty": _git_dirty(),
            "top_k": 5,
            "device": DEVICE,
            "index_dir_mode": "env_override" if os.environ.get("RAG_CHROMA_DIR") else "default",
            "index_dir_name": os.path.basename(os.path.normpath(CHROMA_DB_DIR)),
            "index_fingerprint": index_fingerprint,
            "index_sources": index_sources,
            "index_chunks": index_chunks,
            "note": "依次记录重排后、表格补全后和 gate 后的上下文；gate 前指标不混入最终拒答结果，不调用 DeepSeek。",
        },
        "summary": summary,
        "details": details,
    }
    (_HERE / "retrieval_gate_probe.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("SUMMARY", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
