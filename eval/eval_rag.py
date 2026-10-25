

import json
import os
import requests
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api_headers import api_headers


RAG_URL = os.environ.get("RAG_BASE_URL", "http://127.0.0.1:8001").rstrip("/") + "/ask"
_HERE = Path(__file__).resolve().parent
GOLD_FILE = str(_HERE / "gold_standard.json")
RESULT_FILE = str(_HERE / "eval_results.json")


def collect_run_info(top_k: int = 5) -> Dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=str(_HERE.parent),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        commit = "unknown"

    try:
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(_HERE.parent),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip())
    except Exception:
        dirty = True

    index_info = {}
    try:
        source_resp = requests.get(
            RAG_URL.replace("/ask", "/sources"),
            timeout=30,
            headers=api_headers(),
        )
        source_resp.raise_for_status()
        payload = source_resp.json()
        index_info = {
            "total_sources": payload.get("total_sources"),
            "total_chunks": payload.get("total_chunks"),
        }
    except Exception as e:
        index_info = {"error": f"{type(e).__name__}: {e}"}

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": dirty,
        "top_k": top_k,
        "generation_model": "deepseek-chat",
        "evidence_gate": "facet-aware-v1",
        "index": index_info,
    }


def load_gold_standard() -> List[Dict]:
    if not Path(GOLD_FILE).exists():
        print(f"❌ 找不到标准答案文件：{GOLD_FILE}")
        print("请先填写 gold_standard.json")
        return []
    
    with open(GOLD_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def ask_rag(question: str, top_k: int = 5) -> Dict[str, Any]:
    try:
        response = requests.post(
            RAG_URL,
            json={"question": question, "top_k": top_k},
            timeout=30,
            headers=api_headers(),
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ 调用失败：{e}")
        return None


def normalize_text(text: str) -> str:

    text = text.replace("SGLT2i", "SGLT-2").replace("SGLT2", "SGLT-2")
    text = text.replace("SGLT⁃2", "SGLT-2")

    text = text.replace("T2DM", "2型糖尿病")
    return text


def evaluate_single(question_data: Dict, response: Dict) -> Dict:
    context = response.get("context", "")
    answer = response.get("answer", "")
    

    context_normalized = normalize_text(context)
    answer_normalized = normalize_text(answer)
    

    key_info = question_data.get("key_info", [])
    found_keys = [k for k in key_info if normalize_text(k) in context_normalized]
    recall = len(found_keys) / len(key_info) if key_info else 0
    

    must_have = question_data.get("must_have_chunks", [])
    found_chunks = [c for c in must_have if c in context]
    hit = len(found_chunks) / len(must_have) if must_have else 0
    

    answer_has_keys = [k for k in key_info if normalize_text(k) in answer_normalized]
    answer_recall = len(answer_has_keys) / len(key_info) if key_info else 0
    
    return {
        "question": question_data["question"],
        "recall": recall,
        "found_keys": found_keys,
        "missing_keys": [k for k in key_info if k not in found_keys],
        "hit": hit,
        "found_chunks": found_chunks,
        "missing_chunks": [c for c in must_have if c not in found_chunks],
        "answer_length": len(answer),
        "answer_recall": answer_recall,
        "context_length": len(context),
        "context_preview": context[:200] + "..." if len(context) > 200 else context
    }


def run_evaluation():
    print("=" * 70)
    print("RAG 系统评测")
    print("=" * 70)
    

    gold_data = load_gold_standard()
    if not gold_data:
        return
    
    print(f"✅ 加载了 {len(gold_data)} 道题\n")
    

    results = []
    failed_questions = []
    for i, q in enumerate(gold_data, 1):
        print(f"[{i}/{len(gold_data)}] {q['question'][:50]}...")
        

        response = ask_rag(q["question"])
        if not response:
            failed_questions.append(q["question"])
            continue
        

        result = evaluate_single(q, response)
        results.append(result)
        

        print(f"   Recall: {result['recall']:.1%}  Hit: {result['hit']:.1%}  "
              f"Answer: {result['answer_length']}字")
        
        if result['missing_keys']:
            print(f"   ❌ 缺失关键词：{', '.join(result['missing_keys'][:3])}")
        
        if result['missing_chunks']:
            print(f"   ❌ 缺失表格：{', '.join(result['missing_chunks'][:2])}")
        
        print()
        

        time.sleep(0.5)

    if len(results) != len(gold_data):
        print(
            "❌ 评测不完整：成功 %d / 应测 %d，保留已有 eval_results.json，不写入不完整结果。"
            % (len(results), len(gold_data))
        )
        return


    print("=" * 70)
    print("汇总统计")
    print("=" * 70)
    
    avg_recall = sum(r['recall'] for r in results) / len(results) if results else 0
    avg_hit = sum(r['hit'] for r in results) / len(results) if results else 0
    avg_answer_recall = sum(r['answer_recall'] for r in results) / len(results) if results else 0
    avg_answer_len = sum(r['answer_length'] for r in results) / len(results) if results else 0
    
    print(f"平均 Recall@5:      {avg_recall:.1%}")
    print(f"平均 Hit@5:         {avg_hit:.1%}")
    print(f"平均 Answer Recall: {avg_answer_recall:.1%}")
    print(f"平均回答长度:       {avg_answer_len:.0f}字")
    print()
    

    output = {
        "run_info": collect_run_info(top_k=5),
        "summary": {
            "total_questions": len(results),
            "avg_recall": avg_recall,
            "avg_hit": avg_hit,
            "avg_answer_recall": avg_answer_recall,
            "avg_answer_length": avg_answer_len,
            "failed_questions": len(failed_questions)
        },
        "details": results
    }
    
    with open(RESULT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 详细结果已保存到：{RESULT_FILE}")
    print()
    

    print("=" * 70)
    print("需要改进的题目（Recall < 50%）")
    print("=" * 70)
    
    bad_questions = [r for r in results if r['recall'] < 0.5]
    if bad_questions:
        for r in bad_questions:
            print(f"\n❌ {r['question']}")
            print(f"   Recall: {r['recall']:.1%}")
            print(f"   缺失：{', '.join(r['missing_keys'][:5])}")
    else:
        print("✅ 所有题目 Recall >= 50%")


if __name__ == "__main__":
    run_evaluation()
