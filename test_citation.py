
import os
import requests

from api_headers import api_headers


__test__ = False

BASE = os.environ.get("RAG_BASE_URL", "http://127.0.0.1:8001").rstrip("/")
URL = BASE + "/ask"


def ask(question, history=None):
    payload = {"question": question, "top_k": 5}
    if history:
        payload["history"] = history
    response = requests.post(
        URL,
        json=payload,
        timeout=180,
        headers=api_headers(),
    )
    response.raise_for_status()
    return response.json()


def main():
    print("=" * 70)
    print("测试1：单轮 + 引用")
    print("=" * 70)
    first = ask("糖尿病患者的血糖控制目标是多少")
    print("回答：")
    print(first.get("answer", "")[:800])
    print("\n来源清单：")
    for source in first.get("sources", []):
        print(f"  [{source['index']}] {source['source']}")

    print()
    print("=" * 70)
    print("测试2：多轮追问")
    print("=" * 70)
    history = [
        {"role": "user", "content": "糖尿病患者的血糖控制目标是多少"},
        {"role": "assistant", "content": first.get("answer", "")},
    ]
    second = ask("那老年人呢", history)
    print("改写后：", second.get("rewritten_question", ""))
    print("回答：")
    print(second.get("answer", "")[:600])
    print("\n来源：", [source["source"] for source in second.get("sources", [])])

    print()
    print("=" * 70)
    print("测试3：知识库外问题（安全边界）")
    print("=" * 70)
    third = ask("量子纠缠的原理是什么")
    print("回答：", third.get("answer", ""))


if __name__ == "__main__":
    main()
