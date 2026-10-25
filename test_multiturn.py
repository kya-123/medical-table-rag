
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
    first = ask("糖尿病患者能吃西瓜吗")
    print("第一轮改写后：", first.get("rewritten_question", ""))
    print("第一轮回答：\n", first.get("answer", ""))

    history = [
        {"role": "user", "content": "糖尿病患者能吃西瓜吗"},
        {"role": "assistant", "content": first.get("answer", "")},
    ]
    second = ask("它能吃多少", history)
    print("\n第二轮改写后：", second.get("rewritten_question", ""))
    print("第二轮回答：\n", second.get("answer", ""))

    history.extend([
        {"role": "user", "content": "它能吃多少"},
        {"role": "assistant", "content": second.get("answer", "")},
    ])
    third = ask("那其他水果呢", history)
    print("\n第三轮改写后：", third.get("rewritten_question", ""))
    print("第三轮回答：\n", third.get("answer", ""))


if __name__ == "__main__":
    main()
