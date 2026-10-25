
import os
import requests

from api_headers import api_headers


__test__ = False

BASE = os.environ.get("RAG_BASE_URL", "http://127.0.0.1:8001").rstrip("/")
URL = BASE + "/ask"


def ask(question):
    response = requests.post(
        URL,
        json={"question": question, "top_k": 5},
        timeout=240,
        headers=api_headers(),
    )
    response.raise_for_status()
    return response.json()


def main():
    outside = ask("量子力学的基本原理是什么")
    print("知识库外问题：", outside.get("answer", ""))
    print("结构化判定：", outside.get("decision"))

    inside = ask("糖尿病患者的血糖控制目标是多少")
    print("知识库内问题：", inside.get("answer", "")[:300])
    print("结构化判定：", inside.get("decision"))


if __name__ == "__main__":
    main()
