
import os
from langchain_openai import ChatOpenAI

from config import API_KEY, _FOLLOWUP_HINTS, PROFILE_FIELDS

if API_KEY:
    llm = ChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key=API_KEY,
        temperature=0.3,
        max_tokens=1500,
        timeout=120,
    )
    print("大模型初始化完成：DeepSeek / deepseek-chat")
else:
    llm = None

def _profile_text(profile):
    if not profile:
        return ""
    parts = []
    for key, label, _t in PROFILE_FIELDS:
        v = profile.get(key)
        if v is None or v == "" or v == []:
            continue
        parts.append("%s：%s" % (label, v))
    return "；".join(parts)

def _history_text(history, limit=6):
    if not history:
        return ""
    lines = []
    for m in history[-limit:]:
        role = "用户" if (m or {}).get("role") == "user" else "助手"
        c = ((m or {}).get("content") or "").strip().replace("\n", " ")
        if c:
            lines.append("%s：%s" % (role, c[:200]))
    return "\n".join(lines)

def _needs_rewrite(question, history):
    if not history:
        return False
    q = (question or "").strip()
    if len(q) <= 8:
        return True
    return any(h in q for h in _FOLLOWUP_HINTS)

def _rewrite_query(question, history):
    if not _needs_rewrite(question, history):
        return question
    try:
        msgs = [{"role": "system", "content":
                 "你是查询重写助手。把用户最后一句改写成一句语义完整、"
                 "适合检索的独立问题。规则："
                 "① 把代词（它、这个、那个）替换成对话历史里明确指代的事物；"
                 "② 修正错别字；③ 保持原意，不要回答问题，不要加解释；"
                 "④ 只输出改写后的那一句，不要引号和前缀。"
                 "⑤ 绝对不要回答问题，只输出改写后的问题。"}]

        for m in history[-6:]:
            if (m or {}).get("role") == "user":
                c = ((m or {}).get("content") or "").strip().replace("\n", " ")
                if c:
                    msgs.append({"role": "user", "content": c[:200]})
        msgs.append({"role": "user", "content": question})

        resp = llm.invoke(msgs)
        out = resp.content
        if isinstance(out, list):
            out = "".join(b.get("text", "") if isinstance(b, dict) else str(b)
                          for b in out)
        out = (out or "").strip().strip('"').strip("'").strip()


        if 2 <= len(out) <= 50 and "：" not in out[:6] and "。" not in out:
            return out
        print("【提示】改写结果异常（%s），退回原问题。" % out[:50])
        return question
    except Exception as e:
        print("【提示】查询重写失败（%s），直接用原问题检索。" % e)
        return question
