
import os
import sqlite3
import time
import uuid
import hashlib
from typing import Optional, List, Literal

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import (
    MAX_UPLOAD_BYTES, UPLOAD_TMP_DIR, READ_BLOCK, HISTORY_DIR, PROFILE_DB,
    KNOWLEDGE_DIR,
    PROFILE_FIELDS, RAG_DOMAIN, RAG_ASSISTANT_NAME,
    load_uploaded_state, save_uploaded_state,
)
from parsing import extract_text, text_hash_of
from store import (
    collection, add_text_to_knowledge_base, _load_profile,
    get_reranker, _build_bm25_index,
)
from retrieval import _retrieve_with_assessment
from evidence_sufficiency import normalize_query_text
from generation import llm, _rewrite_query, _profile_text, _history_text

from contextlib import asynccontextmanager, closing

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print("当前身份：%s（领域：%s）" % (RAG_ASSISTANT_NAME, RAG_DOMAIN))
        get_reranker()
    except Exception as e:
        print(f"【警告】重排模型预热失败，检索层将 fail-closed：{type(e).__name__}: {e}")
    try:
        _build_bm25_index()
    except Exception as e:
        print(f"【警告】BM25 索引构建失败，混合检索会退化为纯向量检索：{type(e).__name__}: {e}")
    yield

app = FastAPI(
    title="RAG智能问答系统（医疗健康版）",
    description="基于语义切片的文档检索与生成问答",
    version="2.0.0",
    lifespan=lifespan
)









API_TOKEN = os.environ.get("RAG_API_TOKEN", "").strip()
_PUBLIC_PATHS = ("/health", "/", "/docs", "/redoc", "/openapi.json")


@app.middleware("http")
async def _check_token(request, call_next):
    if API_TOKEN:
        path = request.url.path
        if not (path in _PUBLIC_PATHS or path.startswith("/docs") or path.startswith("/openapi")):
            if request.headers.get("authorization", "") != "Bearer " + API_TOKEN:
                return JSONResponse(
                    {"detail": "未授权：请在请求头带上 Authorization: Bearer <RAG_API_TOKEN>"},
                    status_code=401)
    return await call_next(request)

class ChatMessage(BaseModel):

    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=4000)


class ProfilePayload(BaseModel):

    age: Optional[int] = Field(None, ge=0, le=120)
    gender: Optional[str] = Field(None, max_length=16)
    conditions: Optional[str] = Field(None, max_length=500)
    duration_years: Optional[float] = Field(None, ge=0, le=100)
    meds: Optional[str] = Field(None, max_length=1000)
    allergies: Optional[str] = Field(None, max_length=500)
    labs: Optional[str] = Field(None, max_length=1000)
    complications: Optional[str] = Field(None, max_length=1000)
    bmi: Optional[float] = Field(None, ge=5, le=80)
    diabetes_type: Optional[str] = Field(None, max_length=100)
    hba1c: Optional[float] = None


class AskRequest(BaseModel):

    question: str = Field(..., min_length=1, max_length=1000)

    top_k: int = Field(5, ge=1, le=20)

    history: Optional[List[ChatMessage]] = None
    profile: Optional[ProfilePayload] = None
    user_id: Optional[str] = Field(None, max_length=64)



class DeleteRequest(BaseModel):
    source: str
    confirm: bool = False


class ProfileRequest(ProfilePayload):
    user_id: str = Field(..., min_length=1, max_length=64)

def _safe_remove(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass

@app.post("/upload", summary="上传文档，构建知识库（带去重）")
def upload_file(file: UploadFile = File(...)):



    safe_name = os.path.basename((file.filename or '').replace('\\', '/')).strip()
    if not safe_name:
        raise HTTPException(status_code=400, detail="文件名无效")
    ext = os.path.splitext(safe_name)[1].lower()
    if ext not in ('.txt', '.docx', '.pdf'):
        raise HTTPException(status_code=400, detail="只支持 txt、docx 或 pdf 文件")

    os.makedirs(UPLOAD_TMP_DIR, exist_ok=True)
    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)


    tmp_path = os.path.join(UPLOAD_TMP_DIR,
                            "uploading_%d_%s%s" % (int(time.time() * 1000),
                                                   uuid.uuid4().hex[:8], ext))
    size = 0
    md5_h = hashlib.md5()
    try:
        with open(tmp_path, "wb") as out:
            while True:
                block = file.file.read(READ_BLOCK)
                if not block:
                    break
                size += len(block)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="文件太大（超过 %d MB）。想放宽就重启前执行：set RAG_MAX_UPLOAD_MB=200"
                               % (MAX_UPLOAD_BYTES // 1048576))
                md5_h.update(block)
                out.write(block)
        md5 = md5_h.hexdigest()
    except HTTPException:
        _safe_remove(tmp_path)
        raise
    except Exception as e:
        _safe_remove(tmp_path)
        raise HTTPException(status_code=400, detail="接收文件失败：%s: %s" % (type(e).__name__, e))

    try:

        state = load_uploaded_state()
        if safe_name in state and state[safe_name] == md5:
            return {"code": 200, "message": f"文件 {safe_name} 已存在且内容未变，跳过上传", "data": None}


        try:
            text = extract_text(tmp_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail="读取 %s 失败：%s: %s" % (safe_name, type(e).__name__, e))

        if not text.strip():


            raise HTTPException(
                status_code=400,
                detail="没能从 %s 里提取到任何文字，未入库。"
                       "如果是扫描版 PDF（整页是图片），请先做文字识别（OCR）再上传。" % safe_name)


        digest = text_hash_of(text)
        dup = collection.get(where={"content_hash": digest}, limit=1)
        if dup["ids"]:
            old_source = "已入库的某份文档"
            metas = dup.get("metadatas") or []
            if metas and metas[0]:
                old_source = metas[0].get("source", old_source)
            return {"code": 200,
                    "message": "%s 的正文内容和已入库的「%s」完全相同，跳过上传" % (safe_name, old_source),
                    "data": None}



        file_path = os.path.join(KNOWLEDGE_DIR, safe_name)
        try:
            os.replace(tmp_path, file_path)
        except PermissionError:

            import shutil
            try:
                shutil.copy2(tmp_path, file_path)
                os.remove(tmp_path)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"文件 {safe_name} 被其他程序占用，无法更新。请关闭 WPS/Word 后重试。错误：{e}"
                )
        tmp_path = None


        count = add_text_to_knowledge_base(text, safe_name)
        if count == 0:
            _safe_remove(file_path)
            raise HTTPException(status_code=400, detail="%s 切分后没有可用片段，未入库" % safe_name)


        state[safe_name] = md5
        save_uploaded_state(state)




        try:
            _build_bm25_index()
        except Exception as e:
            print("【警告】上传后重建 BM25 失败，本次新文档只能靠向量检索：%s: %s"
                  % (type(e).__name__, e))

        return {"code": 200, "message": f"上传成功，已切分为{count}个片段", "data": None}
    finally:

        if tmp_path:
            _safe_remove(tmp_path)

@app.post("/ask", summary="向知识库提问")
def ask(request: AskRequest):


    history = [message.model_dump() for message in (request.history or [])]
    profile = (
        request.profile.model_dump(exclude_none=True)
        if request.profile
        else _load_profile(request.user_id)
    )

    if llm is None:
        raise HTTPException(
            status_code=503,
            detail="未配置 DeepSeek API Key，/ask 暂时不可用；请设置环境变量 DEEPSEEK_API_KEY",
        )



    rewritten_query = _rewrite_query(request.question, history)
    if rewritten_query != request.question:
        print("【多轮】原问题「%s」→ 重写为「%s」" % (request.question, rewritten_query))


    normalization = normalize_query_text(rewritten_query)
    query = normalization["normalized_question"]
    if query != rewritten_query:
        print("【问题归一化】「%s」→「%s」" % (rewritten_query, query))


    docs, assessment = _retrieve_with_assessment(query, request.top_k)

    if not docs:
        return {
            "code": 200,
            "message": "success",
            "answer": "未找到相关信息",
            "context": "",
            "sources": [],
            "unique_sources": [],
            "refused": True,
            "rewritten_question": query,
            "normalization": normalization,
            "decision": "refuse",
            "refusal_reason": assessment.get("reason", "证据不足"),
            "sufficiency": assessment,
            "missing_facets": assessment.get("missing_facets", []),
            "data": {"answer": "未找到相关信息", "context": "", "sources": [],
                     "unique_sources": [], "refused": True,
                     "decision": "refuse",
                     "refusal_reason": assessment.get("reason", "证据不足"),
                     "sufficiency": assessment,
                     "missing_facets": assessment.get("missing_facets", [])}
        }





    context_parts, sources = [], []
    for i, (text, src) in enumerate(docs, 1):
        label = src or "未知来源"
        context_parts.append("【资料%d·%s】\n%s" % (i, label, text))
        sources.append({"index": i, "source": label})
    context = "\n\n".join(context_parts)

    unique_sources = list(dict.fromkeys(s["source"] for s in sources))
    profile_block = _profile_text(profile)
    history_block = _history_text(history)


    parts = [
        "你是一位专业的%s。请严格根据下面提供的资料回答问题。" % RAG_ASSISTANT_NAME,
        "",
        "回答要求：",
        "1. 尽量详细、完整 —— 把资料中与问题相关的信息都讲清楚，不要只给一句话结论；",
        "2. 分点作答，每一点写清楚具体数值、标准或做法；",
        "3. 资料里如果针对不同人群（年龄、性别、孕产期、合并症、肝肾功能等）有不同说法，"
        "   【优先讲与用户档案相符的那一类】，其余人群简要带过；",
        "4. 只使用资料中的信息，不要编造资料里没有的内容；",
        "5. 如果资料完全不足以回答这个问题，只回复\"未找到相关信息\"，不要自己发挥；",
        "6. 每条关键结论后面用【资料N】标出它出自哪一段资料（例：空腹血糖目标 4.4~6.1 mmol/L【资料3】），"
        "   不要在正文里抄资料文件名；",
        "7. 涉及具体用药、剂量、手术等高风险决策时，明确提醒以医生面诊意见为准。",
    ]
    if profile_block:
        parts += ["", "【用户健康档案】", profile_block,
                  "（年龄、诊断、用药、合并症等都会影响结论，请据此挑出对这位用户适用的部分。）"]
    if history_block:
        parts += ["", "【最近对话】", history_block,
                  "（注意：下面的【当前问题】是经过改写的独立问题，便于检索。"
                  "请结合上面的历史对话理解用户的原始意图，回答时直接针对原始问题。）"]
    if assessment.get("decision") == "partial":
        missing_names = [
            item.get("facet", "")
            for item in assessment.get("missing_facets", [])
            if item.get("facet")
        ]
        missing_text = "、".join(missing_names) or "部分关键信息"
        parts += [
            "",
            "【证据完整性约束】",
            "当前资料只覆盖了问题的一部分，缺少：%s。" % missing_text,
            "只能回答有资料支持的内容；必须明确写出“当前知识库未提供……”或"
            "“现有资料不足以确认……”，禁止补充资料中没有的数值、步骤或结论。",
            "不要把“当前知识库未检索到”表述成“医学上不存在”。",
        ]
    parts += ["", "【参考资料】", context, "", "【当前问题】", query, "", "回答："]
    prompt = "\n".join(parts)



    try:
        resp = llm.invoke(prompt)

        answer = resp.content
        if isinstance(answer, list):
            answer = "".join(b.get("text", "") if isinstance(b, dict) else str(b)
                             for b in answer)
        answer = (answer or "").strip() or "未找到相关信息"
    except Exception as e:

        print("[ask] 调用失败：prompt=%d 字 / context=%d 条 / 历史=%d 条 / %s: %s"
              % (len(prompt), len(docs), len(history), type(e).__name__, e))
        raise HTTPException(status_code=500, detail=f"调用 DeepSeek 失败：{type(e).__name__}: {e}")


    model_refused = answer.strip().startswith("未找到相关信息")
    decision = "refuse" if model_refused else assessment.get("decision", "answer")
    refused = bool(model_refused)
    if model_refused:
        assessment = {
            **assessment,
            "decision": "refuse",
            "reason": "生成阶段判断资料不足以回答",
        }
    response_extra = {
        "normalization": normalization,
        "decision": decision,
        "model_refused": model_refused,
        "refusal_reason": (
            "生成阶段判断资料不足以回答"
            if model_refused
            else assessment.get("reason", "")
        ),
        "sufficiency": assessment,
        "missing_facets": assessment.get("missing_facets", []),
    }
    return {
        "code": 200,
        "message": "success",
        "answer": answer,
        "context": context,
        "sources": sources,
        "unique_sources": unique_sources,
        "refused": refused,
        "rewritten_question": query,
        **response_extra,
        "data": {"answer": answer, "context": context, "sources": sources,
                 "unique_sources": unique_sources, "refused": refused,
                 "rewritten_question": query, **response_extra}
    }



@app.get("/health", summary="健康检查")
async def health_check():
    return {"code": 200, "message": "RAG服务运行正常（支持txt/docx/pdf）", "data": None}




@app.post("/profile", summary="保存用户健康档案")
def save_profile(p: ProfileRequest):
    data = p.model_dump() if hasattr(p, "model_dump") else p.dict()
    user_id = str(data.pop("user_id"))


    if not data.get("conditions") and data.get("diabetes_type"):
        data["conditions"] = data["diabetes_type"]
    if not data.get("labs") and data.get("hba1c") is not None:
        data["labs"] = "糖化血红蛋白 %s%%" % data["hba1c"]


    allowed = {k for k, _label, _t in PROFILE_FIELDS}
    data = {k: v for k, v in data.items() if k in allowed and v is not None}

    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    cols = ["user_id"] + list(data.keys()) + ["updated_at"]
    vals = [user_id] + list(data.values()) + [ts]
    updates = ", ".join("%s=excluded.%s" % (c, c) for c in list(data.keys()) + ["updated_at"])
    sql = ("INSERT INTO profiles (%s) VALUES (%s) "
           "ON CONFLICT(user_id) DO UPDATE SET %s"
           % (", ".join(cols), ", ".join("?" * len(cols)), updates))

    try:
        with closing(sqlite3.connect(PROFILE_DB)) as conn:
            with conn:
                conn.execute(sql, vals)
    except Exception as e:
        raise HTTPException(status_code=500, detail="保存档案失败：%s" % e)

    filled = sum(1 for v in data.values() if v not in (None, ""))
    return {
        "code": 200,
        "message": "档案已保存（填写了 %d/%d 项）" % (filled, len(data)),
        "data": {"user_id": user_id,
                 "profile": {k: v for k, v in data.items() if v not in (None, "")}},
    }

@app.get("/profile/{user_id}", summary="读取用户健康档案")
def get_profile(user_id: str):
    prof = _load_profile(user_id)
    if not prof or all(v in (None, "") for v in prof.values()):
        return {"code": 200, "message": "这个人还没有档案", "data": None}
    return {"code": 200, "message": "success", "data": prof}


@app.delete("/profile/{user_id}", summary="删除用户健康档案")
def delete_profile(user_id: str):
    try:
        with closing(sqlite3.connect(PROFILE_DB)) as conn:
            with conn:
                cursor = conn.execute(
                    "DELETE FROM profiles WHERE user_id = ?",
                    (str(user_id),),
                )
    except Exception as e:
        raise HTTPException(status_code=500, detail="删除档案失败：%s" % e)
    return {
        "code": 200,
        "message": "档案已删除" if cursor.rowcount else "这个人还没有档案",
        "data": {"user_id": str(user_id), "deleted": bool(cursor.rowcount)},
    }


@app.get("/profiles", summary="列出所有已建档用户（默认关闭，隐私考虑）")
def list_profiles():


    if os.environ.get("RAG_ALLOW_PROFILE_LIST", "").strip().lower() not in ("1", "true", "yes"):
        raise HTTPException(
            status_code=403,
            detail="出于隐私考虑，批量列出用户档案默认关闭。确需使用请设置环境变量 RAG_ALLOW_PROFILE_LIST=1")
    try:
        with closing(sqlite3.connect(PROFILE_DB)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT user_id, updated_at FROM profiles ORDER BY updated_at DESC"
            ).fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail="读档案库失败：%s" % e)
    return {"code": 200, "message": "success", "total": len(rows),
            "data": [{"user_id": r["user_id"], "updated_at": r["updated_at"]} for r in rows]}

@app.get("/sources", summary="列出知识库里有哪些文档、各多少片段")
def list_sources():
    data = collection.get(include=["metadatas"])
    agg = {}
    for m in data["metadatas"]:
        s = (m or {}).get("source", "(未知来源)")
        agg[s] = agg.get(s, 0) + 1
    items = sorted(agg.items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        "code": 200,
        "message": "success",
        "total_chunks": len(data["ids"]),
        "total_sources": len(items),
        "sources": [{"source": s, "chunks": n} for s, n in items],
    }

@app.post("/delete", summary="按来源删除一份文档（手动替换旧资料用）")
def delete_source(req: DeleteRequest):

    name = os.path.basename((req.source or "").replace('\\', '/')).strip()
    if not name:
        raise HTTPException(status_code=400, detail="必须指定 source（要删除的文件名）")

    got = collection.get(where={"source": name}, include=[])
    ids = got.get("ids") or []
    if not ids:
        raise HTTPException(status_code=404, detail="库里找不到来源为「%s」的片段。" % name)


    if not req.confirm:
        return {
            "code": 200,
            "message": "【预览】将删除「%s」的 %d 个片段（尚未执行）。确认请再调一次并传 confirm=true" % (name, len(ids)),
            "data": {"source": name, "chunks": len(ids), "deleted": False},
        }



    moved_to = ""
    src_path = os.path.join(KNOWLEDGE_DIR, name)
    if os.path.exists(src_path):
        os.makedirs(HISTORY_DIR, exist_ok=True)
        moved_to = os.path.join(HISTORY_DIR, "%s__%s" % (time.strftime("%Y%m%d_%H%M%S"), name))
        os.replace(src_path, moved_to)


    collection.delete(where={"source": name})




    try:
        _build_bm25_index()
    except Exception as e:
        print("【警告】删除后重建 BM25 失败，旧文档可能仍被检索到：%s: %s"
              % (type(e).__name__, e))


    state = load_uploaded_state()
    state.pop(name, None)
    save_uploaded_state(state)

    return {
        "code": 200,
        "message": "已下线「%s」：%d 个片段不再参与检索；源文件已移到 %s"
                   % (name, len(ids), moved_to or "(磁盘上没有找到该文件)"),
        "data": {"source": name, "chunks": len(ids), "deleted": True, "moved_to": moved_to},
    }
