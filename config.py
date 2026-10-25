
import os
import sys
import json
import re


BASE_DIR = os.path.dirname(os.path.abspath(__file__))




DEEPSEEK_API_KEY = ""

UPLOADED_STATE_FILE = os.path.join(BASE_DIR, "uploaded_state.json")

def load_uploaded_state() -> dict:
    if os.path.exists(UPLOADED_STATE_FILE):
        try:
            with open(UPLOADED_STATE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print("【警告】上传台账损坏，按空台账处理：%s" % e)
    return {}

def save_uploaded_state(state: dict):
    tmp_file = UPLOADED_STATE_FILE + ".tmp"
    with open(tmp_file, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_file, UPLOADED_STATE_FILE)


def _default_chroma_dir() -> str:
    project_dir = os.path.join(BASE_DIR, "chroma_db")
    safe_path = bool(re.fullmatch(r"[A-Za-z0-9_\-./:\\]+", project_dir))
    if safe_path:
        return project_dir
    data_roots = (
        os.environ.get("LOCALAPPDATA"),
        os.environ.get("XDG_DATA_HOME"),
        os.environ.get("TEMP") or "/tmp",
    )
    candidates = [
        os.path.join(root, "MedRAG", "chroma_db")
        for root in data_roots
        if root
    ]
    candidates.append(
        os.path.join(os.environ.get("SystemDrive", "C:") + os.sep, "MedRAG", "chroma_db")
    )
    fallback_dir = ""
    for candidate in candidates:
        if re.fullmatch(r"[A-Za-z0-9_\-./:\\]+", candidate):
            fallback_dir = candidate
            break
    if not fallback_dir:
        raise RuntimeError(
            "Chroma 索引路径包含中文或空格，且没有找到安全的 fallback。"
            "请显式设置 RAG_CHROMA_DIR 为纯 ASCII 目录。"
        )
    print("【提示】项目路径含中文或空格，Chroma 索引使用安全路径：%s" % fallback_dir)
    return fallback_dir


def _pick_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            print("检测到 GPU：%s（CUDA %s），向量化 / 重排将使用显卡"
                  % (torch.cuda.get_device_name(0), torch.version.cuda))
            return "cuda"
        if torch.version.cuda is None:
            print("【提示】当前 torch 是 CPU 版（+cpu），向量化/重排只能跑 CPU。"
                  "想用显卡见 README「可选：启用 GPU」一节。")
        else:
            print("【提示】torch 带 CUDA 但显卡不可用（驱动/显存问题），本次用 CPU。")
    except Exception as e:
        print("【提示】检查 GPU 失败（%s: %s），本次用 CPU" % (type(e).__name__, e))
    return "cpu"

DEVICE = (os.environ.get("RAG_DEVICE") or _pick_device()).strip()

CONFIG_FILE = os.path.join(BASE_DIR, "config.txt")

def _read_key_from_config_file() -> str:
    if not os.path.exists(CONFIG_FILE):
        return ""
    with open(CONFIG_FILE, "rb") as f:
        raw = f.read()
    text = ""

    for enc in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip().upper() == "DEEPSEEK_API_KEY":
            return value.strip().strip('"').strip("'")
    return ""

def _save_key_to_config_file(key: str) -> None:
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write("# 这是程序自动保存的 DeepSeek API Key\n")
            f.write("# 申请地址：https://platform.deepseek.com/\n")
            f.write("DEEPSEEK_API_KEY=" + key + "\n")
    except Exception as e:
        print(f"（警告：key 写入 config.txt 失败：{e}）")

def load_api_key() -> str:

    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        print("已从环境变量 DEEPSEEK_API_KEY 读取密钥")
        return key


    key = _read_key_from_config_file()
    if key:
        print("已从 config.txt 读取密钥")
        return key


    key = DEEPSEEK_API_KEY.strip()
    if key:
        print("【警告】正在使用 config.py 中硬编码的临时密钥")
        return key


    if sys.stdin and sys.stdin.isatty():
        print("=" * 62)
        print("  还没有配置 DeepSeek API Key")
        print("  申请地址：https://platform.deepseek.com/")
        print("           登录 → 左侧「API keys」→ 创建 → 复制 sk- 开头那串")
        print("=" * 62)
        try:
            key = input("请把 API Key 粘贴到这里，然后按回车：").strip()
        except (EOFError, KeyboardInterrupt):
            key = ""
        if key:
            _save_key_to_config_file(key)
            print("已保存到 config.txt，下次启动不用再输入了。")
            return key
    return ""

API_KEY = load_api_key()
if not API_KEY:
    print(
        "【提示】没有找到 DeepSeek API Key。/health、索引检查和无生成测试仍可运行；"
        "/ask 将返回 503。配置方式见 README。"
    )

PROFILE_DB = os.path.join(BASE_DIR, "profiles.db")
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "my_knowledge")
CHROMA_DB_DIR = os.environ.get("RAG_CHROMA_DIR") or _default_chroma_dir()








RAG_DOMAIN = os.environ.get("RAG_DOMAIN", "医疗健康")
RAG_ASSISTANT_NAME = os.environ.get("RAG_ASSISTANT_NAME", RAG_DOMAIN + "助手")





_PROFILES = {
    "generic":  ("医疗健康", "医疗健康助手"),
    "diabetes": ("糖尿病",   "糖尿病健康管理助手"),
}
_profile = os.environ.get("RAG_PROFILE", "").strip().lower()
if _profile in _PROFILES:
    RAG_DOMAIN, RAG_ASSISTANT_NAME = _PROFILES[_profile]

if os.environ.get("RAG_DOMAIN"):
    RAG_DOMAIN = os.environ["RAG_DOMAIN"]
if os.environ.get("RAG_ASSISTANT_NAME"):
    RAG_ASSISTANT_NAME = os.environ["RAG_ASSISTANT_NAME"]



PROFILE_FIELDS = [
    ("age",            "年龄",              "int"),
    ("gender",         "性别",              "str"),
    ("conditions",     "主要诊断/基础疾病",  "str"),
    ("duration_years", "病程（年）",         "float"),
    ("meds",           "正在服用的药物",     "str"),
    ("allergies",      "过敏史",            "str"),
    ("labs",           "关键化验指标",       "str"),
    ("complications",  "并发症/合并症",      "str"),
    ("bmi",            "BMI",               "float"),
]

_FOLLOWUP_HINTS = ("它", "这个", "那个", "这些", "那些", "上面", "刚才", "前面",
                   "还有", "呢", "这种", "该")

MAX_UPLOAD_BYTES = int(os.environ.get("RAG_MAX_UPLOAD_MB", "50")) * 1024 * 1024
MAX_DOCX_UNCOMPRESSED_BYTES = (
    int(os.environ.get("RAG_MAX_DOCX_UNCOMPRESSED_MB", "200"))
    * 1024
    * 1024
)
MAX_DOCX_ENTRIES = max(100, int(os.environ.get("RAG_MAX_DOCX_ENTRIES", "10000")))
MAX_PDF_PAGES = max(1, int(os.environ.get("RAG_MAX_PDF_PAGES", "1000")))



UPLOAD_TMP_DIR = os.path.join(BASE_DIR, "upload_tmp")
READ_BLOCK = 1 << 20

RERANK_MODEL_PATH = os.environ.get(
    "RAG_RERANK_MODEL",
    os.path.join(BASE_DIR, "model", "models", "BAAI--bge-reranker-base", "snapshots", "master"))
EMBEDDING_MODEL_PATH = os.path.join(
    BASE_DIR, "model", "models", "Xorbits--bge-large-zh-v1.5", "snapshots", "master"
)

try:
    RERANK_POOL = int(os.environ.get("RAG_RERANK_POOL", "40"))
except ValueError:
    RERANK_POOL = 40

RERANK_POOL = max(10, min(100, RERANK_POOL))

DEDUP_ADJACENT = max(0, int(os.environ.get("RAG_DEDUP_ADJACENT", "0")))

RERANK_MIN_SCORE = float(os.environ.get("RAG_MIN_RERANK_SCORE", "0.3"))



EVIDENCE_INTENT_MIN_SCORE = float(
    os.environ.get("RAG_EVIDENCE_INTENT_MIN_SCORE", str(RERANK_MIN_SCORE))
)
EVIDENCE_INTENT_FULL_SCORE = float(
    os.environ.get("RAG_EVIDENCE_INTENT_FULL_SCORE", "0.68")
)
EVIDENCE_ENTITY_MIN_SCORE = float(
    os.environ.get("RAG_EVIDENCE_ENTITY_MIN_SCORE", "0.45")
)
EVIDENCE_COVERAGE_PARTIAL_SCORE = float(
    os.environ.get("RAG_EVIDENCE_COVERAGE_PARTIAL_SCORE", "0.36")
)
EVIDENCE_COVERAGE_FULL_SCORE = float(
    os.environ.get("RAG_EVIDENCE_COVERAGE_FULL_SCORE", "0.65")
)

HYBRID_ON = os.environ.get("RAG_HYBRID", "1") not in ("0", "false", "False")
KEYWORD_MAX_TERMS = 8
KEYWORD_MAX_HITS = 400
KEYWORD_MAX_CHUNKS = 15
KEYWORD_RARE_MAX = 10

KEYWORD_KEEP = 2


_JUNK_CHARS = set("的吗呢吧啊呀么什怎哪有些多是否能否和与或了很太就都还也"
                  "我你他她它这那一上下前后怎么可以要不要有没有两个")

HISTORY_DIR = os.path.join(KNOWLEDGE_DIR, "_历史版本")
