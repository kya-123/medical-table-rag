
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Iterable, List, Optional, Sequence, Tuple

import jieba.posseg as pseg

from config import (
    EVIDENCE_COVERAGE_FULL_SCORE,
    EVIDENCE_COVERAGE_PARTIAL_SCORE,
    EVIDENCE_ENTITY_MIN_SCORE,
    EVIDENCE_INTENT_FULL_SCORE,
    EVIDENCE_INTENT_MIN_SCORE,
)




SAFE_QUERY_ALIASES = {
    "心机梗死": "心肌梗死",
    "心梗": "心肌梗死",
    "阿卡波堂": "阿卡波糖",
    "二甲双瓜": "二甲双胍",
}

_INTENT_RULES = (
    ("comparison", ("区别", "对比", "比较", "哪个更", "哪个好", "有何不同", "差异")),
    ("quantity", ("多少", "几", "多久", "多长时间", "周期", "频率",
                  "剂量", "用量", "摄入量", "目标值", "多少克", "多少单位")),
    ("formula", ("怎么算", "如何计算", "计算公式", "公式", "算法")),
    ("side_effect", ("副作用", "不良反应", "危害", "风险")),
    ("procedure", ("怎么办", "怎么", "如何", "步骤", "用法", "处理方法",
                   "漏服", "漏打", "补服", "补打", "服用方法")),
    ("definition", ("是什么", "定义", "指什么", "指的是", "诊断标准", "单位是")),
    ("safety", ("能不能", "可不可以", "是否", "适合", "安全", "禁忌",
                "能吃", "可以吃", "能否")),
)

_INTENT_SLOT_NAMES = {
    "comparison": "比较结论",
    "quantity": "具体数值或时间",
    "formula": "计算公式",
    "side_effect": "不良反应",
    "procedure": "操作步骤",
    "definition": "概念定义",
    "safety": "安全性结论",
    "general": "直接答案",
}

_SLOT_TOKENS = {
    "什么", "哪些", "哪个", "多少", "几个", "几次", "如何", "怎么", "怎样",
    "是否", "可以", "能不能", "可不可以", "需要", "应该", "请问", "患者",
    "资料", "一下", "这个", "那个", "以及", "还是", "分别", "目前", "一般",
    "主要", "相关", "问题", "方法", "情况", "时候", "之后", "以后", "哪些年",
    "什么药", "吃什么", "怎么办", "区别", "对比", "比较", "步骤", "用法",
    "多久", "多长时间", "周期", "频率", "公式", "计算", "校准",
    "剂量", "用量", "摄入量", "目标", "目标值", "副作用", "不良反应", "风险",
    "定义", "意思", "安全", "禁忌",
}

_HIGH_RISK_MARKERS = (
    "胰岛素", "剂量", "补服", "补打", "漏服", "漏打", "过量",
    "急救", "抢救", "中毒", "禁用药", "孕妇", "妊娠", "哺乳", "儿童",
    "肾功能", "肝功能", "透析", "过敏", "相互作用",
)

_HIGH_RISK_CONDITIONS = (
    "孕妇", "妊娠", "哺乳", "儿童", "婴儿", "老年", "肾功能", "肝功能",
    "透析", "过敏", "并发症", "合并症",
)

_DRUG_PATTERN = re.compile(
    r"(二甲双胍|阿卡波糖|格列|达格列净|恩格列净|司美格鲁肽|胰岛素|"
    r"SGLT2i|DPP-4i|GLP-1(?:RA)?|TZD|"
    r"[一-龥]{3,10}(?:片|胶囊|注射液|药物))",
    re.IGNORECASE,
)

_DRUG_NAME_PATTERN = re.compile(
    r"(二甲双胍|阿卡波糖|格列齐特|格列本脲|格列吡嗪|"
    r"达格列净|恩格列净|司美格鲁肽|SGLT-?2|DPP-?4i|GLP-?1(?:RA)?)"
)
_EXACT_DRUG_FACETS = {
    "二甲双胍", "阿卡波糖", "格列齐特", "格列本脲", "格列吡嗪",
    "达格列净", "恩格列净", "司美格鲁肽", "SGLT-2", "SGLT2",
    "DPP-4i", "DPP4i", "GLP-1", "GLP-1RA",
}
_GENERIC_DRUG_FACETS = {"降糖药", "药物", "类型", "糖尿病人"}

_NUMERIC_PATTERN = re.compile(
    r"\d+(?:\.\d+)?(?:\s*[~～\-至]\s*\d+(?:\.\d+)?)?"
    r"\s*(?:%|mg|g|kg|ml|mL|mmol/L|单位|片|粒|次|年|岁|周|天|小时|克|毫克)"
)

_FREQUENCY_PATTERN = re.compile(
    r"(?:每天|每日|每周|每月|每半年|每年|每\s*\d+\s*个?月|"
    r"每\s*\d+\s*年)"
    r".{0,8}?(?:[一二三四五六七八九十\d]+\s*次|一次|两次|三次)"
)



_MISSING_SUPPORT_FLOOR = 0.15



_FACET_EQUIVALENTS = {
    "孕妇": ("孕妇", "妊娠", "妊娠期", "孕期"),
    "妊娠": ("妊娠", "孕妇", "妊娠期", "孕期"),
    "老年": ("老年", "老人", "老年人"),
    "老人": ("老人", "老年", "老年人"),
    "血糖仪": ("血糖仪", "SMBG", "自我血糖监测", "指尖血糖"),
    "SMBG": ("SMBG", "血糖仪", "自我血糖监测", "指尖血糖"),
    "CGM": ("CGM", "持续葡萄糖监测", "持续血糖监测"),
    "持续葡萄糖监测": ("持续葡萄糖监测", "CGM", "持续血糖监测"),
}

_PRINCIPAL_QUANTITY_MARKERS = (
    "多久", "多长时间", "周期", "频率", "几次", "何时", "什么时候",
    "多少片", "多少单位", "剂量", "用量",
)

_GENERIC_COMPARISON_FACETS = {"糖尿病", "疾病", "患者", "人群", "情况", "指标"}
_CGM_DEVICE_MARKERS = ("CGM", "持续葡萄糖监测", "持续血糖监测", "扫描式葡萄糖监测")

_COMPARISON_SPLIT = re.compile(r"(?:和|与|跟|、|及|VS\.?|vs\.?)", re.IGNORECASE)
_COMPARISON_TAIL = re.compile(r"(?:哪个|哪一个|有何不同|有什么区别|区别|对比|差异).*$")


@dataclass
class Facet:

    name: str
    role: str
    support: float = 0.0
    weight: float = 1.0
    evidence_index: Optional[int] = None


@dataclass
class EvidenceAssessment:

    decision: str
    reason: str
    intent: str
    risk_level: str
    intent_score: float
    mandatory_coverage: float
    optional_coverage: float
    answer_slot_supported: bool
    score: float
    mandatory_facets: List[Facet] = field(default_factory=list)
    optional_facets: List[Facet] = field(default_factory=list)
    missing_facets: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_query_text(question: str) -> dict:
    original = (question or "").strip()
    normalized = original
    changes = []

    for src in sorted(SAFE_QUERY_ALIASES, key=len, reverse=True):
        dst = SAFE_QUERY_ALIASES[src]
        if src in normalized:
            normalized = normalized.replace(src, dst)
            changes.append({"from": src, "to": dst, "type": "safe_alias"})
    return {
        "original_question": original,
        "normalized_question": normalized,
        "changed": bool(changes),
        "changes": changes,
    }


def classify_intent(question: str) -> str:
    q = question or ""
    for intent, markers in _INTENT_RULES:
        if any(marker in q for marker in markers):
            return intent
    return "general"


def _facet_variants(term: str) -> Tuple[str, ...]:
    return _FACET_EQUIVALENTS.get(term, (term,))


def _extract_facets(question: str) -> List[Facet]:
    facets: List[Facet] = []
    seen = set()


    for match in re.finditer(r"(?:1|2|Ⅰ|Ⅱ)\s*型", question or ""):
        token = re.sub(r"\s+", "", match.group())
        if token not in seen:
            facets.append(Facet(name=token, role="entity"))
            seen.add(token)


    explicit_drugs = []
    for match in _DRUG_NAME_PATTERN.finditer(question or ""):
        token = match.group().replace(" ", "")
        if token not in seen:
            facets.append(Facet(name=token, role="entity"))
            seen.add(token)
            explicit_drugs.append(token)
    has_specific_drugs = bool(explicit_drugs)

    if classify_intent(question) == "comparison":
        compare_text = _COMPARISON_TAIL.sub("", question or "")
        compare_text = re.sub(r"^.*?(?:请问|比较|对比)", "", compare_text).strip()
        for token in _COMPARISON_SPLIT.split(compare_text):
            token = re.sub(r"^[\s，,：:]+|[\s，,：:？?]+$", "", token)
            token = re.sub(r"\s+", "", token)
            if 2 <= len(token) <= 20 and not any(
                token in existing or existing in token
                for existing in seen
            ):
                facets.append(Facet(name=token, role="entity"))
                seen.add(token)

    for word, flag in pseg.cut(question or ""):
        token = word.strip()
        if len(token) < 2 or token in _SLOT_TOKENS or token in seen:
            continue
        if has_specific_drugs and token in _GENERIC_DRUG_FACETS:
            continue
        if has_specific_drugs and any(
            token in drug or drug in token for drug in explicit_drugs
        ):
            continue

        if not (flag.startswith("n") or flag.startswith("t") or flag in ("vn", "eng", "an")):
            continue
        if token.isdigit():
            continue
        role = "condition" if any(x in token for x in _HIGH_RISK_CONDITIONS) else "entity"
        weight = 1.2 if role == "condition" else 1.0
        facets.append(Facet(name=token, role=role, weight=weight))
        seen.add(token)
        if len(facets) >= 6:
            break
    return facets


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _bigrams(text: str) -> set:
    value = _compact(text)
    if len(value) < 2:
        return {value} if value else set()
    return {value[i:i + 2] for i in range(len(value) - 1)}


def _sliding_sequence_similarity(term: str, document: str) -> float:
    term = _compact(term)
    document = _compact(document)
    if not term or not document:
        return 0.0
    if term in document:
        return 1.0
    width = len(term)
    if len(document) <= width:
        return SequenceMatcher(None, term, document).ratio()
    best = 0.0

    step = max(1, len(document) // 256)
    for start in range(0, len(document) - width + 1, step):
        ratio = SequenceMatcher(None, term, document[start:start + width]).ratio()
        if ratio > best:
            best = ratio
            if best >= 0.96:
                break
    return best


def _facet_support(term: str, document: str) -> float:
    compact_term = _compact(term)
    compact_doc = _compact(document)
    if not compact_term or not compact_doc:
        return 0.0
    if compact_term in compact_doc:
        return 1.0
    term_bg = _bigrams(compact_term)
    doc_bg = _bigrams(compact_doc)
    overlap = len(term_bg & doc_bg) / max(1, len(term_bg))
    sequence = _sliding_sequence_similarity(compact_term, compact_doc)
    return min(1.0, max(overlap, sequence * 0.9))


def _best_facet_support(term: str, documents: Sequence[str]) -> Tuple[float, Optional[int]]:
    best_score = 0.0
    best_index = None
    for index, document in enumerate(documents):
        variants = _facet_variants(term)
        if (
            re.fullmatch(r"(?:1|2|Ⅰ|Ⅱ)型", _compact(term))
            or _compact(term) in _EXACT_DRUG_FACETS
        ):

            compact_doc = _compact(document)
            score = max(
                1.0 if _compact(variant) in compact_doc else 0.0
                for variant in variants
            )
        else:
            score = max(_facet_support(variant, document) for variant in variants)
        if score > best_score:
            best_score = score
            best_index = index
    return best_score, best_index


def _weighted_geometric_mean(facets: Iterable[Facet]) -> float:
    items = [f for f in facets if f.weight > 0]
    if not items:
        return 0.0
    total_weight = sum(f.weight for f in items)
    log_sum = sum(
        f.weight * math.log(max(_MISSING_SUPPORT_FLOOR, f.support))
        for f in items
    )
    return math.exp(log_sum / total_weight)


def _clauses(document: str) -> List[str]:
    return [
        clause.strip()
        for clause in re.split(r"[。；！？\n]+", document or "")
        if clause.strip()
    ]


def _clause_mentions_facet(clause: str, facets: Sequence[Facet]) -> bool:
    compact_clause = _compact(clause)
    return any(
        _compact(variant) in compact_clause
        for facet in facets
        for variant in _facet_variants(facet.name)
    )


def _clause_matches_target_device(clause: str, question: str) -> bool:
    mentions_meter = any(
        marker in question
        for marker in ("血糖仪", "SMBG", "自我血糖监测", "指尖血糖")
    )
    mentions_cgm = any(marker in question for marker in ("CGM", "持续葡萄糖监测"))
    if mentions_meter and not mentions_cgm:
        return not any(marker in clause for marker in _CGM_DEVICE_MARKERS)
    return True


def _answer_slot_supported(
    intent: str,
    documents: Sequence[str],
    facets: Sequence[Facet],
    question: str,
) -> bool:
    clauses = [clause for document in documents for clause in _clauses(document)]
    if not clauses:
        return False

    if intent == "comparison":

        comparison_facets = [
            facet for facet in facets
            if facet.name not in _GENERIC_COMPARISON_FACETS
        ]
        supported_names = []
        for facet in sorted(comparison_facets, key=lambda item: len(item.name)):
            if facet.support < EVIDENCE_ENTITY_MIN_SCORE:
                continue
            name = _compact(facet.name)
            if any(name in existing or existing in name for existing in supported_names):
                continue
            supported_names.append(name)
        return len(supported_names) >= 2

    def clause_has_slot(clause: str) -> bool:
        if intent == "quantity":
            return bool(
                _NUMERIC_PATTERN.search(clause)
                or _FREQUENCY_PATTERN.search(clause)
            ) and ("校准" not in question or "校准" in clause)
        if intent == "formula":
            return (
                "=" in clause
                or "＝" in clause
                or "除以" in clause
                or ("体重" in clause and "身高" in clause)
            )
        if intent == "procedure":
            return any(x in clause for x in (
                "步骤", "先", "然后", "再", "最后", "用法", "服用", "注射",
                "补服", "停药", "立即", "监测", "处理", "口服", "餐前", "餐后",
            ))
        if intent == "side_effect":
            return any(x in clause for x in (
                "副作用", "不良反应", "风险", "可能导致", "常见", "低血糖", "胃肠道",
            ))
        if intent == "definition":
            return any(x in clause for x in (
                "是指", "定义为", "称为", "指的是", "诊断标准",
                "单位是", "包括",
            ))
        if intent == "safety":
            return any(x in clause for x in (
                "建议", "可以", "可适量", "不宜", "避免", "限制", "禁忌",
                "慎用", "能用", "不能", "需", "应",
            ))
        return bool(clause.strip())

    return any(
        clause_has_slot(clause)
        and _clause_mentions_facet(clause, facets)
        and _clause_matches_target_device(clause, question)
        for clause in clauses
    )


def _risk_level(question: str, intent: str) -> str:
    if any(marker in question for marker in _HIGH_RISK_MARKERS):
        return "high"
    if intent in ("quantity", "procedure", "side_effect") and _DRUG_PATTERN.search(question):
        return "high"
    if intent in ("safety", "side_effect", "quantity", "procedure"):
        return "medium"
    return "low"


def _missing_condition(facets: Sequence[Facet]) -> bool:
    return any(
        facet.role == "condition" and facet.support < EVIDENCE_ENTITY_MIN_SCORE
        for facet in facets
    )


def assess_evidence(
    question: str,
    ranked_evidence: Sequence[Tuple[str, float]],
) -> EvidenceAssessment:
    docs = [str(doc) for doc, _score in ranked_evidence]
    if not ranked_evidence or not docs:
        return EvidenceAssessment(
            decision="refuse",
            reason="没有检索到可用资料",
            intent=classify_intent(question),
            risk_level=_risk_level(question, classify_intent(question)),
            intent_score=0.0,
            mandatory_coverage=0.0,
            optional_coverage=0.0,
            answer_slot_supported=False,
            score=0.0,
        )

    normalized = normalize_query_text(question)
    normalized_question = normalized["normalized_question"]
    intent = classify_intent(normalized_question)
    risk = _risk_level(normalized_question, intent)
    intent_score = max(0.0, min(1.0, float(ranked_evidence[0][1])))

    facets = _extract_facets(normalized_question)
    for facet in facets:
        facet.support, facet.evidence_index = _best_facet_support(facet.name, docs)

    mandatory_coverage = _weighted_geometric_mean(facets) if facets else 0.0
    strongest_entity = max((f.support for f in facets), default=0.0)
    slot_supported = _answer_slot_supported(intent, docs, facets, normalized_question)
    optional = Facet(
        name=_INTENT_SLOT_NAMES.get(intent, "直接答案"),
        role="answer_slot",
        support=1.0 if slot_supported else 0.0,
        weight=1.0,
    )

    missing = [
        {
            "facet": facet.name,
            "role": facet.role,
            "support": round(facet.support, 4),
        }
        for facet in facets
        if facet.support < EVIDENCE_ENTITY_MIN_SCORE
    ]
    if not slot_supported:
        missing.append({
            "facet": optional.name,
            "role": "answer_slot",
            "support": 0.0,
        })


    score = (
        0.45 * intent_score
        + 0.45 * mandatory_coverage
        + 0.10 * (1.0 if slot_supported else 0.0)
    )

    def build(decision: str, reason: str) -> EvidenceAssessment:
        return EvidenceAssessment(
            decision=decision,
            reason=reason,
            intent=intent,
            risk_level=risk,
            intent_score=round(intent_score, 4),
            mandatory_coverage=round(mandatory_coverage, 4),
            optional_coverage=1.0 if slot_supported else 0.0,
            answer_slot_supported=slot_supported,
            score=round(score, 4),
            mandatory_facets=facets,
            optional_facets=[optional],
            missing_facets=missing,
        )

    if intent_score < EVIDENCE_INTENT_MIN_SCORE:
        return build("refuse", "整体意图匹配不足，资料虽然可能字面相关，但不能回答当前问题")
    if not facets or strongest_entity < EVIDENCE_ENTITY_MIN_SCORE:
        return build("refuse", "没有找到问题中的核心实体，继续回答会造成意图漂移")
    if _missing_condition(facets) and risk == "high":
        return build("refuse", "问题涉及高风险人群，但资料没有覆盖该人群")
    if risk == "high" and not slot_supported:
        return build("refuse", "问题涉及用药或具体操作，缺少足以安全回答的关键证据")
    principal_quantity_gap = (
        intent == "quantity"
        and not slot_supported
        and any(marker in normalized_question for marker in _PRINCIPAL_QUANTITY_MARKERS)
    )
    if principal_quantity_gap:
        return build("refuse", "问题的主答案就是具体时间或数值，但资料没有在同一语境中提供")
    if not slot_supported and intent in ("procedure", "comparison", "side_effect", "formula"):
        return build("refuse", "资料没有提供问题要求的步骤、比较、不良反应或计算公式")

    if (
        intent_score >= EVIDENCE_INTENT_FULL_SCORE
        and mandatory_coverage >= EVIDENCE_COVERAGE_FULL_SCORE
        and slot_supported
    ):
        return build("answer", "整体意图、核心分面和答案类型均有证据支持")

    if mandatory_coverage >= EVIDENCE_COVERAGE_PARTIAL_SCORE:
        return build(
            "partial",
            "已找到核心相关信息，但部分信息缺失；只能回答有证据支持的部分",
        )

    return build("refuse", "核心分面覆盖不足，不能仅凭少量关键词作答")
