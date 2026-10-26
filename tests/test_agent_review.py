import unittest
from types import SimpleNamespace

from agent_review import (
    AgentReviewError,
    AnswerAuditorAgent,
    EvidenceJudgeAgent,
    apply_answer_audit,
    evidence_review_needed,
    merge_evidence_decision,
    validate_answer_citations,
)


class FakeModel:
    def __init__(self, content):
        self.content = content

    def invoke(self, _messages):
        return SimpleNamespace(content=self.content)


class EvidenceJudgeTest(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            {
                "evidence_id": 1,
                "source": "指南.docx",
                "text": "糖尿病患者可以适量食用西瓜，但应控制水果总量。",
            }
        ]
        self.assessment = {
            "decision": "partial",
            "risk_level": "medium",
            "intent_score": 0.82,
            "mandatory_coverage": 0.5,
        }

    def test_valid_quote_is_accepted(self):
        model = FakeModel(
            """
            {
              "decision": "partial",
              "reason": "缺少具体克数",
              "supported_facets": [{
                "facet": "食用建议",
                "evidence_id": 1,
                "quote": "糖尿病患者可以适量食用西瓜"
              }],
              "missing_facets": [{"facet": "具体摄入量", "reason": "资料没有克数"}],
              "selected_evidence_ids": [1]
            }
            """
        )
        result = EvidenceJudgeAgent(model).review(
            "糖尿病患者能吃多少西瓜？",
            self.assessment,
            self.candidates,
        )
        self.assertEqual(result["decision"], "partial")
        self.assertEqual(result["selected_evidence_ids"], [1])

    def test_invalid_quote_is_rejected(self):
        model = FakeModel(
            """
            {
              "decision": "answer",
              "reason": "资料完整",
              "supported_facets": [{
                "facet": "食用建议",
                "evidence_id": 1,
                "quote": "每天最多吃一百克西瓜"
              }],
              "selected_evidence_ids": [1]
            }
            """
        )
        with self.assertRaises(AgentReviewError):
            EvidenceJudgeAgent(model).review(
                "糖尿病患者能吃多少西瓜？",
                self.assessment,
                self.candidates,
            )

    def test_agent_cannot_upgrade_answer(self):
        self.assertEqual(
            merge_evidence_decision("answer", "partial", "high"),
            "partial",
        )
        self.assertEqual(
            merge_evidence_decision("partial", "answer", "medium"),
            "partial",
        )
        self.assertEqual(
            merge_evidence_decision("refuse", "answer", "medium"),
            "partial",
        )
        self.assertEqual(
            merge_evidence_decision("refuse", "answer", "high"),
            "refuse",
        )

    def test_uncertain_mode_reviews_partial(self):
        self.assertTrue(evidence_review_needed(self.assessment, "uncertain"))
        self.assertFalse(
            evidence_review_needed(
                {
                    "decision": "answer",
                    "risk_level": "low",
                    "intent_score": 0.95,
                    "mandatory_coverage": 0.90,
                },
                "uncertain",
            )
        )


class AnswerAuditorTest(unittest.TestCase):
    def test_valid_citation_is_accepted(self):
        result = validate_answer_citations("建议适量食用【资料1】", 1)
        self.assertTrue(result["valid"])
        self.assertEqual(result["cited_ids"], [1])

    def test_invalid_citation_number_is_rejected(self):
        result = validate_answer_citations("建议如下【资料3】", 2)
        self.assertFalse(result["valid"])

    def test_answer_without_citation_is_rejected(self):
        result = validate_answer_citations("建议适量食用", 1)
        self.assertFalse(result["valid"])

    def test_agent_pass_is_downgraded_when_citation_missing(self):
        model = FakeModel(
            '{"action":"pass","reason":"无问题","unsupported_claims":[]}'
        )
        audit = AnswerAuditorAgent(model).review(
            "糖尿病患者能吃西瓜吗？",
            "可以适量食用，但应控制总量。",
            "answer",
            [{
                "evidence_id": 1,
                "source": "指南.docx",
                "text": "糖尿病患者可以适量食用西瓜，但应控制水果总量。",
            }],
        )
        self.assertEqual(audit["action"], "downgrade")
        self.assertFalse(audit["citation_check"]["valid"])

    def test_unverified_claim_is_removed_and_downgraded(self):
        answer = "可以适量吃西瓜。每天最多吃 100 克。【资料1】"
        model = FakeModel(
            """
            {
              "action": "pass",
              "reason": "克数没有证据",
              "unsupported_claims": [{
                "claim": "每天最多吃 100 克。",
                "evidence_id": 1,
                "reason": "原文没有这个数值"
              }]
            }
            """
        )
        audit = AnswerAuditorAgent(model).review(
            "糖尿病患者能吃多少西瓜？",
            answer,
            "answer",
            [{
                "evidence_id": 1,
                "source": "指南.docx",
                "text": "糖尿病患者可以适量食用西瓜，但应控制水果总量。",
            }],
        )
        audited, decision, refused = apply_answer_audit(answer, "answer", audit)
        self.assertEqual(audit["action"], "downgrade")
        self.assertEqual(decision, "partial")
        self.assertFalse(refused)
        self.assertNotIn("每天最多吃 100 克", audited)
        self.assertIn("缺少原文支持", audited)

    def test_refuse_action_returns_safe_message(self):
        audited, decision, refused = apply_answer_audit(
            "没有依据的回答",
            "answer",
            {"action": "refuse", "reason": "关键结论无依据"},
        )
        self.assertEqual(decision, "refuse")
        self.assertTrue(refused)
        self.assertIn("不足以确认", audited)

    def test_invalid_json_raises_review_error(self):
        with self.assertRaises(AgentReviewError):
            EvidenceJudgeAgent(FakeModel("not json")).review(
                "问题",
                {"decision": "partial"},
                [{
                    "evidence_id": 1,
                    "source": "指南.docx",
                    "text": "存在相关医学资料。",
                }],
            )


if __name__ == "__main__":
    unittest.main()
