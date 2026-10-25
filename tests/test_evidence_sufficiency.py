
import unittest

from evidence_sufficiency import (
    assess_evidence,
    classify_intent,
    normalize_query_text,
)


class EvidenceSufficiencyTest(unittest.TestCase):
    def test_typo_alias_is_conservative(self):
        result = normalize_query_text("阿卡波堂漏服后怎么办？")
        self.assertEqual(result["normalized_question"], "阿卡波糖漏服后怎么办？")
        self.assertTrue(result["changed"])

    def test_drug_class_is_not_mapped_to_specific_drug(self):
        result = normalize_query_text("降糖药漏服后怎么办？")
        self.assertEqual(result["normalized_question"], "降糖药漏服后怎么办？")
        self.assertFalse(result["changed"])

    def test_complete_numeric_evidence_can_answer(self):
        question = "糖尿病患者空腹血糖目标是多少？"
        evidence = [(
            "糖尿病患者空腹血糖控制目标为 4.4~6.1 mmol/L，"
            "非空腹血糖目标为 10.0 mmol/L 以下。",
            0.95,
        )]
        result = assess_evidence(question, evidence)
        self.assertEqual(classify_intent(question), "quantity")
        self.assertEqual(result.decision, "answer")
        self.assertTrue(result.answer_slot_supported)

    def test_missing_quantity_returns_partial_answer(self):
        question = "糖尿病患者能吃多少西瓜？"
        evidence = [(
            "糖尿病患者可以适量食用西瓜，但应控制水果总量，并结合血糖监测。",
            0.82,
        )]
        result = assess_evidence(question, evidence)
        self.assertEqual(result.decision, "partial")
        self.assertEqual(result.risk_level, "medium")
        self.assertTrue(any(
            item["facet"] == "具体数值或时间" for item in result.missing_facets
        ))

    def test_high_risk_dose_is_not_inferred(self):
        question = "二甲双胍漏服后补多少片？"
        evidence = [(
            "二甲双胍应随餐服用。漏服时请咨询医生或药师，不要自行加倍。",
            0.91,
        )]
        result = assess_evidence(question, evidence)
        self.assertEqual(result.risk_level, "high")
        self.assertEqual(result.decision, "refuse")

    def test_high_risk_population_is_not_routed_to_general_answer(self):
        question = "孕妇糖尿病患者能吃西瓜吗？"
        evidence = [(
            "糖尿病患者可以适量食用西瓜，注意控制总量。",
            0.86,
        )]
        result = assess_evidence(question, evidence)
        self.assertEqual(result.decision, "refuse")
        self.assertIn("高风险人群", result.reason)

    def test_two_strong_facets_and_one_missing_can_be_partial(self):
        question = "老年糖尿病患者能吃西瓜吗？"
        evidence = [(
            "糖尿病患者可以适量食用西瓜，并应根据血糖情况控制水果总量。",
            0.84,
        )]
        result = assess_evidence(question, evidence)
        self.assertEqual(result.decision, "partial")
        self.assertTrue(any(item["facet"] == "老年" for item in result.missing_facets))

    def test_unrelated_procedure_is_refused(self):
        question = "红烧肉怎么做？"
        evidence = [(
            "糖尿病患者应限制红肉和饱和脂肪摄入，饮食以清淡为主。",
            0.50,
        )]
        result = assess_evidence(question, evidence)
        self.assertEqual(result.decision, "refuse")

    def test_comparison_table_without_comparison_words_can_answer(self):
        question = "1 型和 2 型糖尿病有什么区别？"
        evidence = [(
            "1 型糖尿病以胰岛素绝对缺乏为主；"
            "2 型糖尿病以胰岛素抵抗为主，常伴相对胰岛素不足。",
            0.90,
        )]
        result = assess_evidence(question, evidence)
        self.assertEqual(result.intent, "comparison")
        self.assertEqual(result.decision, "answer")

    def test_comparison_requires_both_objects(self):
        question = "1 型和 2 型糖尿病有什么区别？"
        evidence = [(
            "2 型糖尿病以胰岛素抵抗为主，常伴相对胰岛素不足。",
            0.88,
        )]
        result = assess_evidence(question, evidence)
        self.assertEqual(result.decision, "refuse")

    def test_pregnancy_alias_supports_condition_facet(self):
        question = "孕妇糖尿病患者能吃西瓜吗？"
        evidence = [(
            "妊娠期糖尿病患者可以适量食用西瓜，但应根据血糖情况控制总量。",
            0.86,
        )]
        result = assess_evidence(question, evidence)
        self.assertEqual(result.decision, "answer")

    def test_cgm_calibration_cannot_answer_glucose_meter_interval(self):
        question = "血糖仪多久校准一次？"
        evidence = [(
            "CGM 期间每日 1~4 次毛细血管血糖监测用于校准；"
            "使用同一台血糖仪及同一批试纸。",
            0.92,
        )]
        result = assess_evidence(question, evidence)
        self.assertEqual(result.decision, "refuse")
        self.assertFalse(result.answer_slot_supported)

    def test_cgm_and_meter_in_one_sentence_still_do_not_share_interval(self):
        question = "血糖仪多久校准一次？"
        evidence = [(
            "CGM 系统要求每日至少进行 1~4 次毛细血管血糖监测以进行校准，"
            "应使用同一台血糖仪及同一批试纸。",
            0.91,
        )]
        result = assess_evidence(question, evidence)
        self.assertEqual(result.decision, "refuse")

    def test_formula_question_can_answer_without_procedure_words(self):
        question = "BMI 怎么算？"
        evidence = [(
            "BMI = 体重(kg) / 身高(m) 的平方。",
            0.88,
        )]
        result = assess_evidence(question, evidence)
        self.assertEqual(result.intent, "formula")
        self.assertEqual(result.decision, "answer")

    def test_annual_calibration_frequency_can_answer(self):
        question = "血糖仪校准周期是多久？"
        evidence = [(
            "血糖仪应每年校准一次，并使用同一批试纸。",
            0.91,
        )]
        result = assess_evidence(question, evidence)
        self.assertEqual(result.decision, "answer")

    def test_daily_frequency_can_answer(self):
        question = "每天测几次血糖？"
        evidence = [(
            "建议每天测一次空腹血糖，并根据医嘱调整监测频率。",
            0.89,
        )]
        result = assess_evidence(question, evidence)
        self.assertEqual(result.decision, "answer")

    def test_diagnostic_standard_definition_can_answer(self):
        question = "妊娠期糖尿病的诊断标准是什么？"
        evidence = [(
            "妊娠期糖尿病诊断标准：空腹血糖≥5.1 mmol/L。",
            0.93,
        )]
        result = assess_evidence(question, evidence)
        self.assertEqual(result.intent, "definition")
        self.assertEqual(result.decision, "answer")

    def test_unit_definition_is_not_treated_as_high_risk(self):
        question = "血糖单位是什么？"
        evidence = [(
            "血糖单位是 mmol/L。",
            0.90,
        )]
        result = assess_evidence(question, evidence)
        self.assertNotEqual(result.risk_level, "high")
        self.assertEqual(result.decision, "answer")

    def test_drug_comparison_extracts_both_objects(self):
        question = "二甲双胍和格列齐特哪个好？"
        evidence = [(
            "二甲双胍常见胃肠道反应；格列齐特可能增加低血糖风险。",
            0.88,
        )]
        result = assess_evidence(question, evidence)
        self.assertEqual(result.intent, "comparison")
        self.assertEqual(result.decision, "answer")
        names = {facet.name for facet in result.mandatory_facets}
        self.assertIn("二甲双胍", names)
        self.assertIn("格列齐特", names)

    def test_multi_drug_question_missing_one_drug_is_partial(self):
        question = (
            "二甲双胍、格列齐特、SGLT-2 抑制剂分别适合哪类糖尿病患者？"
        )
        evidence = [(
            "二甲双胍适合部分 2 型糖尿病患者；SGLT-2 抑制剂有心血管保护证据。",
            0.90,
        )]
        result = assess_evidence(question, evidence)
        self.assertEqual(result.decision, "partial")
        names = {facet.name for facet in result.mandatory_facets}
        self.assertIn("二甲双胍", names)
        self.assertIn("格列齐特", names)
        self.assertIn("SGLT-2", names)

    def test_table_completion_evidence_can_rescue_comparison(self):
        question = "1 型和 2 型糖尿病有什么区别？"
        one_row = "1 型糖尿病以胰岛素绝对缺乏为主。"
        completed_table = (
            "1 型糖尿病以胰岛素绝对缺乏为主；"
            "2 型糖尿病以胰岛素抵抗为主。"
        )
        first = assess_evidence(question, [(one_row, 0.91)])
        second = assess_evidence(question, [(completed_table, 0.91)])
        self.assertEqual(first.decision, "refuse")
        self.assertEqual(second.decision, "answer")


if __name__ == "__main__":
    unittest.main()
