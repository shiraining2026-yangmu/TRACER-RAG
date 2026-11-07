import sys
from pathlib import Path
import contextlib
import io
import json
from argparse import Namespace
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from screen_medical_qa import screen, screen_archive


class ScreeningTests(unittest.TestCase):
    def test_ambiguous_general_medical_words_do_not_establish_domain(self):
        for q, a in [
            ("麦粒肿为什么要湿热敷？", "湿热敷可以使眼部局部温度升高，帮助缓解疼痛，但应避免温度过高，并根据病情及时就诊。"),
            ("我头痛像针刺一样是什么病因？", "头痛可能涉及多种原因，建议咨询主治医师，了解病因并安排检查，不要仅凭疼痛性质自行判断。"),
        ]:
            tier, reasons, _ = screen(q, a)
            self.assertEqual("excluded", tier)
            self.assertIn("no_specific_tcm_subject_anchor", reasons)

    def test_reusable_concept_is_candidate_not_verified_truth(self):
        tier, flags, _ = screen("气虚与血虚有什么区别？", "气虚与血虚是不同的证候。气虚主要反映气的功能不足，常见少气懒言、乏力等表现；血虚主要反映血的濡养不足，可见面色淡白、头晕等表现。辨证应结合四诊信息。")
        self.assertEqual("priority_candidate", tier)
        self.assertEqual([], flags)

    def test_specific_dose_and_patient_are_not_priority(self):
        tier, flags, _ = screen("我最近气虚，可以吃黄芪吗？", "气虚可能出现乏力、少气懒言等表现。黄芪具有补气等作用，但是否适合需要辨证，使用时应核实相关禁忌，不应仅凭单个症状用药，每次10克。")
        self.assertEqual("review_required", tier)
        self.assertIn("individual_patient_context", flags)
        self.assertIn("dose_or_procedure_quantity_requires_reference", flags)

    def test_real_consultations_and_ambiguous_conditions_are_not_priority(self):
        examples = [
            ("颈椎病和气血虚到底有什么区别,每天浑身没劲,每天晚上都会醒。", "颈椎病与气血虚属于不同概念。气血不足可能涉及多种表现，具体辨证应结合病史与四诊信息，不能仅凭乏力和失眠作出结论。"),
            ("黄疸,尿液深黄,这是什么病情,是不是肝炎", "黄疸和尿液深黄可以由多种原因引起，不能单凭这两个表现判断病因，需要结合病史、体征和相关检查进行诊断。"),
        ]
        for q, a in examples:
            tier, _, _ = screen(q, a)
            self.assertNotEqual("priority_candidate", tier)

    def test_generic_advice_with_incidental_acupuncture_is_excluded(self):
        tier, flags, _ = screen("腰椎间盘突出应当怎么办？", "腰椎间盘突出可能导致腰痛。一般需结合病情选择休息、物理治疗、针灸等方法，不应仅凭这段描述自行处理，建议到医院咨询医生。")
        self.assertEqual("excluded", tier)
        self.assertIn("no_specific_tcm_subject_anchor", flags)

    def test_observed_bad_formula_is_excluded(self):
        tier, flags, _ = screen("双黄连的组成是什么？", "双黄连是一种常见中药制剂，主要成分包括黄连、黄芩、连翘等。用药时应注意适应证和禁忌，不能自行长期服用。")
        self.assertEqual("excluded", tier)
        self.assertIn("observed_shuanghuanglian_component_conflict", flags)

    def test_formula_list_requires_reference_even_if_plausible(self):
        tier, flags, _ = screen("四君子汤的组成是什么？", "四君子汤的组成包括人参、白术、茯苓、甘草。它是补气的代表方之一，涉及具体使用时应结合证候与原始文献核验，不能直接将历史用法套用于个人。")
        self.assertEqual("review_required", tier)
        self.assertIn("formula_composition_requires_reference", flags)

    def test_archive_export_checks_conflicts_and_answer_side_exam_overlap(self):
        q = "气虚与血虚有什么区别？"
        a = "气虚与血虚是不同的证候。气虚主要反映气的功能不足，常见少气懒言、乏力等表现；血虚主要反映血的濡养不足，可见面色淡白、头晕等表现。辨证应结合四诊信息。"
        stem = "患者咳嗽十余日伴口干舌红少苔午后潮热盗汗五心烦热脉象细数"
        rows = [
            {"input": q, "output": a},
            {"input": q, "output": a.replace("乏力", "气短")},
            {"input": q, "output": a},
            {"input": "气虚和阴虚有什么区别？", "output": a + stem},
            {"input": "", "output": a},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, exam, out = root / "source.zip", root / "exam.jsonl", root / "output"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("medicalQA.json", "\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
            exam.write_text(json.dumps({"application_id": "Q1", "question_id": "Q1", "question": stem,
                                        "retrieval_query": stem}, ensure_ascii=False) + "\n", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                screen_archive(Namespace(output_dir=out, exam_questions=[exam], archive=archive, member="medicalQA.json"))
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(5, summary["counts"]["source_records"])
            self.assertEqual(0, summary["priority_candidates"])
            self.assertEqual(2, summary["review_required"])
            self.assertEqual(1, summary["duplicates_removed"])
            self.assertEqual(1, summary["initial_reason_counts"]["frozen_exam_stem_overlap"])
            retained = [json.loads(line) for line in (out / "review_required.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all("same_question_different_answers" in r["review_flags"] for r in retained))
            self.assertTrue(all(not r["eligible_for_index"] for r in retained))


if __name__ == "__main__":
    unittest.main()
