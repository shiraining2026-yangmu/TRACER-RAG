import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_dense_chunk_rag import ContaminationGuard, SourcePool, chunks, contract, exact_top3, messages, questions


class DenseRAGTests(unittest.TestCase):
    def test_source_recovery_requires_all_fragments_and_question(self):
        pool = SourcePool([
            {"source": {"record_hash": "s1", "dataset": "text_cleaned"}, "text": "脾胃虚寒的核心病机是中焦阳气不足"},
            {"source": {"record_hash": "s1", "dataset": "text_cleaned"}, "text": "治疗原则为温中散寒健脾益气"},
        ], [{"source": {"record_hash": "s1"}, "question": "胃寒如何治疗？"}])
        answer = "脾胃虚寒的核心病机是：中焦阳气不足。治疗原则为温中散寒、健脾益气。"
        self.assertEqual({"s1"}, pool.match("text_cleaned", "胃寒如何治疗", answer))
        self.assertEqual(set(), pool.match("text_cleaned", "另一个问题", answer))
        self.assertEqual(set(), pool.match("vision_cleaned", "胃寒如何治疗", answer))
        self.assertEqual(set(), pool.match("text_cleaned", "胃寒如何治疗", "脾胃虚寒的核心病机是中焦阳气不足"))

    def test_chunks_cover_original_with_exact_overlap(self):
        text = "甲乙丙丁戊己庚辛壬癸" * 153
        parts = list(chunks(text, 700, 100))
        restored = parts[0][2] + "".join(p[2][100:] for p in parts[1:])
        self.assertEqual(text, restored)
        self.assertTrue(all(len(p[2]) <= 700 for p in parts))
        self.assertEqual([], list(chunks("")))
        with self.assertRaises(ValueError):
            list(chunks(text, 100, 100))

    def test_blocked_search_matches_full_sort_and_ties(self):
        rng = np.random.default_rng(42)
        docs = rng.normal(size=(53, 8)).astype("float32")
        docs /= np.linalg.norm(docs, axis=1, keepdims=True)
        docs[25] = docs[2]
        query = np.stack([docs[2], docs[20]])
        positions, scores = exact_top3(docs, query, block_size=7)
        expected = np.argsort(-(docs @ query.T), axis=0, kind="stable")[:3].T
        np.testing.assert_array_equal(positions, expected)
        np.testing.assert_allclose(scores, np.take_along_axis((docs @ query.T).T, expected, 1), atol=1e-6)
        self.assertEqual([2, 25], positions[0, :2].tolist())

    def test_exam_overlap_screen_without_answers(self):
        stem = "患者咳嗽十余日伴有口干舌红少苔应采用何种治疗方法同时可见午后潮热盗汗五心烦热两颧发红夜间症状较重脉象细数"
        guard = ContaminationGuard([stem])
        self.assertIsNotNone(guard.match("请问，" + stem + "？"))
        self.assertIsNotNone(guard.match(stem.replace("十余", "十多")))
        self.assertIsNone(guard.match("阴阳学说的基本概念"))

    def test_contract_rejects_changed_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "contract.json"
            contract(path, {"top_k": 3})
            contract(path, {"top_k": 3})
            with self.assertRaises(ValueError):
                contract(path, {"top_k": 5})

    def test_answer_fields_and_option_retrieval_are_rejected(self):
        row = {"application_id": "Q1", "question": "题干：何为阴阳？\n选项：\nA.甲\nB.乙", "retrieval_query": "何为阴阳？"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "questions.jsonl"
            def write(value):
                path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
            write(row)
            self.assertEqual(1, len(questions([path])[0][1]))
            write({**row, "答案": "A"})
            with self.assertRaises(ValueError):
                questions([path])
            write({**row, "retrieval_query": row["question"]})
            with self.assertRaises(ValueError):
                questions([path])

    def test_prompt_uses_only_question_and_chunk_text(self):
        row = {"question": "题目及选项", "evidence": [{"doc_id": "D1", "text": "中医资料", "source": {"secret": "must_not_appear"}}], "answer": "official_answer_must_not_appear"}
        prompt = json.dumps(messages(row), ensure_ascii=False)
        self.assertIn("中医资料", prompt)
        self.assertNotIn("must_not_appear", prompt)
        self.assertIn("A、B、C、D、E", prompt)


if __name__ == "__main__":
    unittest.main()
