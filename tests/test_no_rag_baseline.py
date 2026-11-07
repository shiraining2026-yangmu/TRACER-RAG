import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_questionbank_no_rag import VERSION, messages, resume_rows


class NoRAGTests(unittest.TestCase):
    def test_only_question_options_enter_prompt(self):
        row = {"question": "题干：阴阳为何？\n选项：\nA.甲\nB.乙", "retrieval_query": "DO_NOT_SEND_QUERY_METADATA",
               "evidence": [{"text": "DO_NOT_SEND_EVIDENCE"}], "answer": "DO_NOT_SEND_GOLD",
               "critique": "DO_NOT_SEND_CRITIQUE", "source": "DO_NOT_SEND_SOURCE"}
        prompt = messages(row)
        self.assertEqual(["system", "user"], [r["role"] for r in prompt])
        self.assertEqual("题目：\n" + row["question"], prompt[1]["content"])
        self.assertNotIn("DO_NOT_SEND", json.dumps(prompt))

    def test_resume_cannot_skip_failures_or_changed_questions(self):
        frozen = [{"application_id": "q1", "question": "题目1"}]
        good = {**frozen[0], "status": "ok", "method": VERSION, "answer": "答案：B"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            path.write_text(json.dumps(good) + "\n", encoding="utf-8")
            self.assertEqual({"q1"}, resume_rows(path, frozen)[1])
            for change in [{"status": "failed"}, {"question": "其他题目"}, {"answer": "无法回答"}]:
                path.write_text(json.dumps({**good, **change}) + "\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    resume_rows(path, frozen)
            path.write_text((json.dumps(good) + "\n") * 2, encoding="utf-8")
            with self.assertRaises(ValueError):
                resume_rows(path, frozen)


if __name__ == "__main__":
    unittest.main()
