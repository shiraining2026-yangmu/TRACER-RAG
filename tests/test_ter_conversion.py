import unittest
from prepare_ter_corpus import prepare


class ConversionTests(unittest.TestCase):
    def fixtures(self):
        apps = [{"application_id": a, "question": "Symbol " + a, "answer": "Synthetic"} for a in ("B", "D", "T")]
        knowledge = [{"knowledge_id": "K", "text": "Synthetic symbol rule"}]
        edges = [{"edge_id": "E" + a, "application_id": a, "knowledge_id": "K", "relation": "explanatory_support"} for a in ("B", "D", "T")]
        splits = [{"application_id": a, "split": s} for a, s in zip(("B", "D", "T"), ("build", "dev", "test"))]
        return apps, knowledge, edges, splits

    def test_build_index_excludes_held_out_applications(self):
        apps, _, edges, queries = prepare(*self.fixtures())
        self.assertEqual([e["application_id"] for e in edges], ["B"])
        self.assertEqual(edges[0]["e_id"], "EB")
        self.assertEqual([a["release"]["split"] for a in apps], ["build", "dev", "test"])
        self.assertTrue(all("answer" not in q for q in queries))
        self.assertEqual(queries[-1]["evaluation_knowledge_ids"], ["K"])

    def test_cross_split_question_duplicate_rejected(self):
        apps, knowledge, edges, splits = self.fixtures()
        apps[1]["question"] = apps[0]["question"] + "!"
        with self.assertRaises(ValueError):
            prepare(apps, knowledge, edges, splits)

    def test_unresolved_link_or_missing_split_rejected(self):
        apps, knowledge, edges, splits = self.fixtures()
        with self.assertRaises(ValueError):
            prepare(apps, knowledge, edges, splits[:-1])
        edges[0]["knowledge_id"] = "absent"
        with self.assertRaises(ValueError):
            prepare(apps, knowledge, edges, splits)
