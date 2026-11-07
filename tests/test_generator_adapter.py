"""Input and generation contract checks; run in the remote torch environment."""
import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

try:
    import torch
    import transformers
    import fastapi
    import uvicorn
except ImportError as exc:
    raise unittest.SkipTest("Generator deployment tests require the optional model-serving dependencies") from exc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import serve_single_generator as server


class Encoding(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        if any(m["role"] == "system" for m in messages):
            raise ValueError("System role not supported")
        self.messages = messages
        return "<bos>" + messages[0]["content"]

    def encode(self, prompt, *, add_special_tokens):
        if add_special_tokens:
            raise AssertionError("chat template must not get a second BOS")
        return [1] + [3] * len(prompt)

    def pad(self, rows, padding, return_tensors):
        width = max(len(row["input_ids"]) for row in rows)
        self.padded = Encoding(
            input_ids=torch.tensor([[0] * (width - len(row["input_ids"])) + row["input_ids"] for row in rows]),
            attention_mask=torch.tensor([[0] * (width - len(row["input_ids"])) + row["attention_mask"] for row in rows]),
        )
        return self.padded

    def decode(self, ids, **kwargs):
        return "答案：A" if 4 in ids else "答案：B"


class FakeModel:
    generation_config = SimpleNamespace(eos_token_id=[2, 5])

    def generate(self, **kwargs):
        if "eos_token_id" in kwargs:
            raise AssertionError("official EOS configuration was overridden")
        tail = torch.tensor([[4, 5, 0], [6, 6, 6]][:len(kwargs["input_ids"])])
        return torch.cat([kwargs["input_ids"], tail], dim=1)


class GeneratorAdapterTests(unittest.TestCase):
    def setUp(self):
        server.runtime.tokenizer = FakeTokenizer()
        server.runtime.model = FakeModel()
        server.runtime.model_id = "test"
        server.runtime.device = torch.device("cpu")
        server.runtime.metadata = {"context_limit": 1000}

    def request(self, text="q", max_tokens=3):
        return server.ChatCompletionRequest(model="test", messages=[
            server.ChatMessage(role="system", content="system"),
            server.ChatMessage(role="user", content=text)], max_tokens=max_tokens)

    def test_gemma_merge_preserves_system_and_adds_no_second_bos(self):
        ids = server.input_ids(self.request())
        self.assertEqual(1, ids.count(1))
        self.assertEqual([{"role": "user", "content": "system\n\nq"}], server.runtime.tokenizer.messages)

    def test_left_padding_eos_finish_reason_and_real_token_usage(self):
        results = server.render([self.request(), self.request("much longer")])
        self.assertEqual(("stop", 2), (results[0]["finish_reason"], results[0]["completion_tokens"]))
        self.assertEqual(("length", 3), (results[1]["finish_reason"], results[1]["completion_tokens"]))
        padded = server.runtime.tokenizer.padded
        self.assertEqual(0, padded["attention_mask"][0][0].item())
        self.assertEqual(results[0]["prompt_tokens"], padded["attention_mask"][0].sum().item())
        self.assertEqual("stop", server.payload("test", results[0], 0)["choices"][0]["finish_reason"])

    def test_context_overflow_fails_instead_of_truncating(self):
        server.runtime.metadata["context_limit"] = 1
        with self.assertRaisesRegex(ValueError, "silent truncation"):
            server.render([self.request()])


if __name__ == "__main__":
    unittest.main()
