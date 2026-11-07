#!/usr/bin/env python3
"""Serve one 4-bit base generator through a compact OpenAI-compatible API.

The benchmark deliberately keeps this service separate from the frozen
Qwen3.5+LoRA Critic.  This makes cross-family generation possible without
attempting to load a Qwen adapter into an incompatible Llama/InternLM model.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=64, ge=1, le=1024)
    stream: bool = False


class BatchRequest(BaseModel):
    requests: list[ChatCompletionRequest] = Field(min_length=1, max_length=4)


class Runtime:
    tokenizer: Any
    model: Any
    device: torch.device
    model_id: str
    lock: asyncio.Lock
    metadata: dict[str, Any]


runtime = Runtime()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def conversation(request: ChatCompletionRequest) -> list[dict[str, str]]:
    return [{"role": item.role, "content": item.content} for item in request.messages]


def flattened_system_conversation(request: ChatCompletionRequest) -> list[dict[str, str]]:
    """Convert a leading system turn for templates such as Gemma-2."""
    messages = conversation(request)
    if messages and messages[0]["role"] == "system":
        if len(messages) < 2 or messages[1]["role"] != "user":
            raise ValueError("A leading system message must be followed by a user message")
        messages = [{"role": "user", "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}] + messages[2:]
    return messages


def render_chat_prompt(request: ChatCompletionRequest) -> str:
    # Preserve each model's official template, including Qwen3 non-thinking mode.
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True,
                              "enable_thinking": False}
    try:
        return runtime.tokenizer.apply_chat_template(conversation(request), **kwargs)
    except Exception as exc:
        if "system role not supported" not in str(exc).lower():
            raise
        return runtime.tokenizer.apply_chat_template(flattened_system_conversation(request), **kwargs)


def input_ids(request: ChatCompletionRequest) -> list[int]:
    prompt = render_chat_prompt(request)
    # The official chat template already supplies its own special tokens.
    return runtime.tokenizer.encode(prompt, add_special_tokens=False)


@torch.inference_mode()
def render(requests: list[ChatCompletionRequest]) -> list[dict[str, Any]]:
    if not requests:
        raise ValueError("Empty batch")
    if any(request.model != runtime.model_id for request in requests):
        raise ValueError("Unknown model id")
    if len({request.max_tokens for request in requests}) != 1:
        raise ValueError("Batch must use a single max_tokens value")
    if any(request.temperature != 0 for request in requests):
        raise ValueError("This benchmark uses greedy decoding (temperature=0)")
    tokens = [input_ids(request) for request in requests]
    limit = runtime.metadata["context_limit"]
    if any(len(ids) + request.max_tokens > limit for ids, request in zip(tokens, requests)):
        raise ValueError("Prompt plus output exceeds model context; silent truncation is forbidden")
    inputs = runtime.tokenizer.pad(
        [{"input_ids": ids, "attention_mask": [1] * len(ids)} for ids in tokens],
        padding=True, return_tensors="pt",
    ).to(runtime.device)
    generated = runtime.model.generate(
        **inputs, max_new_tokens=requests[0].max_tokens, do_sample=False, num_beams=1, use_cache=True,
        # Preserve generation_config's model-specific EOS list (e.g. Llama's
        # end-of-turn token), rather than replacing it with one tokenizer EOS.
        pad_token_id=runtime.tokenizer.pad_token_id,
    )
    width = inputs["input_ids"].shape[1]
    eos = runtime.model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos or [])
    result = []
    for ids, prompt_ids, request in zip(generated[:, width:].tolist(), tokens, requests):
        stop = next((i for i, token in enumerate(ids) if token in eos), None)
        used = ids if stop is None else ids[:stop + 1]
        content = runtime.tokenizer.decode(used, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        result.append({"content": content, "prompt_tokens": len(prompt_ids),
                       "input_ids_sha256": hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest(),
                       "completion_tokens": len(used), "finish_reason": "stop" if stop is not None else "length"})
    if any(not item["content"] for item in result):
        raise RuntimeError("Empty model completion")
    print(json.dumps({"event": "generation_audit", "model": runtime.model_id,
                      "batch_size": len(requests), "max_new_tokens": requests[0].max_tokens,
                      "usage": [{k: v for k, v in item.items() if k != "content"} for item in result]}), flush=True)
    return result


def payload(model: str, completion: dict[str, Any], started: float) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
        "created": int(started), "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": completion["content"]},
                     "finish_reason": completion["finish_reason"]}],
        "usage": {"prompt_tokens": completion["prompt_tokens"], "completion_tokens": completion["completion_tokens"],
                  "total_tokens": completion["prompt_tokens"] + completion["completion_tokens"]},
        "input_adapter_version": "official-template-no-extra-special-tokens-v2",
        "input_ids_sha256": completion["input_ids_sha256"],
    }


def build_app(args: argparse.Namespace) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        quantization = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        runtime.tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, local_files_only=True, trust_remote_code=args.trust_remote_code
        )
        if runtime.tokenizer.pad_token_id is None:
            runtime.tokenizer.pad_token = runtime.tokenizer.eos_token
        runtime.tokenizer.padding_side = "left"
        config = AutoConfig.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=args.trust_remote_code)
        # The installed SDPA backend ignores Gemma-2's softcap argument. Eager
        # implements both attention softcapping and its sliding-window masks.
        attention_options = {"attn_implementation": "eager"} if config.model_type == "gemma2" else {}
        runtime.model = AutoModelForCausalLM.from_pretrained(
            args.model_path, local_files_only=True, trust_remote_code=args.trust_remote_code,
            torch_dtype=torch.bfloat16, device_map="auto", quantization_config=quantization,
            **attention_options,
        ).eval()
        device_map = getattr(runtime.model, "hf_device_map", {})
        if any(str(device) in {"cpu", "disk"} for device in device_map.values()):
            raise ValueError("Unexpected CPU/disk offload; candidate must fit GPU before benchmarking")
        if runtime.model.generation_config.eos_token_id is None:
            runtime.model.generation_config.eos_token_id = runtime.tokenizer.eos_token_id
        if runtime.model.generation_config.eos_token_id is None:
            raise ValueError("Model has no end-of-turn/end-of-sequence token")
        runtime.device = next(runtime.model.parameters()).device
        runtime.model_id = args.model_id
        runtime.lock = asyncio.Lock()
        runtime.metadata = {
            "model_id": args.model_id, "model_type": runtime.model.config.model_type,
            "model_path": args.model_path,
            "context_limit": int(runtime.model.config.max_position_embeddings),
            "input_adapter_version": "official-template-no-extra-special-tokens-v2",
            "eos_token_id": runtime.model.generation_config.eos_token_id,
            "pad_token_id": runtime.tokenizer.pad_token_id, "padding_side": runtime.tokenizer.padding_side,
            "device_map": {key: str(value) for key, value in device_map.items()},
            "quantization": "NF4 4-bit double quantization; bfloat16 compute",
            "tokenizer_class": type(runtime.tokenizer).__name__,
            "attention_implementation": runtime.model.config._attn_implementation,
            "attn_logit_softcapping": getattr(runtime.model.config, "attn_logit_softcapping", None),
            "final_logit_softcapping": getattr(runtime.model.config, "final_logit_softcapping", None),
            "service_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
            "transformers_version": __import__("transformers").__version__,
            "torch_version": torch.__version__,
        }
        yield
        del runtime.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    app = FastAPI(lifespan=lifespan)

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": runtime.model_id, "object": "model", "owned_by": "local"}]}

    @app.get("/v1/benchmark_runtime")
    async def benchmark_runtime() -> dict[str, Any]:
        return runtime.metadata

    @app.post("/v1/chat/completions")
    async def chat(request: ChatCompletionRequest) -> dict[str, Any]:
        if request.stream:
            raise HTTPException(status_code=400, detail="Streaming is not supported")
        async with runtime.lock:
            started = time.time()
            try:
                content = (await asyncio.to_thread(render, [request]))[0]
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
        return payload(request.model, content, started)

    @app.post("/v1/chat/completions/batch")
    async def batch(request: BatchRequest) -> dict[str, Any]:
        if any(item.stream for item in request.requests):
            raise HTTPException(status_code=400, detail="Streaming is not supported")
        async with runtime.lock:
            started = time.time()
            try:
                contents = await asyncio.to_thread(render, request.requests)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"object": "list", "data": [payload(item.model, value, started) for item, value in zip(request.requests, contents)]}

    return app


def main() -> None:
    args = parse_args()
    uvicorn.run(build_app(args), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
