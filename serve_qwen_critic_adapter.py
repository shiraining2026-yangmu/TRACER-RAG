#!/usr/bin/env python3
"""Expose a QLoRA Critic adapter through a minimal OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=700, ge=1, le=4096)
    stream: bool = False


class ChatCompletionBatchRequest(BaseModel):
    requests: list[ChatCompletionRequest] = Field(min_length=1, max_length=4)


class Runtime:
    processor: Any
    model: Any
    device: torch.device
    lock: asyncio.Lock
    generator_model_id: str
    critic_model_id: str


runtime = Runtime()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--generator-model-id", required=True)
    parser.add_argument("--critic-model-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    return parser.parse_args()


def to_conversation(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    return [{"role": message.role, "content": message.content} for message in messages]


def render_completions(requests: list[ChatCompletionRequest]) -> list[str]:
    if not requests:
        raise ValueError("Batch must contain at least one request")
    model_ids = {request.model for request in requests}
    if len(model_ids) != 1:
        raise ValueError("Each batch must target exactly one model id")
    model_id = next(iter(model_ids))
    if model_id not in {runtime.generator_model_id, runtime.critic_model_id}:
        raise ValueError(f"Unknown model id: {model_id}")
    max_tokens = {request.max_tokens for request in requests}
    if len(max_tokens) != 1:
        raise ValueError("Each batch must use one max_tokens value")
    prompts = [
        runtime.processor.apply_chat_template(
            to_conversation(request.messages),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for request in requests
    ]
    inputs = runtime.processor(text=prompts, return_tensors="pt", padding=True).to(runtime.device)
    generation_kwargs = {**inputs, "max_new_tokens": next(iter(max_tokens)), "do_sample": False, "use_cache": True}
    if model_id == runtime.generator_model_id:
        # The generator and Critic share one quantized base model on the single GPU.
        with runtime.model.disable_adapter():
            generated = runtime.model.generate(**generation_kwargs)
    else:
        generated = runtime.model.generate(**generation_kwargs)
    prompt_width = inputs["input_ids"].shape[1]
    texts = [
        text.strip()
        for text in runtime.processor.batch_decode(
            generated[:, prompt_width:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    ]
    if any(not text for text in texts):
        raise RuntimeError("Model returned an empty completion")
    return texts


def completion_payload(model: str, content: str, started: float) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(started),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
    }


def build_app(model_path: str, adapter_path: str, generator_model_id: str, critic_model_id: str) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        runtime.processor = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if runtime.processor.pad_token_id is None:
            runtime.processor.pad_token = runtime.processor.eos_token
        runtime.processor.padding_side = "left"
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quantization,
            local_files_only=True,
        )
        runtime.model = PeftModel.from_pretrained(base_model, adapter_path).eval()
        runtime.device = next(runtime.model.parameters()).device
        runtime.lock = asyncio.Lock()
        runtime.generator_model_id = generator_model_id
        runtime.critic_model_id = critic_model_id
        yield
        del runtime.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    app = FastAPI(lifespan=lifespan)

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [
            {"id": runtime.generator_model_id, "object": "model", "owned_by": "local"},
            {"id": runtime.critic_model_id, "object": "model", "owned_by": "local"},
        ]}

    @app.post("/v1/chat/completions")
    async def chat(request: ChatCompletionRequest) -> dict[str, Any]:
        if request.stream:
            raise HTTPException(status_code=400, detail="Streaming is not supported")
        async with runtime.lock:
            started = time.time()
            try:
                content = (await asyncio.to_thread(render_completions, [request]))[0]
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
        return completion_payload(request.model, content, started)

    @app.post("/v1/chat/completions/batch")
    async def batch_chat(request: ChatCompletionBatchRequest) -> dict[str, Any]:
        if any(item.stream for item in request.requests):
            raise HTTPException(status_code=400, detail="Streaming is not supported")
        async with runtime.lock:
            started = time.time()
            try:
                contents = await asyncio.to_thread(render_completions, request.requests)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {
            "object": "list",
            "data": [
                completion_payload(item.model, content, started)
                for item, content in zip(request.requests, contents)
            ],
        }

    return app


def main() -> None:
    args = parse_args()
    app = build_app(args.model, args.adapter, args.generator_model_id, args.critic_model_id)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
