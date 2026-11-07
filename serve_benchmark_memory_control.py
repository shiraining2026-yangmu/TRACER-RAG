#!/usr/bin/env python3
"""Wrap unchanged inference services with explicit CUDA-cache lifecycle control."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

def with_memory_policy(render, policy, release):
    def controlled(requests):
        try:
            return render(requests)
        finally:
            if policy == "release_after_request":
                release()
    return controlled


def main():
    import torch
    import uvicorn

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--service-kind", choices=("generator", "critic"), required=True)
    parser.add_argument("--memory-policy", choices=("retain", "release_after_request"), required=True)
    options, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    if options.service_kind == "generator":
        import serve_single_generator as service
        args = service.parse_args()
        app = service.build_app(args)
        name = "render"
    else:
        import serve_qwen_critic_adapter as service
        args = service.parse_args()
        app = service.build_app(args.model, args.adapter, args.generator_model_id, args.critic_model_id)
        name = "render_completions"
    original = getattr(service, name)

    def release():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    setattr(service, name, with_memory_policy(original, options.memory_policy, release))

    def memory():
        free, total = torch.cuda.mem_get_info()
        return {"allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved(),
                "device_free_bytes": free, "device_total_bytes": total,
                "memory_policy": options.memory_policy, "service_kind": options.service_kind,
                "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    @app.get("/v1/benchmark_memory")
    async def get_memory():
        return memory()

    @app.post("/v1/benchmark_release_cache")
    async def release_cache():
        async with service.runtime.lock:
            release()
            return memory()

    print(json.dumps({"memory_policy": options.memory_policy, "service_kind": options.service_kind}), flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
