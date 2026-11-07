#!/usr/bin/env python3
"""Resume Gemma and Yi with cache release and a reachable synthetic preflight."""
from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
import signal

import run_three_model_wrong_only_memory_v2 as memory_v2
from preflight_wrong_only_memory_v3 import STRESS_POLICY
from summarize_ter_wrong_only_memory_v3 import PROTOCOL, summarize

ADDITIONAL_CODE = ("run_three_model_wrong_only_memory_v3.py", "preflight_wrong_only_memory_v3.py",
                   "summarize_ter_wrong_only_memory_v3.py")


@contextlib.contextmanager
def version_bindings():
    saved = memory_v2.PROTOCOL, memory_v2.summarize, memory_v2.ADDITIONAL_CODE
    memory_v2.PROTOCOL = PROTOCOL
    memory_v2.summarize = summarize
    memory_v2.ADDITIONAL_CODE = (*saved[2], *ADDITIONAL_CODE)
    try:
        with memory_v2.version_bindings():
            commit = memory_v2.original.commit_manifest
            def commit_manifest(work, manifest):
                manifest["preflight_stress_policy"] = STRESS_POLICY
                commit(work, manifest)
            memory_v2.original.commit_manifest = commit_manifest
            yield
    finally:
        memory_v2.PROTOCOL, memory_v2.summarize, memory_v2.ADDITIONAL_CODE = saved


class Supervisor(memory_v2.Supervisor):
    def run(self, script, *args):
        script = {"preflight_wrong_only.py": "preflight_wrong_only_memory_v3.py",
                  "summarize_ter_wrong_only.py": "summarize_ter_wrong_only_memory_v3.py"}.get(script, script)
        return super().run(script, *args)


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    app = Path(__file__).resolve().parent
    output = args.output_root or app / "questionbank_eval/model_benchmarks/ter_wrong_only_batch4_memory_v3"
    output.mkdir(parents=True, exist_ok=True)
    with version_bindings(), (output / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        supervisor = Supervisor(app, output)
        def stop(signum, frame):
            raise memory_v2.original.StopRequested()
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            supervisor.run_queue()
        except memory_v2.original.StopRequested:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            supervisor.state(supervisor.stage, "interrupted")
            supervisor.stop_servers()
            raise SystemExit(143)
        finally:
            supervisor.stop_servers()


if __name__ == "__main__":
    main()
