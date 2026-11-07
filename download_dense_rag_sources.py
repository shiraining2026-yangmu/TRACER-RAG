#!/usr/bin/env python3
"""Download pinned ShizhenGPT instruction JSONs, without image/audio archives."""
import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

REPO = "FreedomIntelligence/TCM-Instruction-Tuning-ShizhenGPT"
REVISION = "718596f04350cc6c4c7735f8d62899a5a1649cb8"
FILES = {
    "TCM_text_instruction.json": (296530748, "068c31d3990fb2fc1d7f13a3b882c6fc1a0a6e5a9f7579d2bd76dfcbb1d056a9"),
    "TCM_vision_instruction.json": (145110968, "f7d7c715b51edbc670d9b243262e9c5b3dce4c4cd4b1bf177ab4580ffc357c32"),
    "TCM_speech_instruction.json": (145218430, "c9b2af87017cb1779aed1d8ab7ef8f45b7f57ae1186fc7b22de3d4701db68d0e"),
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name, (size, checksum) in FILES.items():
        target = directory / name
        if target.exists():
            if target.stat().st_size == size and sha256(target) == checksum:
                print(json.dumps({"stage": "download", "file": name, "status": "verified_existing"}), flush=True)
                continue
            raise ValueError(f"Existing source has wrong size/hash: {target}")
        temporary = directory / (name + ".part")
        for attempt in range(4):
            try:
                # Restart partial downloads; published LFS SHA-256 is checked before use.
                url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}"
                with urllib.request.urlopen(url, timeout=90) as response, temporary.open("wb") as out:
                    while block := response.read(4 * 1024 * 1024):
                        out.write(block)
                if temporary.stat().st_size != size or sha256(temporary) != checksum:
                    raise ValueError("Downloaded source failed published size/SHA-256 check")
                temporary.replace(target)
                print(json.dumps({"stage": "download", "file": name, "status": "verified_downloaded"}), flush=True)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
    (directory / "source_manifest.json").write_text(json.dumps({
        "repo": REPO, "revision": REVISION,
        "files": {name: {"bytes": size, "sha256": digest} for name, (size, digest) in FILES.items()},
        "note": "Original instruction sources; equivalence to historical *_cleaned.jsonl must be audited separately.",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    download(parser.parse_args().output)
