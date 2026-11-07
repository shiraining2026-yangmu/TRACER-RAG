"""Resumable, revision-pinned candidate weights with verified completion markers."""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from benchmark_integrity import sha256

PATTERNS = ("*.safetensors", "*.json", "*.model", "*.py", "*.jinja", "*.txt", "LICENSE*", "README*", "USE_POLICY*")


def selected(name: str) -> bool:
    return "/" not in name and any(fnmatch.fnmatch(name, pattern) for pattern in PATTERNS)


def exclude_duplicate_weights(items: list, name_key: str) -> list:
    """Avoid downloading duplicate consolidated weights beside an index."""
    def value(item):
        return item[name_key] if isinstance(item, dict) else getattr(item, name_key)

    names = {value(item) for item in items}
    has_index = any(Path(name).name == "model.safetensors.index.json" for name in names)
    if not has_index:
        return items
    return [item for item in items if Path(value(item)).name != "consolidated.safetensors"]


def valid_marker(path: Path, marker: dict) -> bool:
    return bool(marker.get("files")) and all(
        (path / item["path"]).is_file() and (path / item["path"]).stat().st_size == item["size"]
        for item in marker["files"])


def validate_shards(path: Path) -> None:
    if not (path / "config.json").is_file() or not (path / "tokenizer_config.json").is_file():
        raise ValueError("Missing model/tokenizer configuration")
    index = path / "model.safetensors.index.json"
    files = set(json.loads(index.read_text())["weight_map"].values()) if index.exists() else {"model.safetensors"}
    for name in files:
        if not (path / name).is_file() or (path / name).stat().st_size < 1024:
            raise ValueError(f"Missing/truncated weight shard: {name}")


def modelscope_file(path: Path, repo: str, item: dict) -> dict:
    name = item["Path"]
    target = path / name
    expected = int(item["Size"])
    checksum = item["Sha256"]
    if target.exists() and target.stat().st_size == expected and sha256(target) == checksum:
        return {"path": name, "size": expected, "sha256": checksum}
    partial = target.with_suffix(target.suffix + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset >= expected:
        if offset == expected and sha256(partial) == checksum:
            partial.replace(target)
            return {"path": name, "size": expected, "sha256": checksum}
        # A completed but uncommitted or corrupt temporary download must not
        # cause an endless HTTP 416 loop on every resumed attempt.
        with partial.open("wb"):
            pass
        offset = 0
    url = f"https://modelscope.cn/api/v1/models/{repo}/repo"
    with requests.get(url, params={"Revision": item["Revision"], "FilePath": name},
                      headers={"Range": f"bytes={offset}-"} if offset else {}, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        if response.status_code == 206:
            if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise ValueError("Unexpected HTTP range")
            mode = "ab"
        else:
            mode = "wb"
        with partial.open(mode) as handle:
            for block in response.iter_content(8 * 1024 * 1024):
                handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
    if partial.stat().st_size != expected or sha256(partial) != checksum:
        raise ValueError(f"Download checksum mismatch: {name}")
    partial.replace(target)
    print(f"verified {name} ({expected} bytes)", flush=True)
    return {"path": name, "size": expected, "sha256": checksum}


def download(args):
    marker_path = args.path / "benchmark_download_complete.json"
    if marker_path.exists():
        marker = json.loads(marker_path.read_text())
        if marker.get("repo") == args.repo and valid_marker(args.path, marker):
            validate_shards(args.path)
            args.manifest.write_text(json.dumps(marker, indent=2) + "\n")
            return
    args.path.mkdir(parents=True, exist_ok=True)
    if args.modelscope:
        response = requests.get(f"https://modelscope.cn/api/v1/models/{args.modelscope}/repo/files",
                                params={"Recursive": "true"}, timeout=30)
        response.raise_for_status()
        items = [item for item in response.json()["Data"]["Files"] if item["Type"] == "blob" and selected(item["Path"])]
        items = exclude_duplicate_weights(items, "Path")
        with ThreadPoolExecutor(max_workers=2) as pool:
            files = list(pool.map(lambda item: modelscope_file(args.path, args.modelscope, item), items))
        source = {"provider": "modelscope", "mirror_repo": args.modelscope,
                  "file_revisions": {item["Path"]: item["Revision"] for item in items}}
    else:
        from huggingface_hub import HfApi, snapshot_download
        # Token stays inside the process environment; never a shell argument.
        token = os.environ.get("HF_TOKEN") or None
        info = HfApi().model_info(args.repo, token=token, files_metadata=True, timeout=30)
        items = [item for item in info.siblings if selected(item.rfilename)]
        items = exclude_duplicate_weights(items, "rfilename")
        snapshot_download(repo_id=args.repo, revision=info.sha, local_dir=args.path, token=token,
                          allow_patterns=[item.rfilename for item in items], max_workers=2, etag_timeout=30)
        files = []
        for item in items:
            target = args.path / item.rfilename
            if target.stat().st_size != item.size:
                raise ValueError(f"Size mismatch: {item.rfilename}")
            checksum = sha256(target)
            if item.lfs and checksum != item.lfs.sha256:
                raise ValueError(f"Weight checksum mismatch: {item.rfilename}")
            files.append({"path": item.rfilename, "size": item.size, "sha256": checksum})
        source = {"provider": "huggingface", "revision": info.sha}
    validate_shards(args.path)
    marker = {"repo": args.repo, **source, "files": files, "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    marker_path.write_text(json.dumps(marker, indent=2) + "\n")
    args.manifest.write_text(json.dumps(marker, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--modelscope")
    args = parser.parse_args()
    download(args)


if __name__ == "__main__":
    main()
