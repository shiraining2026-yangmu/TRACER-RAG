#!/usr/bin/env python3
"""Conventional dense chunk RAG for answer-blind external exam questions.

No AEK atomization, BM25, relations, reranker, or Critic. Each stage has its
own immutable contract. Generation never opens a workbook or an answer key.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time

VERSION = "dense-chunk-rag-v1"
FORBIDDEN = {"answer", "correct_answer", "standard_answer", "答案", "标准答案", "正确答案", "答案内容", "解析", "explanation"}
ALLOWED = {"application_id", "question_id", "question", "retrieval_query", "year", "exam", "unit", "source_number", "question_type", "shared_option_group", "data_status", "source_row", "split"}


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def log(stage, **kwargs):
    print(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "stage": stage, **kwargs}, ensure_ascii=False), flush=True)


def contract(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Resume contract mismatch: {path}. Use a new output directory.")
    else:
        atomic_json(path, value)


def compact(text):
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text.lower())


def clean_text(text):
    text = str(text)
    markers = "锛銆锟璇烽棶鎮ㄦ湁"
    if sum(text.count(c) for c in markers) >= 2:
        try:
            repaired = text.encode("gb18030").decode("utf-8")
            if sum(repaired.count(c) for c in markers) < sum(text.count(c) for c in markers):
                text = repaired
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    text = re.sub(r"<(?:image|audio|video)>", " ", str(text))
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"[*#>`_]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def chunks(text, size=700, overlap=100):
    if size <= overlap or overlap < 0:
        raise ValueError("Require chunk_size > overlap >= 0")
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        yield start, end, text[start:end]
        if end == len(text):
            break
        start = end - overlap


def questions(paths):
    result = []
    for path in paths:
        rows = list(read_jsonl(path))
        for row in rows:
            if set(row) - ALLOWED or FORBIDDEN & set(row):
                raise ValueError(f"Unexpected/answer-bearing input fields: {set(row) - ALLOWED}")
            if not all(isinstance(row.get(k), str) and row[k].strip() for k in ("application_id", "question", "retrieval_query")):
                raise ValueError("Question needs nonempty ID, question, retrieval_query")
            stem = row["question"].split("题干：", 1)[-1].split("\n选项：", 1)[0].strip()
            if stem != row["retrieval_query"].strip():
                raise ValueError("retrieval_query must be the stem only")
        if len({r["application_id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate question IDs")
        result.append((Path(path), rows))
    all_ids = [r["application_id"] for _, rows in result for r in rows]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Duplicate IDs across exam files")
    return result


class ContaminationGuard:
    """Conservative stem-only exact/near duplicate screen of source questions.

    Does not read official answers. This does not certify absence of paraphrases
    or pretraining contamination; excluded source records are independently logged.
    """
    def __init__(self, stems):
        self.stems = [compact(s) for s in stems]
        self.grams = [{s[i:i+5] for i in range(len(s)-4)} for s in self.stems]
        self.inverse = defaultdict(set)
        for i, grams in enumerate(self.grams):
            for g in grams:
                self.inverse[g].add(i)

    def match(self, source_question):
        text = compact(source_question)
        grams = {text[i:i+5] for i in range(len(text)-4)}
        counts = Counter(i for g in grams for i in self.inverse.get(g, ()))
        for i, shared in counts.items():
            stem = self.stems[i]
            if len(stem) >= 12 and stem in text:
                return i, "normalized_stem_containment"
            union = len(grams) + len(self.grams[i]) - shared
            if union and shared / union >= .80:
                return i, "question_5gram_jaccard_ge_0.80"
        return None


class SourcePool:
    """Recover pre-cleaning records using all retained K text, plus Q if known.

    Punctuation removal in the historical cleaned files changed record hashes.
    A normalized match must contain EVERY retained K fragment, in the same
    modality, and match the normalized historical question when available.
    """
    def __init__(self, knowledge_rows, application_rows):
        self.fragments = defaultdict(list)
        self.datasets = {}
        self.source_questions = {}
        for row in knowledge_rows:
            source = row["source"]
            key = source["record_hash"]
            self.fragments[key].append(compact(row["text"]))
            self.datasets[key] = source["dataset"]
        for row in application_rows:
            self.source_questions[row["source"]["record_hash"]] = compact(row["question"])
        self.by_question = defaultdict(set)
        self.no_question_signatures = defaultdict(set)
        for key, fragments in self.fragments.items():
            if key in self.source_questions:
                self.by_question[(self.datasets[key], self.source_questions[key])].add(key)
            else:
                # One retained fragment signature narrows candidates; full fragments
                # are still checked below. No fuzzy content matching is used.
                longest = max(fragments, key=len)
                self.no_question_signatures[(self.datasets[key], longest[:12])].add(key)
        self.no_question_datasets = {d for d, _ in self.no_question_signatures}

    def match(self, dataset, question, answer):
        q, a = compact(question), compact(answer)
        candidates = set(self.by_question.get((dataset, q), ()))
        if dataset in self.no_question_datasets:
            for i in range(max(0, len(a)-11)):
                candidates.update(self.no_question_signatures.get((dataset, a[i:i+12]), ()))
        return {key for key in candidates if all(fragment and fragment in a for fragment in self.fragments[key])}


def prepare(args, exams):
    from download_dense_rag_sources import FILES, REPO, REVISION
    directory = args.index_dir
    directory.mkdir(parents=True, exist_ok=True)
    source_info = []
    for name, (size, sha) in FILES.items():
        path = args.source_dir / name
        if path.stat().st_size != size or file_hash(path) != sha:
            raise ValueError(f"Source hash failed: {path}")
        source_info.append({"name": name, "sha256": sha, "bytes": size})
    pool = SourcePool(read_jsonl(args.aek_knowledge), read_jsonl(args.aek_applications))
    allowed_sources = set(pool.fragments)
    spec = {"version": VERSION, "repo": REPO, "revision": REVISION, "sources": source_info,
            "chunk_size_chars": args.chunk_size, "overlap_chars": args.overlap,
            "questions": {p.name: file_hash(p) for p, _ in exams},
            "contamination_screen": "source question: normalized stem containment >=12 chars or 5gram Jaccard >=0.80",
            "corpus": "full original QA of source records contributing to frozen K corpus; no atom/edge processing of chunk contents",
            "source_record_allowlist_knowledge_sha256": file_hash(args.aek_knowledge),
            "source_record_questions_applications_sha256": file_hash(args.aek_applications),
            "source_recovery": "exact record SHA256 OR same modality + ALL normalized K fragments contained + normalized historical Q equal when available",
            "minimum_source_coverage": args.minimum_source_coverage,
            "script_sha256": file_hash(Path(__file__))}
    contract(directory / "corpus_contract.json", spec)
    db = directory / "chunks.sqlite"
    if (directory / "corpus_summary.json").exists():
        with sqlite3.connect(db) as c:
            count = c.execute("select count(*) from chunks").fetchone()[0]
        if count != json.loads((directory / "corpus_summary.json").read_text())["chunks"]:
            raise ValueError("Corpus database count mismatch")
        return count
    temporary = directory / "chunks.building.sqlite"
    # Only this known disposable staging file can be rebuilt after an interruption.
    if temporary.exists():
        temporary.unlink()
    c = sqlite3.connect(temporary)
    c.execute("create table chunks (position integer primary key, chunk_id text unique, text text, source_json text)")
    guard = ContaminationGuard([r["retrieval_query"] for _, rows in exams for r in rows])
    flat = [r for _, rows in exams for r in rows]
    seen = set()
    counts = Counter()
    source_membership = set()
    matched_sources = set()
    recovery_modes = Counter()
    audit_path = directory / "excluded_source_records.jsonl"
    with audit_path.open("w", encoding="utf-8") as audit:
        for name in FILES:
            data = json.loads((args.source_dir / name).read_text(encoding="utf-8"))
            for index, row in enumerate(data):
                counts["source_records"] += 1
                conv = row.get("conversations", [])
                q = "\n".join(str(m.get("content", "")) for m in conv if m.get("role") == "user")
                a = "\n".join(str(m.get("content", "")) for m in conv if m.get("role") == "assistant")
                q, a = clean_text(q), clean_text(a)
                record_hash = digest(q + "\n" + a)
                dataset = {"TCM_text_instruction.json": "text_cleaned", "TCM_vision_instruction.json": "vision_cleaned", "TCM_speech_instruction.json": "speech_cleaned"}[name]
                recovered = {record_hash} if record_hash in allowed_sources else pool.match(dataset, q, a)
                if not recovered:
                    counts["outside_shared_source_pool"] += 1
                    continue
                matched_sources.update(recovered)
                recovery_modes["exact_record_sha256" if record_hash in allowed_sources else "normalized_full_K_content_and_Q_when_available"] += 1
                match = guard.match(q)
                if match:
                    counts["excluded_exam_overlap_records"] += 1
                    audit.write(json.dumps({"file": name, "record_index": index, "question_id": flat[match[0]]["application_id"], "reason": match[1]}, ensure_ascii=False) + "\n")
                    continue
                if not a:
                    counts["empty_response"] += 1
                    continue
                source_membership.update(recovered)
                text = (q + "\n" + a).strip()
                for start, end, fragment in chunks(text, args.chunk_size, args.overlap):
                    key = digest(fragment)
                    if key in seen:
                        counts["duplicate_chunks"] += 1
                        continue
                    seen.add(key)
                    source = {"file": name, "record_index": index, "start_char": start, "end_char": end, "source_text_sha256": digest(text), "original_record_hash": record_hash, "historical_record_hashes": sorted(recovered)}
                    c.execute("insert into chunks values (?,?,?,?)", (counts["chunks"], "CH-" + key, fragment, json.dumps(source)))
                    counts["chunks"] += 1
                if index % 5000 == 0:
                    c.commit()
                    log("chunking", file=name, record=index, **counts)
            c.commit()
            del data
    c.close()
    coverage = len(matched_sources) / len(allowed_sources)
    if coverage < args.minimum_source_coverage:
        atomic_json(directory / "source_audit_failed.json", {"matched": len(matched_sources), "expected": len(allowed_sources), "coverage": coverage})
        raise ValueError(f"Insufficient source recovery: {len(matched_sources)}/{len(allowed_sources)}. Do not substitute a different corpus.")
    temporary.replace(db)
    # Provenance audit only; never used for retrieval or answer selection.
    audit = {"frozen_K_unique_source_records": len(allowed_sources), "matched_original_source_records": len(matched_sources), "source_record_coverage": coverage, "recovery_modes": dict(recovery_modes), "sources_without_historical_question": len(allowed_sources-set(pool.source_questions))}
    if args.aek_applications:
        originals = {r.get("source", {}).get("record_hash") for r in read_jsonl(args.aek_applications)} - {None}
        audit.update({"historical_aek_unique_source_records": len(originals), "verified_source_matches_after_exam_screen": len(originals & source_membership),
                 "note": "Ordinary chunking of the shared K-source records; exact/near exam duplicates excluded. R4 evidence coverage differs and is reported separately."})
    summary = {**counts, "source_provenance_audit": audit, "corpus_contract_sha256": file_hash(directory / "corpus_contract.json")}
    atomic_json(directory / "corpus_summary.json", summary)
    log("corpus_ready", **summary)
    return counts["chunks"]


def encoder_load(args):
    from sentence_transformers import SentenceTransformer
    import torch
    encoder = SentenceTransformer(str(args.embedding_model), device=args.device, trust_remote_code=False)
    encoder.max_seq_length = 1024
    if args.device.startswith("cuda"):
        encoder.half()
    return encoder


def exact_top3(vectors, batch, block_size=8192):
    """Return exact cosine/IP top three, including deterministic ties."""
    import numpy as np
    count = len(vectors)
    if count < 3:
        raise ValueError("At least three chunks required")
    best_s = np.full((len(batch), 3), -np.inf, dtype="float32")
    best_p = np.full((len(batch), 3), count, dtype="int64")
    for start in range(0, count, block_size):
        scores = np.asarray(vectors[start:start+block_size]) @ batch.T
        for j in range(len(batch)):
            take = np.argsort(-scores[:, j], kind="stable")[:3]
            positions = np.concatenate([best_p[j], start + take])
            values = np.concatenate([best_s[j], scores[take, j]])
            order = np.lexsort((positions, -values))[:3]
            best_p[j], best_s[j] = positions[order], values[order]
    return best_p, best_s


def embed_and_retrieve(args, exams, count):
    import numpy as np
    import torch
    directory = args.index_dir
    encoder = encoder_load(args)
    dim = encoder.get_sentence_embedding_dimension()
    spec = {"corpus_contract_sha256": file_hash(directory / "corpus_contract.json"), "model": str(args.embedding_model),
            "model_config_sha256": file_hash(args.embedding_model / "config.json"), "dimension": dim,
            "max_sequence_tokens": 1024, "normalized": True, "inference_dtype": "float16" if args.device.startswith("cuda") else "float32", "storage_dtype": "float32"}
    contract(directory / "embedding_contract.json", spec)
    path = directory / "dense_vectors.npy"
    state_path = directory / "embedding_progress.json"
    done = json.loads(state_path.read_text())["completed"] if state_path.exists() else 0
    if done > count or (done and not path.exists()):
        raise ValueError("Invalid embedding progress")
    vectors = np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+", dtype="float32", shape=(count, dim))
    if vectors.shape != (count, dim):
        raise ValueError("Embedding shape mismatch")
    c = sqlite3.connect(directory / "chunks.sqlite")
    started = time.monotonic()
    for start in range(done, count, args.encode_batch):
        rows = c.execute("select text from chunks where position>=? and position<? order by position", (start, start + args.encode_batch)).fetchall()
        encoded = encoder.encode([r[0] for r in rows], batch_size=args.encode_batch, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        if not np.isfinite(encoded).all():
            raise ValueError("Nonfinite embeddings")
        # Normalize in FP32 too, so exact IP is cosine despite half precision encoding.
        encoded = np.asarray(encoded, dtype="float32")
        encoded /= np.maximum(np.linalg.norm(encoded, axis=1, keepdims=True), 1e-12)
        vectors[start:start+len(rows)] = encoded
        vectors.flush()
        atomic_json(state_path, {"completed": start + len(rows), "total": count})
        if start == done or (start // args.encode_batch) % 50 == 0 or start + len(rows) == count:
            speed = (start + len(rows) - done) / max(time.monotonic() - started, .001)
            log("embedding", completed=start+len(rows), total=count, chunks_per_second=round(speed, 2), eta_seconds=round((count-start-len(rows))/max(speed,.001)))
    # Encode only stem strings, freeze all retrievals, then release the GPU encoder.
    for question_path, rows in exams:
        out = args.output_dir / question_path.stem.replace("questions_", "")
        out.mkdir(parents=True, exist_ok=True)
        contract(out / "retrieval_contract.json", {"questions_sha256": file_hash(question_path), "embedding_contract_sha256": file_hash(directory / "embedding_contract.json"), "top_k": 3, "query": "stem_only", "search": "exact_normalized_inner_product"})
        target = out / "retrieval.jsonl"
        existing = list(read_jsonl(target)) if target.exists() else []
        if existing:
            if len(existing) != len(rows) or {r["application_id"] for r in existing} != {r["application_id"] for r in rows}:
                raise ValueError("Incomplete frozen retrieval file")
            continue
        query_vectors = encoder.encode([r["retrieval_query"] for r in rows], batch_size=args.encode_batch, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        query_vectors = np.asarray(query_vectors, dtype="float32")
        query_vectors /= np.maximum(np.linalg.norm(query_vectors, axis=1, keepdims=True), 1e-12)
        temporary = target.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as f:
            # Blocked exact search bounds RAM. Stable tie-break is ascending chunk position.
            for qstart in range(0, len(rows), 16):
                batch = query_vectors[qstart:qstart+16]
                best_p, best_s = exact_top3(vectors, batch)
                for j, row in enumerate(rows[qstart:qstart+16]):
                    evidence = []
                    for position, score in zip(best_p[j], best_s[j]):
                        doc_id, text, source = c.execute("select chunk_id,text,source_json from chunks where position=?", (int(position),)).fetchone()
                        evidence.append({"doc_id": doc_id, "score": float(score), "text": text, "source": json.loads(source)})
                    f.write(json.dumps({**row, "evidence": evidence}, ensure_ascii=False) + "\n")
                log("retrieval", exam=question_path.name, completed=min(qstart+16,len(rows)), total=len(rows))
        temporary.replace(target)
    c.close()
    del vectors, encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def messages(row):
    evidence = "\n\n".join(f"[D{i} | {e['doc_id']}]\n{e['text']}" for i, e in enumerate(row["evidence"], 1))
    return [{"role": "system", "content": (
        "你正在完成中医执业医师单选题作答。展示的检索资料仅供参考：相关时可用于辅助判断，"
        "不相关或不足时也必须结合题干和选项作出最佳选择，绝不能以证据不足、无法判断或建议咨询代替作答。"
        "只输出一行，严格写成“答案：X”（X 必须为 A、B、C、D、E 之一）；不要输出依据、解释或任何其他文字。")},
        {"role": "user", "content": f"题目：\n{row['question']}\n\n检索参考资料：\n{evidence}"}]


def generate(args, exams):
    # Reuse the existing generator client and answer-format policy, never its retriever.
    from run_questionbank_rag_vs_alignrag import batch_client, enforce_choices
    client = batch_client(args.api_base, args.generator_model, 32, 300, 2, 20)
    for question_path, rows in exams:
        out = args.output_dir / question_path.stem.replace("questions_", "")
        retrieval = out / "retrieval.jsonl"
        contract(out / "generation_contract.json", {"retrieval_sha256": file_hash(retrieval), "model": args.generator_model,
            "temperature": 0.0, "max_tokens": 32, "batch_size": args.batch_size,
            "prompt_version": VERSION, "prompt_template_sha256": digest(json.dumps(messages({"evidence": [], "question": "{question}"}),ensure_ascii=False)),
            "format_policy": "existing one repair then explicitly marked forced_default_A"})
        result = out / "dense_rag_results.jsonl"
        # Atomic rewrite per batch means a stopped writer never leaves a partial JSONL row.
        completed = list(read_jsonl(result)) if result.exists() else []
        ids = {r["application_id"] for r in completed if r.get("status") == "ok" and re.fullmatch(r"答案：[A-E]", r.get("answer", ""))}
        if len(ids) != len(completed) or not ids <= {r["application_id"] for r in rows}:
            raise ValueError("Corrupt, duplicate, or invalid existing results")
        pending = [r for r in read_jsonl(retrieval) if r["application_id"] not in ids]
        started = time.monotonic()
        initial = len(completed)
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start+args.batch_size]
            raw = client([messages(r) for r in batch])
            resolved = enforce_choices(client, batch, raw)
            for row, raw_answer, (answer, origin) in zip(batch, raw, resolved):
                completed.append({**row, "answer": answer, "raw_answer": raw_answer, "choice_origin": origin, "status": "ok", "method": VERSION})
            temporary = result.with_suffix(".jsonl.tmp")
            with temporary.open("w", encoding="utf-8") as f:
                for row in completed:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            temporary.replace(result)
            elapsed = time.monotonic()-started
            summary = {"status": "complete" if len(completed)==len(rows) else "running", "completed": len(completed), "total": len(rows),
                       "choice_origins": dict(Counter(r["choice_origin"] for r in completed)), "elapsed_this_run_seconds": round(elapsed),
                       "eta_seconds": round((len(rows)-len(completed))*elapsed/max(len(completed)-initial,1))}
            atomic_json(out / "summary.json", summary)
            log("generation", exam=question_path.name, **summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--questions", type=Path, nargs="+", required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--aek-applications", type=Path, required=True, help="Read normalized historical questions for source verification only")
    parser.add_argument("--aek-knowledge", type=Path, required=True, help="Read source hashes only to share source-record pool; never index K atoms")
    parser.add_argument("--minimum-source-coverage", type=float, default=.99)
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--encode-batch", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, choices=range(1,5), default=4)
    parser.add_argument("--api-base", default="http://127.0.0.1:8001")
    parser.add_argument("--generator-model", default="qwen-generator-repair-v2")
    parser.add_argument("--stage", choices=["prepare", "retrieve", "generate", "all"], default="all")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Single writer for both index and result directory on Linux server.
    import fcntl
    with (args.index_dir.parent / (args.index_dir.name + ".lock")).open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        exams = questions(args.questions)
        if args.stage in ("prepare", "retrieve", "all"):
            count = prepare(args, exams)
            if args.stage != "prepare":
                embed_and_retrieve(args, exams, count)
        if args.stage in ("generate", "all"):
            generate(args, exams)
    log("finished", stage_requested=args.stage)


if __name__ == "__main__":
    main()
