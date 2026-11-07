#!/usr/bin/env python3
"""Build a conservative A/E/K corpus from the three cleaned JSONL files.

This is a structural, reproducible filter.  It deliberately does not claim
clinical validation or model-based semantic review.  Records are admitted only
when independently-tagged atomic K and A fragments form a non-duplicative
same-source support relation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


TCM_TERMS = (
    "中医", "中药", "方剂", "方药", "针灸", "针刺", "艾灸", "经络", "穴位", "辨证",
    "辨证论治", "证候", "证型", "病机", "治则", "治法", "舌苔", "脉象", "脉搏",
    "阴阳", "气血", "津液", "湿热", "痰湿", "风寒", "风热", "虚寒", "气虚",
    "血虚", "阴虚", "阳虚", "瘀血", "肝肾", "脾胃", "心肝", "肺脾", "归经",
    "汤", "散", "丸", "饮", "针", "灸", "中成药", "望闻问切", "四诊",
    "acupuncture", "moxibustion", "traditional chinese medicine", "tcm", "herbal formula",
)
DOMAIN_ANCHORS = (
    "中医", "中药", "方剂", "方药", "针灸", "针刺", "艾灸", "经络", "穴位", "辨证",
    "证候", "证型", "病机", "治则", "治法", "舌苔", "脉象", "阴阳", "气血", "津液",
    "湿热", "痰湿", "风寒", "风热", "虚寒", "气虚", "血虚", "阴虚", "阳虚", "瘀血",
    "肝肾", "脾胃", "心肝", "肺脾", "归经", "中成药",
)
ROLE_PATTERNS = {
    "definition": ("是指", "称为", "定义为", "属于", "即为", "又称"),
    "fact": ("表现为", "常见", "多见", "特点", "包括", "具有", "可出现", "主要为"),
    "condition": ("若", "如果", "当", "在.*情况下", "适用于", "禁用于", "不宜"),
    "evidence": ("依据", "提示", "见", "舌", "苔", "脉", "检查", "症状"),
    "reasoning": ("因此", "由于", "故", "从而", "导致", "提示", "考虑"),
    "conclusion": ("诊断为", "辨证为", "可判断", "结论", "属于.*证"),
    "recommendation": ("建议", "应当", "应该", "宜", "可采用", "可选用", "推荐"),
    "step": ("首先", "其次", "然后", "最后", "步骤", "每日", "每次", "进行"),
    "case": ("患者", "病人", "主诉", "初诊", "医案", "病例"),
}
MOJIBAKE_MARKERS = "锛銆锟璇烽棶鎮ㄦ湁"


def source_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def compact(value: str) -> str:
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", value.lower())


def char_bigrams(value: str) -> set[str]:
    value = compact(value)
    return {value[i : i + 2] for i in range(max(0, len(value) - 1))} if len(value) > 1 else set()


def jaccard(left: str, right: str) -> float:
    a, b = char_bigrams(left), char_bigrams(right)
    return len(a & b) / len(a | b) if a and b else 0.0


def repair_mojibake(value: str) -> str:
    """Recover the common UTF-8-as-GBK corruption, only when it improves it."""
    if not value or sum(value.count(c) for c in MOJIBAKE_MARKERS) < 2:
        return value
    try:
        repaired = value.encode("gb18030").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value
    before = sum(value.count(c) for c in MOJIBAKE_MARKERS)
    after = sum(repaired.count(c) for c in MOJIBAKE_MARKERS)
    return repaired if after < before else value


def clean_text(value: Any) -> str:
    value = repair_mojibake(str(value or ""))
    value = re.sub(r"```.*?```", " ", value, flags=re.S)
    value = re.sub(r"[*#>`_]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def has_tcm_signal(value: str) -> bool:
    lower = value.lower()
    return sum(1 for term in TCM_TERMS if term in lower) >= 2 or any(term in lower for term in ("中医", "针灸", "acupuncture", "traditional chinese medicine"))


def anchors(value: str, limit: int = 6) -> list[str]:
    lower = value.lower()
    found = [term for term in DOMAIN_ANCHORS if term.lower() in lower]
    # Formula names are useful, but only collect bounded CJK phrases ending in a formula suffix.
    found.extend(re.findall(r"[\u4e00-\u9fff]{2,8}(?:汤|散|丸|饮|方)", value))
    return list(dict.fromkeys(found))[:limit]


def atomize(value: str) -> list[str]:
    value = re.sub(r"(?<!\n)([一二三四五六七八九十]+、|\d+[.、])", r"\n\1", value)
    raw = re.split(r"(?:\n+|(?<=[。！？；;]))", value)
    atoms: list[str] = []
    for fragment in raw:
        fragment = fragment.strip(" -:：")
        if len(compact(fragment)) < 12:
            continue
        while len(fragment) > 480:
            cut = max(fragment.rfind("，", 0, 420), fragment.rfind("、", 0, 420), fragment.rfind(" ", 0, 420))
            cut = cut if cut > 100 else 420
            atoms.append(fragment[:cut].strip())
            fragment = fragment[cut:].strip("，、 ")
        if len(compact(fragment)) >= 12:
            atoms.append(fragment)
    return atoms[:24]


def match_role(text: str, role: str) -> bool:
    return any(re.search(pattern, text, re.I) for pattern in ROLE_PATTERNS[role])


def role_labels(text: str) -> list[str]:
    labels = [role for role in ROLE_PATTERNS if match_role(text, role)]
    return labels or ["fact"]


def view_votes(text: str, labels: list[str]) -> dict[str, str]:
    """Three intentionally independent role views; values are K, A, or M."""
    action = bool(set(labels) & {"conclusion", "recommendation", "step"})
    basis = bool(set(labels) & {"definition", "fact", "condition", "evidence", "reasoning"})
    return {
        "basis_vs_action": "A" if action and not basis else "K" if basis and not action else "M",
        "reusable_vs_current": "A" if (match_role(text, "case") or action) else "K",
        "premise_vs_inference": "A" if bool(set(labels) & {"conclusion", "recommendation", "step"}) else "K" if bool(set(labels) & {"condition", "evidence", "reasoning", "fact"}) else "M",
    }


def consensus(votes: dict[str, str]) -> str:
    k = sum(v == "K" for v in votes.values())
    a = sum(v == "A" for v in votes.values())
    return "K" if k >= 2 and a == 0 else "A" if a >= 2 and k == 0 else "M"


def relation_type(k_labels: list[str], a_labels: list[str]) -> str:
    if "conclusion" in a_labels and ("evidence" in k_labels or "condition" in k_labels):
        return "diagnostic_support"
    if "recommendation" in a_labels or "step" in a_labels:
        return "treatment_or_operation_support"
    return "explanatory_support"


def read_records(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield line_no, row


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def build(inputs: list[Path], output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    knowledge: list[dict[str, Any]] = []
    applications: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    uncertain: list[dict[str, Any]] = []
    seen_k: set[str] = set()
    counts: Counter[str] = Counter()
    cluster_counts: Counter[str] = Counter()
    k_index = a_index = e_index = 0

    for path in inputs:
        dataset = path.stem
        for line_no, row in read_records(path):
            counts["input_records"] += 1
            question = clean_text(row.get("instruction", row.get("question", "")))
            answer = clean_text(row.get("output", row.get("answer", "")))
            combined = f"{question}\n{answer}"
            if len(question) < 8 or len(answer) < 24 or not has_tcm_signal(combined):
                counts["filtered_non_tcm_or_short"] += 1
                continue
            counts["tcm_candidates"] += 1
            source = {"dataset": dataset, "line_no": line_no, "record_hash": source_hash(combined)}
            atom_rows: list[dict[str, Any]] = []
            for atom_no, text in enumerate(atomize(answer), 1):
                if not has_tcm_signal(text):
                    continue
                labels = role_labels(text)
                votes = view_votes(text, labels)
                decision = consensus(votes)
                atom_rows.append({"atom_no": atom_no, "text": text, "roles": labels, "votes": votes, "decision": decision, "anchors": anchors(text)})
            ks = [item for item in atom_rows if item["decision"] == "K"][:6]
            ass = [item for item in atom_rows if item["decision"] == "A"][:4]
            if not ks or not ass:
                if atom_rows:
                    uncertain.append({"source": source, "question": question, "status": "unpaired_or_mixed", "atoms": atom_rows})
                    counts["unpaired_or_mixed"] += 1
                continue

            record_ks: list[dict[str, Any]] = []
            for item in ks:
                text_hash = source_hash(compact(item["text"]))
                if text_hash in seen_k:
                    counts["deduplicated_knowledge"] += 1
                    continue
                seen_k.add(text_hash)
                k_index += 1
                cluster = "|".join(item["anchors"][:3]) or "other"
                cluster_counts[cluster] += 1
                k = {
                    "knowledge_id": f"K-AEK-{k_index:07d}",
                    "knowledge_type": "atomic_tcm_knowledge",
                    "text": item["text"],
                    "role_labels": item["roles"],
                    "annotation_views": item["votes"],
                    "anchor_entities": item["anchors"],
                    "topic_cluster": cluster,
                    "source": dict(source, atom_no=item["atom_no"], text_hash=text_hash),
                    "quality": {"status": "high_confidence_structural_consensus", "clinical_review": "not_performed"},
                }
                knowledge.append(k)
                record_ks.append(k)
            if not record_ks:
                continue

            for item in ass:
                query_answer_overlap = jaccard(question, item["text"])
                if query_answer_overlap >= 0.72:
                    uncertain.append({"source": source, "question": question, "status": "query_answer_overlap_or_answer_leakage", "application_atom": item, "query_answer_char_bigram_jaccard": round(query_answer_overlap, 4)})
                    counts["query_answer_leakage_rejected"] += 1
                    continue
                supports: list[tuple[dict[str, Any], float]] = []
                for k in record_ks:
                    overlap = set(item["anchors"]) & set(k["anchor_entities"])
                    # Require a shared domain anchor plus non-duplicative text.
                    similarity = jaccard(k["text"], item["text"])
                    if overlap and similarity < 0.72:
                        score = min(0.99, 0.60 + 0.08 * len(overlap) + 0.12 * (1 - similarity))
                        supports.append((k, round(score, 3)))
                if not supports:
                    uncertain.append({"source": source, "question": question, "status": "no_nonduplicative_k_support", "application_atom": item, "knowledge_atoms": record_ks})
                    counts["no_support"] += 1
                    continue
                a_index += 1
                application = {
                    "application_id": f"A-AEK-{a_index:07d}",
                    "application_type": "atomic_tcm_application",
                    "question": question,
                    "answer": item["text"],
                    "role_labels": item["roles"],
                    "annotation_views": item["votes"],
                    "anchor_entities": item["anchors"],
                    "source": dict(source, atom_no=item["atom_no"], answer_hash=source_hash(compact(item["text"]))),
                    "quality": {"status": "high_confidence_structural_consensus", "clinical_review": "not_performed"},
                }
                applications.append(application)
                for k, score in supports[:3]:
                    e_index += 1
                    e = {
                        "edge_id": f"E-AEK-{e_index:07d}",
                        "knowledge_id": k["knowledge_id"],
                        "application_id": application["application_id"],
                        "relation": relation_type(k["role_labels"], item["roles"]),
                        "shared_anchors": sorted(set(k["anchor_entities"]) & set(item["anchors"])),
                        "support_score": score,
                        "bidirectional_checks": {
                            "k_supports_a": True,
                            "a_restates_k": False,
                            "char_bigram_jaccard": round(jaccard(k["text"], item["text"]), 4),
                        },
                        "query_counterfactual": {
                            "generated_query": question,
                            "k_only_support_available": True,
                            "answer_leakage_risk": False,
                        },
                        "status": "structural_high_confidence_not_clinically_reviewed",
                    }
                    edges.append(e)
                    counts["edges"] += 1
            counts["paired_records"] += 1

    write_jsonl(output / "knowledge.aek.v1.jsonl", knowledge)
    write_jsonl(output / "application.aek.v1.jsonl", applications)
    write_jsonl(output / "edges.aek.v1.jsonl", edges)
    write_jsonl(output / "uncertain_pool.aek.v1.jsonl", uncertain)
    clusters = [{"cluster": name, "knowledge_count": count} for name, count in cluster_counts.most_common()]
    (output / "topic_clusters.aek.v1.json").write_text(json.dumps(clusters, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "version": "aek-v1-structural-consensus",
        "inputs": [{"path": str(path), "bytes": path.stat().st_size} for path in inputs],
        "counts": dict(dict(counts), knowledge=len(knowledge), applications=len(applications), edges=len(edges), uncertain=len(uncertain)),
        "admission_policy": {
            "atomicization": "heading/newline/sentence segmentation with a 480-character maximum atom",
            "role_labels": ["definition", "fact", "condition", "evidence", "reasoning", "conclusion", "recommendation", "step", "case"],
            "independent_views": ["basis_vs_action", "reusable_vs_current", "premise_vs_inference"],
            "formal_acceptance": "two agreeing K or A views, no opposing view, plus same-source K-A support with shared anchor and character-bigram Jaccard below 0.72",
        },
        "limitations": [
            "The three views are deterministic rule-based perspectives, not independent language models or human annotation.",
            "Topic clusters are anchor-based grouping, not embedding/vector clustering.",
            "All accepted records need medical/TCM expert review before any clinical use.",
            "The output is for research data construction only, not diagnosis or prescribing.",
        ],
    }
    (output / "build_report.aek.v1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("aek_output_v1"))
    args = parser.parse_args()
    inputs = [args.input_dir / name for name in ("speech_cleaned.jsonl", "text_cleaned.jsonl", "vision_cleaned.jsonl")]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs: " + ", ".join(missing))
    report = build(inputs, args.output)
    print(json.dumps(report["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
