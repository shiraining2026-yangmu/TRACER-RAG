#!/usr/bin/env python3
"""Record a bounded editorial content audit after rule-based subject screening.

Selections below were reviewed from source questions/answers, not exam keys or
model scores. These are question seeds, NOT corrected answers or validated K.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from run_dense_chunk_rag import file_hash

SEEDS = [
    ("当归的功效与应用", [14973]),
    ("肾阴虚与肾阳虚的概念和辨证区别", [26254, 312080, 444012]),
    ("逍遥方的组成、功效与适用证候", [37139]),
    ("山茱萸的功效与应用", [45551]),
    ("党参与人参的功效异同及替代边界", [88069]),
    ("二陈汤的组成、功效与主治", [96803]),
    ("金匮肾气丸的功效与主治", [172285]),
    ("西洋参的功效与适用证候", [214532]),
    ("茯苓的功效及其在补气方中的配伍意义", [229552]),
    ("六味地黄丸与知柏地黄丸的组成和功效区别", [266648]),
    ("薄荷的功效与应用", [281253, 300604]),
    ("沉细脉、弱脉的脉象特点与鉴别", [319599]),
    ("四君子汤的组成、功效与主治", [357768]),
    ("生杜仲与盐杜仲的炮制和应用区别", [390223]),
    ("阴虚与阳虚的基本概念", [501011]),
]
ANSWER_NOTES = {
    14973: "个体痛经咨询；原回答的中药功效与适用性需用教材核验，不直接转成通用知识。",
    26254: "夹杂个人性功能咨询，症状列表须核验；仅保留知识问题。",
    37139: "把逍遥丸主要成分写为枸杞子、淫羊藿、川牛膝等，与教材常见逍遥方组成不符。",
    45551: "山茱萸功效被泛化为改善疲劳，缺少教材意义上的完整知识依据。",
    88069: "党参、人参的核心功效表述及湿热时适用性需要核验，不能直接作替代用药依据。",
    96803: "二陈汤组成被写为黄连、黄芩、大黄、甘草、生姜、石膏、泽泻和桔梗等，与经典组成不符。",
    172285: "保留金匮肾气丸的知识问题；原回答为个体适用性咨询，尚无逐条来源。",
    214532: "西洋参预防疾病、改善肝肾功能、适合各年龄等泛化主张缺乏可追溯依据。",
    229552: "以增强肌肉组织保水能力解释茯苓补气，缺乏来源；不保留此机制为事实。",
    266648: "六味地黄丸及知柏地黄丸的组成均出现与教材标准组成不符的药味。",
    281253: "为茶饮搭配建议，不能替代教材中药功效与适用证候。",
    300604: "末尾停在‘但是,如果你有’，答案截断；多项功效还需核验。",
    312080: "保留辨证区别问题；食疗和症状归属需核验，不整条纳入 K。",
    319599: "沉细、沉弱的描述混入紧、缓等特征，不能直接作为规范脉象定义。",
    357768: "将四君子汤中的甘草写成细辛；原回答排除。",
    390223: "将盐杜仲制法描述为盐水浸泡后发酵；不能作为盐炙杜仲知识。",
    444012: "阳虚症状列表混入口臭、舌苔黄等表现，需重新查证，原答案不作通用定义。",
    501011: "仅摘取开头的基本概念句作为未核验片段候选；症状清单不纳入。",
}


def load(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write(path, rows):
    with path.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def finalize(directory):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    priority = load(directory / "priority_candidates.jsonl")
    educational = load(directory / "educational_review.jsonl")
    records = priority + load(directory / "review_required.jsonl")
    by_line = {r["source_line"]: r for r in records}
    selected_lines = {line for _, lines in SEEDS for line in lines}
    assert selected_lines == set(ANSWER_NOTES)
    assert selected_lines <= set(by_line)
    seeds = []
    for index, (question, lines) in enumerate(SEEDS, 1):
        seeds.append({"seed_id": f"TCM-TOPIC-{index:03d}", "knowledge_question": question,
                      "question_origin": "editorial_rephrasing_of_source_questions_not_an_exam_answer",
                      "source_records": [{"record_id": by_line[n]["record_id"], "source_line": n,
                                          "original_question": by_line[n]["question"]} for n in lines],
                      "reference_answer": None, "status": "requires_authoritative_source_reconstruction",
                      "eligible_for_index": False, "eligible_for_sft": False})
    write(directory / "考点问题种子_15项.jsonl", seeds)
    original_audit = []
    for n in sorted(selected_lines):
        r = by_line[n]
        original_audit.append({"source_line": n, "record_id": r["record_id"],
                               "question": r["question"], "original_answer": r["answer"],
                               "review_note": ANSWER_NOTES[n],
                               "review_scope": "content_triage_by_assistant_not_expert_or_external_reference_validation",
                               "whole_answer_accepted": False, "eligible_for_index": False})
    write(directory / "原答案内容复核_18条.jsonl", original_audit)
    selected_audit = []
    for r in sorted(priority + educational, key=lambda r: r["source_line"]):
        selected_audit.append({"record_id": r["record_id"], "source_line": r["source_line"],
                               "question": r["question"],
                               "question_triage": "topic_seed_only" if r["source_line"] in selected_lines else "not_selected_for_core_knowledge",
                               "triage_basis": "generalizable_core_TCM_question_content_only_no_exam_scores",
                               "answer_reviewed": r["source_line"] in selected_lines,
                               "whole_answer_accepted": False})
    write(directory / "通用候选问题复核_78条.jsonl", selected_audit)
    r = by_line[501011]
    text = r["answer"].split(" 阴虚的症状包括:", 1)[0]
    assert text and text in r["answer"] and "阳虚则是指" in text
    start = r["answer"].index(text)
    fragments = [{"fragment_id": r["record_id"] + "-concept", "text": text,
                  "source_record_id": r["record_id"], "source_line": 501011,
                  "source": r["source"], "normalized_answer_char_start": start,
                  "normalized_answer_char_end_exclusive": start + len(text),
                  "factual_verification": "not_verified_against_authoritative_reference",
                  "review_scope": "basic_definition_candidate_only_symptom_lists_excluded",
                  "eligible_for_index": False, "eligible_for_sft": False}]
    write(directory / "基础概念片段候选_1条.jsonl", fragments)
    manifest = {"rule_summary": summary, "editorial_question_candidates_reviewed": len(selected_audit),
                "source_answers_reviewed_in_full": len(original_audit), "topic_seeds": len(seeds),
                "basic_concept_fragment_candidates": len(fragments),
                "verified_ready_for_index": 0, "verified_ready_for_sft": 0,
                "source_answers_not_rewritten": True,
                "input_sha256": {name: file_hash(directory / name) for name in [
                    "contract.json", "summary.json", "priority_candidates.jsonl", "review_required.jsonl", "educational_review.jsonl"]},
                "review_script_sha256": file_hash(Path(__file__)),
                "limitations": ["Conservative screening may miss relevant or correct records",
                                "No whole-corpus factual audit or external reference validation",
                                "No cross-dedup with the remote old corpus",
                                "No retrieval index, training set, or previous result was modified"]}
    (directory / "final_review_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = [
        "# 54 万条医学问答筛选报告", "",
        "本轮处理的是 chinese-medical.zip 中的 medicalQA.json。原始压缩包保持不变。",
        "目标是找出中医考试学科相关、可复用的知识候选；不以试卷错题、标准答案或准确率决定保留项。", "",
        "## 筛选结果", "",
        "| 项目 | 数量 | 含义 |", "|---|---:|---|",
        f"| 原始问答 | {summary['counts']['source_records']:,} | 全量读取 |",
        f"| 规则排除 | {summary['counts']['initial_excluded']:,} | 非特定中医内容、泛化回复、明显风险信号等 |",
        f"| 保留待核验 | {len(records):,} | 有相关学科信号，不表示正确 |",
        f"| 其中：一般咨询/其他待核验 | {summary['consultation_review']:,} | 未纳入通用考点候选 |",
        f"| 其中：通用问答规则候选 | {len(selected_audit)} | 3 条较少规则风险、75 条仍需核验；已逐条查看问题 |",
        f"| 另行阅读全文复核的源回答 | {len(original_audit)} | 包括从咨询池补选的核心知识问题 |",
        f"| 整理的考点问题种子 | {len(seeds)} | 合并同一知识主题；只有问题，没有可靠参考答案 |",
        "| 基本概念片段候选 | 1 | 只保留原文定义句，仍缺权威来源核验 |",
        "| 已验证、可直接入库或训练 | 0 | 本轮不伪称完成医学核验 |", "",
        "上述待核验总数 = 9,702 + 78 = 9,780；不是 9,780 条合格 K 或 SFT。",
        "18 条源回答不是统计抽样，其结果不能用于估计整个数据集的错误率。",
        "保守规则会漏掉一部分有用内容，不能据此声称其余所有记录都错误。", "",
        "## 内容复核发现", "",
        "- 第 357,768 行：四君子汤中将甘草写成细辛。",
        "- 第 96,803 行：二陈汤的组成被写成黄连、黄芩、大黄等药味。",
        "- 第 266,648 行：六味地黄丸和知柏地黄丸的组成存在明显混淆。",
        "- 第 390,223 行：盐杜仲被写为盐水浸泡后发酵制成。",
        "- 第 229,552 行：用增强组织保水能力来解释茯苓补气，缺少可靠依据。",
        "- 第 300,604 行：回答在‘但是,如果你有’处截断。", "",
        "原回答未被自动修正。错误或可疑原答案保留在复核文件中作为溯源记录，不进入知识索引。", "",
        "## 整理出的知识问题", "",
    ]
    for seed in seeds:
        report.append(f"- {seed['knowledge_question']}（源行：" + "、".join(str(r["source_line"]) for r in seed["source_records"]) + "）。")
    report.extend(["", "## 后续使用边界", "",
                   "这些问题可用于从教材、药典等有出处材料重建知识，而不能沿用原数据的无来源回答。",
                   "方剂 Excel 中同名异方很多，也不能用任意同名记录给这里的标准方答案背书。",
                   "本轮没有调用远端模型或付费 API，没有训练、修改 R4/K-only 索引或重跑考试。",
                   "冻结试卷的题干仅用于重合排除；候选中未触发该文本重合规则，不等于排除了全部改写题。",
                   "与旧语料的交叉去重尚未执行。以后比较知识库增益时，各 RAG 组须共享新增来源和排除规则。", "",
                   "## 文件", "",
                   "- 考点问题种子_15项.jsonl：15 个问题主题及原问题行号，无伪造参考答案。",
                   "- 原答案内容复核_18条.jsonl：原回答全文及不直接接纳的原因。",
                   "- 通用候选问题复核_78条.jsonl：每条问题的取舍及复核范围。",
                   "- 基础概念片段候选_1条.jsonl：定义原文及字符偏移，明确不可直接入库。",
                   "- review_required.jsonl：9,777 条有规则风险的记录；priority_candidates.jsonl：3 条规则优先候选。",
                   "- contract.json / summary.json / final_review_manifest.json：来源、规则、指纹及统计。", ""])
    (directory / "筛选报告.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ["editorial_question_candidates_reviewed", "source_answers_reviewed_in_full",
                                              "topic_seeds", "basic_concept_fragment_candidates", "verified_ready_for_index"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    finalize(parser.parse_args().directory)
