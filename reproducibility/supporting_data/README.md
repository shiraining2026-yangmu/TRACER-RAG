# Numerical supporting data

Released 2026-11-10 from preserved local research records. These are actual archived numerical records and metadata, not synthetic examples or newly generated experiments. [中文说明](#中文说明) · [Field dictionary](DATA_DICTIONARY.md) · [Source hashes and export rules](PROVENANCE.json)

## Contents and interpretation

| Files | Size / coverage | What they support |
| --- | --- | --- |
| `archived_exam_correctness.csv` | 1,138 scored questions × 8 models × 3 arms | Binary correctness, year/unit and explicit shared-option groups; no answer keys or predictions. |
| `archived_paired_statistics_primary.csv`, `archived_paired_statistics_sensitivity.csv` | 72 comparisons each | Primary 1,137-item analysis excluding known training overlap; all-1,138 sensitivity analysis. Counts and exact tests can be checked against the matrix. |
| `archived_exam_protocol.json` | Eight model histories | The historical EviCR arm used answer-key-selected wrong-only routing and retained correct upstream answers. **These are conditional repair diagnostics, not blind full-Test deployment effects.** Qwen3.5 also had an answer-aware Critic stage. |
| `structure_assessments.csv`, `structure_protocol.json` | 111 queries × 2 arms = 222 assessments | Frozen ordered Top-3 text, with versus without relation labels and A/K boundaries. Categorical ratings come from an automated assessor of the same model family as the generator, not independent experts. |
| `open_qa_ai_scores.csv`, `open_qa_ai_protocol.json` | 300 questions × 2 candidates = 600 rows | MedQuAD scores averaged over two reciprocal AI judges. The judges are also candidate models; self-preference and pretraining exposure remain possible. No clinical validation is implied. |
| `strict_ablation_aggregates.csv`, `strict_ablation_protocols.json` | Two models, four arms, yearly/combined = 24 rows | Archived 1,137-item tables, including negative changes. Despite the historical directory name, `evicr_tuned` uses wrong-y1-only routing and retains correct y1. These tables do not establish a blind q,D-only ablation. |
| `external_transfer_aggregates.json` | Two generators; CMExam 500 and MedQuAD 300 each | Pinned benchmark repositories, aggregate transfer accuracy/overlap and usage. Direct generation without a frozen evidence corpus; evidence fidelity was not measured. |
| `test111_konly_vs_ter_aggregates.json` | 111 Test queries | Archived automated K-only versus relation-evidence assessment: support-pass 15 versus 45, unsupported-expansion rates 18.92% versus 23.42%. Only aggregate judgments were recovered locally. This is a different comparison from the structure ablation. |
| `corpus_snapshot.json`, `evaluation_query_membership.csv` | Corpus counts/hashes; 345 query IDs | Build/Dev/Test query membership 119/115/111. Snapshot: 44,677 applications, 142,089 knowledge items and 61,749 Build edges. No A/K source text is included. |
| `critic_seed_build_membership.csv` | 3,000 Build seed records | Seed-selection membership and relation types, not teacher labels, SFT text or proof of the exact data used for any released adapter. |
| `archived_no_rag_vs_dense.csv` | 1,138 jointly scored questions | Earlier Qwen baseline: 916 no-RAG versus 908 dense correct. This is a different archived run from the revision matrix's Qwen3.5 929/930/966; do not combine these as a single experiment. |
| `numeric_summary.json` | Recomputed descriptive values | Deterministic counts and mean scores derived from these CSVs. No new model inference or confidence intervals were generated. |

The revision matrix has 554/584 scored items for 2021/2022. The primary paired and strict-ablation analyses exclude `QB-2022-TCM-03-080`, a known Critic-training overlap, leaving 554/583. Excluding it from scoring does not demonstrate that the adapter was retrained without the overlap. All-1,138 numbers are preserved as sensitivity records.

Explicit option groups are qualified by **year and unit** when clustering. The archived matrix does not establish that every shared clinical stem or source dependence has been captured. Published intervals are archived estimates from their recorded procedures, not proof of independent observations.

## Verify without private data or models

From the repository root, using Python 3.10+ and only the standard library:

```bash
python scripts/check_supporting_data.py
```

This validates the nine CSV schemas, export hashes, all 144 paired count/test records, structure-rating rates, reciprocal AI mean scores and seed/evaluation boundaries. It also checks the saved descriptive summary. Archived confidence intervals and Holm-adjusted values are retained; the checker does not regenerate them.

To obtain correctness-only pairs for a **new** bootstrap calculation:

```bash
python scripts/check_supporting_data.py --paired-output outputs/qwen35-archived-pairs.csv --model qwen35_9b --before ter --after evicr
python paired_statistics.py --input outputs/qwen35-archived-pairs.csv --output outputs/qwen35-archived-statistics.json --draws 10000
```

The default pair export excludes the known overlap (1,137 rows). Add `--all-items` for the 1,138-item sensitivity set. New bootstrap intervals may differ from historical intervals because the implementation/random-number generator differs; they do not change the conditional-routing limitation. Correctness-only data cannot verify grading, prediction provenance or evidence fidelity without the underlying authorized artifacts.

## Availability boundary

This package makes selected numerical findings checkable. Raw corpora, clinical/source narratives, question/answer texts, answer keys, teacher/SFT labels, model weights/adapters, exact historical serving environments and independent expert annotations are not provided. Some supporting records are aggregates only. See [DATA_AVAILABILITY.md](../../DATA_AVAILABILITY.md) for the full scope and [RIGHTS.md](../../RIGHTS.md) for licensing.

`PROVENANCE.json` records original project-relative locations and byte hashes, exported hashes and field transformations. Source paths describe local provenance and are not public download links. Original private files were preserved. `SHA256SUMS.txt` in the repository root covers this package, including its documentation and provenance.

## 中文说明

本目录补充公开真实历史记录中的逐题正确性、自动评估分数、固定查询划分、Build 种子成员、语料规模和来源哈希，并提供无需模型和私有数据的复核命令。

需结合各文件的历史协议理解结果：错题由答案键筛选的 EviCR 属于条件性修复诊断；自动模型评分不等于专家评价；1,137 题与 1,138 题版本不能混用；旧 Qwen 基线与修订矩阵是不同运行。本目录没有补造缺失的实验、专家评分或训练权重，也不构成全部论文结果的一键复现包。
