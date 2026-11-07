# Data contracts

All examples here are symbolic. JSONL files contain one JSON object per line; they remain local and are ignored by Git. Preserve immutable source files and hashes.

## TER corpus and splits

`build_aek_from_cleaned.py` reads `speech_cleaned.jsonl`, `text_cleaned.jsonl`, and `vision_cleaned.jsonl`. Its rule-based structural filters are not independent clinician annotation. It produces `application.aek.v1.jsonl`, `knowledge.aek.v1.jsonl`, and `edges.aek.v1.jsonl`, without recreating the study's private split assignment.

`prepare_ter_corpus.py` is a release utility that accepts those outputs plus an explicit `application_id,split` CSV. It converts `edge_id` to `e_id`, attaches `release.split` to applications, outputs Build-only edges and answer-free query rows. It rejects exact normalized-question overlap across splits. This is not a substitute for source-group/semantic deduplication, and it does not recover the archived split or 115/111 query selection.

Retrieval expects:

| File | Required fields |
|---|---|
| Applications | `application_id`, `question`, historical `answer`, `release.split` |
| Knowledge | `knowledge_id`, `text` |
| Build edges | `e_id`, `application_id`, `knowledge_id`, `relation` |
| Queries | `application_id`, `question`, `split`; `sample_id` for supervision; `evaluation_knowledge_ids` for retrieval metrics |

Relations are `diagnostic_support`, `treatment_or_operation_support`, or `explanatory_support`. Evaluation knowledge IDs derived from source links represent retrieval relevance, not independently reviewed clinical correctness. They are not formatted into the Critic prompt.

## Frozen evidence for current EviCR

```json
{
  "application_id": "TEST-SYMBOL-1",
  "question": "Does symbolic A imply B under Z?",
  "split": "test",
  "retrieved": [{"payload": {
    "e_id": "EDGE-SYMBOL-9",
    "application_id": "BUILD-SYMBOL-9",
    "knowledge_id": "KNOWLEDGE-SYMBOL-9",
    "relation": "explanatory_support",
    "application_question": "Historical symbolic question",
    "application_answer": "Historical symbolic application",
    "knowledge_text": "A implies B only under Z."
  }}]
}
```

There must be one to three evidence blocks and no self-retrieval. Each evidence field is truncated to 850 characters by default. The query's current answer and answer key must not be present in the frozen input. **Historical `application_answer` is part of D(q)**. The public runner rejects top-level current-answer/key fields, but this structural check cannot detect answer leakage hidden inside arbitrary text.

The separate initial-answer file contains `application_id`, the identical `question`, and `answer`. IDs must be unique and exactly match the frozen evidence set. Only the answer reviser receives this answer.

## Evidence-only supervision

Use Build retrieval rows with `sample_id` and `application_id`. `prepare_evicr_qd_supervision.py` copies only identifiers, question and formatted evidence into seeds, dropping old answer fields. `generate_evicr_qd_labels.py` creates the evidence labels; `export_evicr_qd_sft.py` validates them and exports messages.

| Critic output | Contract |
|---|---|
| `evidence_status` | `supported`, `partial`, `insufficient`, `conflict` |
| `evidence_actions` | Exactly one `{evidence_id, action, reason}` per E1…En; action is `keep`, `downweight`, `discard` |
| `knowledge_supplements` | Zero to two `{claim, basis, confidence}`; basis always `model_knowledge`; confidence `high`, `medium`, `low` |
| `change_support` | `none`, `weak`, `strong` |
| `answer_guidance` | Nonempty string, at most 900 characters |

No extra output fields are accepted. SFT input is `[system; q; D(q)]`, output is the JSON package; deduplication fingerprint is `sha256(q + newline + formatted_D)`. Train/validation partitioning uses a deterministic application-ID hash within Build. This partition is different from the retrieval Dev/Test split. Removing external-test overlap and source-related duplicates remains an upstream data-curation step.

## Controlled inference and statistics

`run_evicr.py` processes all rows, does not accept a grading file, uses one JSON repair attempt, and fails on a still-invalid Critic package or empty answer. Use fresh output directories. Manifests distinguish dry-run, complete and failed execution. Saved Critic packages carry prompt hashes.

For no-supplements inference, reuse the full arm's `critic_qd.jsonl`; the delivered copy changes only `knowledge_supplements` to `[]`. Keep models, y0, evidence, generation budgets and prompts otherwise fixed. For fine-tuned versus unfine-tuned Critic comparisons, use the same runner/configuration and separate Critic service/model IDs; do not reuse the fine-tuned package for the unfine-tuned arm.

`paired_statistics.py` accepts only CSV columns `id,year,before_correct,after_correct` and optional `cluster_id`. Correctness is 0/1; shared-case/option questions must use a shared cluster ID within a year. It reports the paired 2×2 table, gains/losses, exact McNemar, and a 10,000-draw whole-cluster paired bootstrap interval for the accuracy difference. When clusters exist, the all-item ordinary McNemar p value is labeled a sensitivity analysis; singleton McNemar excludes clustered items.
