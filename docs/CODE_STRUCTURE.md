# Code map

## Current evidence-only protocol

| Stage | Entry points |
|---|---|
| A–K–E construction / split conversion / TER retrieval | `build_aek_from_cleaned.py`, `prepare_ter_corpus.py`, `run_aek_hybrid_r4.py` |
| Dev weight search | `run_rerank_weight_search.py` (archived R4 experiment contract) |
| q,D-only supervision | `prepare_evicr_qd_supervision.py`, `generate_evicr_qd_labels.py`, `export_evicr_qd_sft.py` |
| Adapter training / service | `train_critic_lora.py`, `serve_qwen_critic_adapter.py`, `serve_single_generator.py` |
| Full-set EviCR / controlled no-supplements | **`run_evicr.py`** (arbitrary set size, exact SFT prompt builders) |
| Original 111-query runner / fidelity judge | `run_test111_evicr_qd_ablation.py`, `evaluate_evicr_qd_fidelity.py` (explicit 111-item contract) |
| TER representation ablations | `run_test111_ter_structure_ablation.py` |
| Paired exam statistics | `paired_statistics.py` (correctness-only CSV, whole-cluster bootstrap) |
| Offline demo / checks | `examples/offline_demo.py`, `tests/`, `scripts/check_release.py` |

## Baselines and evaluation utilities

`run_questionbank_no_rag.py`, `run_dense_chunk_rag.py`, `run_frozen_evidence_generation.py`, the `grade_*.py` programs, `run_external_evaluation.py`, `summarize_external_evaluation.py`, `score_open_qa_ai_judge.py` and `combine_open_qa_ai_judges.py` support baseline inference and separate grading. `prepare_questionbank_input.py`, `screen_medical_qa.py` and `finalize_medical_qa_screen.py` support preprocessing.

`benchmark_integrity.py` contains ID, source-hash and merge checks. `download_benchmark_model.py`, `download_dense_rag_sources.py` and the preflight utilities are optional deployment helpers; they do not grant dataset/model permissions.

## Historical: answer-aware Critic

`prepare_critic_seed_set.py` → `materialize_critic_retrieval.py` → `generate_critic_supervision.py` → `export_critic_sft.py`, and `run_tcm_alignrag.py`, `run_questionbank_rag_vs_alignrag.py`, `run_qwen35_answer_refinement.py`, `run_remote_critic_pipeline.sh` use or orchestrate the older answer-aware task. **They are not the current q,D-only training/inference path.**

## Historical: answer-key-selected wrong-only diagnostics

`run_evidence_refinement_wrong_questions.py`, `run_three_model_wrong_only*.py`, `preflight_wrong_only*.py`, `reuse_wrong_only_baselines.py`, `replace_targeted_inference_rows.py`, and `summarize_ter_wrong_only*.py` preserve a conditional error-repair workflow. Correct answers were retained using an offline answer key. These diagnostics do not estimate an unbiased full-Test effect. `memory_v2/v3` denotes deployment variants, not independent efficacy evidence.

`run_three_model_qwen35_replay.py`, `reuse_qwen35_baselines.py`, `summarize_qwen35_replay.py`, `run_multimodel_exam_benchmark.sh` and related summaries preserve older experiment orchestration. Consult their protocol before reuse.

Historical Chinese method notes and audit notes in this directory retain their original context; the current contract is in `DATA_CONTRACTS.md` and the top-level READMEs.
