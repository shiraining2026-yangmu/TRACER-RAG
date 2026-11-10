# Field dictionary

All CSVs are UTF-8 with a header and LF line endings. Empty aggregate cells mean not reported/applicable, never zero. Identifiers link rows without redistributing questions, medical/source narratives or answer keys. Binary correctness means archived grading success (1) or failure (0); the checker cannot independently regrade absent predictions.

## Exam records

`archived_exam_correctness.csv`:

| Field | Meaning |
| --- | --- |
| `application_id` | Stable `QB-YYYY-TCM-UU-NNN` exam identifier. |
| `year` | `2021` or `2022`. |
| `unit` | Original unit enum `第一`, `第二`, `第三`, `第四`. |
| `shared_option_group` | Original question-number range or blank. Qualify by year and unit; blank rows were treated as singletons in archived statistics. |
| `<model>__dense`, `<model>__ter`, `<model>__evicr` | Archived 0/1 grading. Eight model slugs and display names are in `scripts/check_supporting_data.py`. Historical routing and intermediate Critic stages are described in `archived_exam_protocol.json`. |

The two `archived_paired_statistics_*.csv` tables use the same schema:

| Fields | Meaning |
| --- | --- |
| `model`, `year`, `comparison` | Display model name; year or `combined`; `dense_to_ter`, `ter_to_evicr`, `dense_to_evicr`. |
| `n`, `baseline_correct`, `candidate_correct` | Paired denominator and marginal correct counts. |
| `both_correct`, `baseline_correct_candidate_wrong`, `baseline_wrong_candidate_correct`, `both_wrong` | Paired 2×2 cell counts. |
| `gain_minus_loss_questions`, `paired_difference_percentage_points` | Corrected minus regressed items; that difference divided by n × 100. |
| `independent_question_mcnemar_n`, `independent_question_mcnemar_gains`, `independent_question_mcnemar_losses` | Counts among rows with blank shared-option group. The historical field name does not establish actual independence. |
| `independent_question_mcnemar_exact_two_sided_p`, `independent_question_mcnemar_holm_adjusted_p` | Archived singleton-only exact test and its multiplicity adjustment. |
| `all_question_mcnemar_unclustered_sensitivity_only_p` | Unclustered exact test, sensitivity statistic only. |
| `question_bootstrap_ci95_percentage_points`, `cluster_bootstrap_ci95_percentage_points` | Archived JSON-encoded `[lower, upper]` intervals for the paired accuracy difference. |
| `cluster_count` | Number of year/unit-qualified option clusters plus singleton IDs. |

`archived_no_rag_vs_dense.csv` contains `application_id`, `year`, `no_rag_correct`, `dense_correct`, from an earlier run. Only rows with valid reference status and jointly available binary scores were exported.

## Automated assessments

`structure_assessments.csv`:

| Field | Values / meaning |
| --- | --- |
| `application_id`, `arm` | Stable `A-AEK-NNNNNNN` query ID and `ter_full`/`ter_flat`. Every ID has both arms. |
| `status` | `valid`, the archived format-validation status. |
| `evidence_support` | `supported`, `partial`, `unsupported`; strict support counts only `supported`. |
| `historical_case_misuse`, `key_condition_omission`, `unsupported_expansion` | `yes` or `no` from the automated assessor. |
| `candidate_sha256` | Archived SHA-256 of the assessed candidate; candidate text is omitted. |

`open_qa_ai_scores.csv`:

| Field | Values / meaning |
| --- | --- |
| `row_id` | Stable numeric source-question ID; one row per candidate model. |
| `model` | `strict-qwen35-generator` (candidate A) or `strict-mistral-generator` (candidate B). These are serving aliases, not exact model revisions. |
| `correctness`, `completeness`, `safety` | Scores on [0,2], averaged across two automated judges; fractional scores are retained. These are ordinal exploratory ratings. |
| `composite_mean_0_to_2` | Arithmetic mean of the three dimension scores. |
| `judge_a_preference`, `judge_b_preference` | `A`, `B`, `tie`, `neither`; repeated on both candidate rows. Judge A is `strict-qwen35-untuned`; judge B is `strict-mistral-generator`. |

The structure and open-QA JSON protocol files retain the relevant judge limitations and archived summaries. The 111-query K-only/TER JSON is an aggregate-only automated assessment, separate from structure flattening.

## Other aggregates and membership

`strict_ablation_aggregates.csv`:

| Fields | Meaning |
| --- | --- |
| `model`, `arm`, `year` | `qwen35`/`mistral`; `dense`, `ter_y0`, `tuned_critic`, `evicr_tuned`; year or `combined`. |
| `correct`, `scored`, `accuracy_percent` | Archived marginal outcomes. Scored denominators: 554, 583, combined 1,137. |
| `y0_gains`, `y0_losses`, `y0_delta_accuracy_pp`, `y0_ci95_low`, `y0_ci95_high`, `y0_mcnemar_singletons_p` | Paired changes and archived inference relative to `ter_y0`. Baseline rows retain historical placeholders. |
| Corresponding `y1_*` fields | Relative to the answer-aware `tuned_critic` stage, when reported. |
| `answer_prompt_tokens`, `answer_completion_tokens`, `critic_prompt_tokens`, `critic_completion_tokens` | Archived reported token counts for that arm. Historical zero values need not imply a cost-free end-to-end pipeline. No dollar cost is imputed. |

`evaluation_query_membership.csv` contains `application_id`, `split` (`build`, `dev`, `test`), and `relation_types`. `critic_seed_build_membership.csv` adds `sample_id` and is Build-only. Relation types are sorted, pipe-separated members of `diagnostic_support`, `explanatory_support`, `treatment_or_operation_support`. Membership is not source text, a training label or a fitted adapter.

`corpus_snapshot.json` distinguishes archived input hashes from freshly checked hashes of the archived baseline-input files. Those are different file representations; do not compare their hashes as though they were the same bytes. `external_transfer_aggregates.json` retains original dataset URLs/commits and aggregate metrics only. `numeric_summary.json` is computed by the released checker. `PROVENANCE.json` records per-export source hashes and transformations; source rights are not inferred from those hashes.
