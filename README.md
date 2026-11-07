# TRACER-RAG

Research code for **Traceable Relation Evidence Retrieval (TER)** and **Evidence Critique-guided Refinement (EviCR)** for Chinese traditional medicine question answering.

[中文说明](README.zh-CN.md) · [Reproduction guide](docs/REPRODUCIBILITY.md) · [Data contracts](docs/DATA_CONTRACTS.md) · [Code map](docs/CODE_STRUCTURE.md)

## Current protocol

TER retrieves application–knowledge–evidence (A–K–E) units using lexical/dense recall and soft relation-aware reranking. EviCR reviews the frozen evidence and guides one answer revision.

```text
OFFLINE (Build only)
q + frozen D(q) -> evidence teacher -> structured evidence label
system + q + D(q) -> QLoRA Critic training target

ONLINE (every evaluation query)
q + frozen D(q) -> Critic -> structured evidence package
q + D(q) -> generator -> initial answer y0
q + D(q) + y0 + evidence package -> generator -> revised answer y1
```

**The current Critic never receives the current initial/weak answer or an answer key.** Historical application answers inside D(q) are evidence, not the current y0. The same system/user message builders are used for teacher labeling, SFT and the public online runner.

The Critic outputs `evidence_status`, `evidence_actions`, `knowledge_supplements`, `change_support`, and `answer_guidance`. Model-knowledge supplements are explicitly distinguished from retrieved support. See the validator in `generate_evicr_qd_labels.py` for exact allowed values.

## Run a small offline example

Python 3.10+; no GPU, model download, API key or medical data required:

```bash
python examples/offline_demo.py --output-dir outputs/demo
```

This creates symbolic synthetic examples, prepares Build supervision, exports q,D-only SFT, validates a training dry run, and previews online Critic prompts. Its labels are hand-constructed demo fixtures, not research results or teacher-generated medical judgments.

For automated checks:

```bash
pip install -r requirements-test.txt
python -m pytest -q
python scripts/check_release.py
```

## Reproduce the research pipeline

Install a CUDA-compatible PyTorch build appropriate to your hardware, then the remaining dependencies in `requirements.txt`. Its lower bounds are not a tested model/environment lock: choose Transformers/PEFT versions that support your exact model architecture. Full training and generation require separately obtained models and authorized data.

1. Build A–K–E data, convert an explicitly supplied split with `prepare_ter_corpus.py`, and retrieve with `run_aek_hybrid_r4.py`.
2. Select reranking weights on Dev using `run_rerank_weight_search.py`; freeze the choice before Test evaluation.
3. Prepare and label evidence-only Build samples with `prepare_evicr_qd_supervision.py` and `generate_evicr_qd_labels.py`.
4. Export SFT with `export_evicr_qd_sft.py`; train with `train_critic_lora.py`.
5. Serve the adapter/generator and run **`run_evicr.py`** on every frozen Test query. Run the no-supplements arm by reusing the identical saved Critic packages.
6. Grade frozen predictions separately, then compute paired counts, McNemar tests and cluster bootstrap intervals with `paired_statistics.py`.

Commands, required input schemas, service configuration and experiment boundaries are in the [reproduction guide](docs/REPRODUCIBILITY.md).

## Results and provenance

The release includes reviewed **aggregate Dev weight-search results**, with input hashes and selection rules, under [`reproducibility/weight_search`](reproducibility/weight_search/README.md). Source files and publication-package changes are recorded in [`SOURCE_MANIFEST.json`](reproducibility/SOURCE_MANIFEST.json); [`SHA256SUMS.txt`](SHA256SUMS.txt) binds the released files.

This is a code release, not a self-contained reproduction of every manuscript number. Private corpora, adapter weights and per-question outcomes are not bundled. Synthetic smoke tests establish software contracts; they do not establish clinical accuracy or experimental efficacy. The presence of an experiment runner does not establish that the experiment finished or that its result was positive.

## Historical code

Answer-aware Critic scripts and answer-key-selected wrong-only diagnostics remain for provenance. They are **not the default EviCR protocol**. In particular, `generate_critic_supervision.py`, `export_critic_sft.py`, `run_tcm_alignrag.py` and `run_*wrong_only*` must not be substituted for the current evidence-only pipeline. Historical wrong-only summaries are conditional error-repair diagnostics, not blind full-Test effects. [Code map](docs/CODE_STRUCTURE.md).

## Data, rights and intended use

No API credentials, patient narratives, raw question banks/answer keys, SFT records, model weights, unpublished manuscripts or referee documents are included. Data access and missing artifacts are described in [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md).

No software reuse license has been assigned; see [RIGHTS.md](RIGHTS.md). Model and dataset terms apply separately. This is research software, not a validated clinical decision tool.
