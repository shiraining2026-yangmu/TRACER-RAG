# Reproduction guide

The offline synthetic demo is self-contained. Full research experiments require the authorized private artifacts listed in `DATA_AVAILABILITY.md`; the archived study split and adapter cannot be reconstructed from this release alone. Commands below use example local paths and must be run from the repository root. Every output/index directory must be fresh.

## 1. Environment and offline checks

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
python examples/offline_demo.py --output-dir outputs/demo
```

The demo validates data and prompt contracts without tokenizing with a real model, training, or calling an API. Model deployment tests are skipped when serving dependencies are absent; the private 3,000-record dataset test is skipped when that dataset is absent.

For GPU runs, first install the correct CUDA/PyTorch build, then `requirements.txt`. Select versions supporting the exact generator/Critic architecture; record `pip freeze`, GPU/CUDA details, model revision, tokenizer/chat template, quantization and adapter hashes. The repository's lower bounds do not certify Qwen3.5 compatibility with every Transformers version. Historical shell files have example `/home/ubuntu/tcm_rag` defaults; override paths for your machine.

## 2. TER construction and split conversion

```bash
python build_aek_from_cleaned.py --input-dir data/cleaned --output outputs/aek
python prepare_ter_corpus.py --applications outputs/aek/application.aek.v1.jsonl --knowledge outputs/aek/knowledge.aek.v1.jsonl --edges outputs/aek/edges.aek.v1.jsonl --splits data/splits.csv --output-dir outputs/corpus
```

Supply `data/splits.csv` with `application_id,split`, one row per application. Freeze source-group-aware splits before tuning, and curate duplicates/contamination externally. The converter enforces exact normalized-question separation; it does not invent the original assignment or reproduce its query subsampling. If using an already released R4 corpus, retain its original IDs and split manifest instead. See `DATA_CONTRACTS.md` for the input schema.

```bash
python run_aek_hybrid_r4.py --queries outputs/corpus/queries.jsonl --build-edges outputs/corpus/build_edges.jsonl --applications outputs/corpus/applications.jsonl --knowledge outputs/corpus/knowledge.jsonl --embedding-model models/bge-m3 --index-dir indexes/build --run-dir outputs/dev-retrieval --split dev --phase retrieve --rerank-weights 0.48 0.35 0.10 0.07 0.05 0.05
```

This creates the index and a retrieval-only `generation_results.jsonl`; it does not call the generator. The lexical and dense scores are min–max normalized per candidate union (constant scores map to 1), local overlaps are character-bigram Jaccard scores, and relation intent is keyword-count/max-count. All six are bounded in [0,1], though they are not identically distributed.

## 3. Dev selection, then frozen Test

```bash
python run_rerank_weight_search.py --queries data/frozen-r4-queries.jsonl --index-dir indexes/build --embedding-model models/bge-m3 --run-dir outputs/weight-search --grid-step 0.05 --bootstrap-replicates 10000
```

This script retains the **archived R4 experiment's validation contract**: both Dev and Test are required, and the legacy Test baseline must reproduce 38/111 Recall@3 and its saved MRR. It is not a generic tuner for arbitrary new datasets. It chooses Dev MRR@3, then Recall@3, then minimum L1 distance to the normalized legacy vector; Test feature evaluation occurs after the choice. The saved search has 115 Dev queries, 111 Test queries and 15,982 candidates. Details and the saved vector are in `reproducibility/weight_search/README.md`.

For the archived selected vector, pass `--rerank-weights 0.10 0.15 0.30 0.05 0.35 0.05` explicitly to `run_aek_hybrid_r4.py`. Its defaults preserve the legacy vector to avoid silently relabeling historical runs. The script refuses an existing index directory; use a fresh directory when materializing another retrieval run. Save its manifest/hashes. Do not choose a setting from the Test sensitivity table.

## 4. Build-only evidence supervision and QLoRA

Create Build retrieval with the same fixed configuration (`--split build --phase retrieve`, fresh index/run directories). Queries must carry `sample_id`. Then:

```bash
python prepare_evicr_qd_supervision.py --source data/build-retrieval.jsonl --output-dir outputs/qd-seeds --field-limit 850
python generate_evicr_qd_labels.py --seeds outputs/qd-seeds/build_qd_seeds.jsonl --output-dir outputs/qd-labels --api-base https://YOUR_AUTHORIZED_ENDPOINT/v1 --model YOUR_TEACHER_MODEL --api-key-env TEACHER_API_KEY
python export_evicr_qd_sft.py --labels outputs/qd-labels/teacher_qd_labels.jsonl --output-dir outputs/qd-sft
python train_critic_lora.py --model-name-or-path models/critic-base --train-file outputs/qd-sft/critic_sft_train.jsonl --validation-file outputs/qd-sft/critic_sft_validation.jsonl --output-dir outputs/qd-adapter --dry-run
```

Set `TEACHER_API_KEY` through your shell/secret manager; never put it in code or a committed command. Only send data to an endpoint authorized for that data. The teacher and student receive q,D only. Before training, remove overlap with external evaluation questions and rebuild supervision; Build-only filtering does not by itself detect all such overlap.

After validation, remove `--dry-run` to actually train (GPU and full model dependencies required). The default trainer uses NF4 4-bit double quantization, LoRA r=32/alpha=64/dropout=.05, two epochs, learning rate 2e-4, maximum sequence length 4096, batch size 1 and accumulation 16. Overlength samples are dropped and counted. Check the actual manifest for every run; defaults are not evidence that a specific paper adapter used those settings.

## 5. Services and full-set EviCR

In separate terminals, on a machine with sufficient GPU memory:

```bash
python serve_qwen_critic_adapter.py --model models/critic-base --adapter outputs/qd-adapter --generator-model-id critic-base-unadapted --critic-model-id evicr-critic --port 8001
python serve_single_generator.py --model-path models/answer-model --model-id generator --port 8002
```

The first service exposes the adapter-enabled Critic ID and adapter-disabled base ID. Both use the same base weights. Keep services on localhost unless separately secured.

Prepare answer-free `data/frozen-test.jsonl` and a separate `data/ter-y0.jsonl` containing TER initial answers with matching IDs/questions. Generate y0 with a fixed evidence-constrained baseline and retain its exact prompt/configuration. Do not copy answer keys or current answers into frozen evidence.

```bash
python run_evicr.py --frozen-test data/frozen-test.jsonl --ter-y0 data/ter-y0.jsonl --output-dir outputs/preview --dry-run
python run_evicr.py --frozen-test data/frozen-test.jsonl --ter-y0 data/ter-y0.jsonl --output-dir outputs/full --critic-model evicr-critic
python run_evicr.py --frozen-test data/frozen-test.jsonl --ter-y0 data/ter-y0.jsonl --output-dir outputs/no-ks --no-knowledge-supplements --reuse-critic outputs/full/critic_qd.jsonl
python run_evicr.py --frozen-test data/frozen-test.jsonl --ter-y0 data/ter-y0.jsonl --output-dir outputs/unfinetuned --critic-model critic-base-unadapted
```

The unfine-tuned arm uses identical evidence, schema, prompt builders, budgets and answer-revision prompt. The no-KS arm reuses the identical Critic packages, clearing only the supplement list at delivery. All rows are processed; the runner has no grading/answer-key input. It fails rather than retaining a known-correct answer through key-based routing. It does not claim resume support; use a fresh directory for a rerun and retain failed manifests.

## 6. Evidence fidelity and paired exam statistics

For the original fixed 111-query dataset only:

```bash
python evaluate_evicr_qd_fidelity.py --frozen-test data/frozen-test.jsonl --ter-y0 data/ter-y0.jsonl --evicr-full outputs/full/evicr_full.jsonl --evicr-no-ks outputs/no-ks/evicr_no_ks.jsonl --output-dir outputs/fidelity --api-base https://YOUR_AUTHORIZED_ENDPOINT/v1 --model YOUR_INDEPENDENT_JUDGE --api-key-env JUDGE_API_KEY --bootstrap-draws 10000
```

The automated judge sees anonymized arm labels and permuted answer order, but this is **model judging, not clinician blind review**. Use an independent judge, record model/version/configuration and adjudicate clinical conclusions separately. This evaluator requires exactly 111 Test IDs; the public inference runner itself accepts arbitrary sizes. It reports question-level paired bootstrap intervals, not case-cluster intervals.

After freezing exam predictions, grade separately and prepare a correctness-only CSV with `id,year,before_correct,after_correct,cluster_id`. Keep a shared `cluster_id` for questions derived from the same case/options:

```bash
python paired_statistics.py --input data/qwen35-paired-correctness.csv --output outputs/qwen35-paired-stats.json --draws 10000
```

This reports combined and per-year paired 2×2 tables, correction/regression counts and whole-cluster bootstrap intervals. Ordinary all-item McNemar assumes independent pairs and is a sensitivity statistic for clustered questions. See `DATA_CONTRACTS.md` for the singleton analysis. Use actual paired predictions; aggregate accuracy differences cannot reconstruct the discordant counts.

## 7. Release integrity

`SOURCE_MANIFEST.json` distinguishes original scripts from publication utilities and modifications. `SHA256SUMS.txt` covers all tracked content except itself; use `python scripts/check_release.py` to validate syntax, selected secret patterns and checksums. A static scan cannot certify data permissions or detect every possible sensitive value.
