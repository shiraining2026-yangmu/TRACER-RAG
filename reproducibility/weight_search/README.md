# Archived Dev weight selection

These aggregate files are copied from the saved R4 weight-search experiment identified by `summary.json` input hashes. They contain metrics and weights, without questions, answers, individual predictions or evidence text. No experiments were rerun merely to prepare this repository.

Selection used **115 Dev queries**, MRR@3 as the primary criterion, Recall@3 as the secondary criterion, and minimum L1 distance to the normalized legacy vector to break remaining ties. The candidate set has **15,982 vectors**, with grid step 0.05, BM25 >= 0.10, dense >= 0.10 and relation-intent <= 0.20, plus the normalized legacy vector. The frozen Test has **111 queries** and is evaluated after selection.

| Feature | Legacy raw weight | Dev-selected weight |
|---|---:|---:|
| BM25 | 0.48 | 0.10 |
| Dense | 0.35 | 0.15 |
| Question overlap | 0.10 | 0.30 |
| Historical-answer overlap | 0.07 | 0.05 |
| Knowledge overlap | 0.05 | 0.35 |
| Relation intent | 0.05 | 0.05 |

The legacy raw vector sums to 1.10 and is normalized in the saved search; common positive scaling does not change a linear reranking order. Historical-answer overlap means the retrieved application's answer, not the current y0.

`dev_grid_top100.csv` lists candidate rankings. `dev_ablation_sensitivity.csv` and `test_ablation_sensitivity.csv` report feature-removal and +/-20% sensitivity settings. A zero weight removes that feature's contribution to the score; it **does not remove its recall branch from the candidate union**. In particular, `no_dense` is not a dense-recall-removal experiment.

Relevance is determined by linked knowledge IDs in the archived queries, not by independent expert annotation of clinical adequacy. These tables describe retrieval metrics only; they cannot establish EviCR accuracy or evidence fidelity. The selected vector need not win every Test comparison; do not reselect weights from Test sensitivity results. The underlying private queries/index are not redistributed, so this release cannot independently regenerate the aggregates without additional authorized artifacts.
