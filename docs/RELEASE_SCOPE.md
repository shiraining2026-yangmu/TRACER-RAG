# Publication-package scope

Prepared 2026-11-07 from the existing local code export and research project. The project inventory contained approximately 4,200 files totaling 2.85 GiB after excluding Git metadata, node_modules and Python bytecode caches. Most of that workspace is not suitable as public repository content.

## Included and supplemented

- The pre-existing code export and original pipeline modules; the older answer-aware and wrong-only protocols are explicitly identified as historical.
- TER construction/retrieval and the saved Dev rerank search, current evidence-only label/SFT/fidelity runners, baselines and preprocessing utilities.
- New publication utilities: `prepare_ter_corpus.py` (explicit supplied split conversion), `run_evicr.py` (arbitrary-size full-set q,D-only inference with SFT message builders), `paired_statistics.py` (correctness-only paired analysis), symbolic demo and offline protocol tests.
- A configurable rerank-weight option retaining the legacy default, a correction to same-question paired bootstrap resampling in the TER ablation script, lightweight-import/test adjustments, documentation and CI.
- Reviewed aggregate weight-search JSON/CSV files. No query/evidence/patient text accompanies them.
- Added 2026-11-10: reviewed numerical supporting records in `reproducibility/supporting_data`, including the 1,138-item/eight-model correctness matrix, primary/sensitivity paired statistics, 222 structure-assessment rows, 600 reciprocal AI-score rows, 24 strict-ablation aggregate rows, external-transfer and Test111 aggregate metadata, 345 query and 3,000 Build-seed membership rows, and corpus counts/hashes. A field dictionary, per-export provenance and `scripts/check_supporting_data.py` accompany the records.

`reproducibility/SOURCE_MANIFEST.json` records relative source locations and hashes, distinguishing source copies from modified or new release files. `SHA256SUMS.txt` covers the final public content. No absolute workstation paths or credentials are required by the current example commands.

## Excluded

Raw medical/source narratives, full corpora and question/split text files, answer keys, per-question text predictions, expert workbooks, SFT/teacher text outputs, weights/adapters/indexes, server login helpers, local secret/configuration files, temporary dependencies, manuscript sources, referee reports, and figure-editing projects are excluded. Identifier-only membership, binary correctness and selected categorical/numeric ratings are included as listed above. Original local files and the pre-existing export backup remain local.

## Verification boundaries

The offline test suite verifies synthetic contracts and selected utilities. It does not retrain the private adapter, run medical inference, reproduce all manuscript results, establish a dataset license, or verify clinical safety. Model-serving tests require optional dependencies, and the private-data test is skipped when its dataset is absent.

The TER ablation bootstrap now samples paired indices. Previously saved intervals produced with independent resampling must not be represented as recomputed by this release. No new medical outcome values were generated during packaging.

The archived weight-search script includes a fixed original-Test baseline assertion; the corpus converter and generic EviCR runner are additional release tools, not evidence of the original study's execution. See `DATA_AVAILABILITY.md` for missing research artifacts and `RIGHTS.md` for the unresolved software license.

The 2026-11-10 numerical additions preserve the archived protocols: wrong-only EviCR is conditional on answer-key routing, reciprocal AI ratings are not expert annotations, and 1,137/1,138-item analyses are distinct. The checker recomputes paired counts, exact unadjusted tests and descriptive score summaries; archived intervals/adjustments are retained. It does not infer absent predictions, retrain an adapter or validate clinical efficacy.
