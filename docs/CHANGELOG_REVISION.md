# Revision log

## Phase 1

Audited the data pipeline, archived the starting result records, locked the local environment and documented outstanding provenance requirements. Review items 2–4, 23, 27–30 and 35–49. Evidence is in results_v1_archive and AUDIT.md.

## Phase 2

Implemented exact-feature grouping with optional source manifest ingestion, three-way splits, train-only scalers, matched initialization, validation-only checkpoint selection, per-round logs, probability and class metrics, Sattler-style adaptive grouping, prediction exports, calibration probes and class-conditional explanations. Seven regression tests passed. Review items 4, 19, 23, 24, 29–43, 45, 47, 49 and 57.

## Phase 3

Completed the five-seed suite, two-seed reconnaissance holdout, target-excluded adaptation, weighting and class-balance controls, validation-only output diagnostics and the sensitivity grid. Validated 294 prediction files, exact zero-penalty FedAvg parity and equal optimizer step counts for the compute-matched central comparator. Earlier results are preserved. New outputs are under results/revision and results/revision_holdout. Review items 1, 12, 35, 39–41, 45–47, 55, 56 and 60–70.

## Phase 4

Verified cited DOI metadata and expanded the bibliography with primary references for deployment context, calibration, representation similarity and benchmark comparison. The supervisor's related federated IDS work is cited. The intended XAI-specific citation remains open. Review items 9–21, 68 and 72. See refs/doi_audit.json and refs/reading_notes.md.

## Phase 5

Rewrote the manuscript around grouped experiment outputs. Generated its numerical macros, detection tables, score reliability curves, sensitivity figures and adaptive topology. Added validation-selected diagnostic results and bounded comparisons. Review items 1–72 and equations R1–R12. Numerical claims point to results/revision/paper_summary.json or the committed configuration.

## Phase 6

Synced the dashboard and README to the generated primary results, added read-time compatibility for archived spread keys, and documented one-command regeneration. The local package contains manifests, predictions and models and omits the traffic capture archives. Guide items 9 and 15–17. Evidence is in the reproducibility checklist and release bundle.
