# Revision log

## Phase 1

Audited the data pipeline, archived the starting result records, locked the local environment and documented outstanding provenance requirements. Review items 2–4, 23, 27–30 and 35–49. Evidence is in results_v1_archive and AUDIT.md.

## Phase 2

Implemented exact-feature grouping with optional source manifest ingestion, three-way splits, train-only scalers, matched initialization, validation-only checkpoint selection, per-round logs, probability and class metrics, Sattler-style adaptive grouping, prediction exports, calibration probes and class-conditional explanations. Seven regression tests passed. Review items 4, 19, 23, 24, 29–43, 45, 47, 49 and 57.

## Phase 3

Completed the five-seed suite, two-seed reconnaissance holdout, target-excluded adaptation, weighting and class-balance controls, validation-only output diagnostics and the sensitivity grid. Validated 294 prediction files, exact zero-penalty FedAvg parity and equal optimizer step counts for the compute-matched central comparator. Earlier results are preserved. New outputs are under results/revision and results/revision_holdout. Review items 1, 12, 35, 39–41, 45–47, 55, 56 and 60–70.
