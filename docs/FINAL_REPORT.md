# Final revision report

## Delivery

The local branch is revision/ieee-access. Phase commits cover the provenance audit, implementation, experiment outputs, verified references, manuscript, and reproducibility materials. The original JSON results are archived in results_v1_archive. Byte comparisons between the source and archive matched all 19 JSON files.

The evaluation suite started with empty revised result directories. It produced 70 main models across five seeds and 28 withheld-family models across two seeds. The main runs used 164,460 optimizer steps and summed to 16.9 minutes of per-run training time. The withheld-family runs used 63,324 steps and summed to 6.6 minutes. Runs executed concurrently. The full-budget trial was stopped before completing the suite. The retained schedule uses six rounds, one local epoch, batch size 256 and six pooled epochs. All eligible training, validation and test records were available. Sensitivity and adaptation use seeds 42 and 7, as does family withholding. The calibration and SHAP diagnostics use all five seeds.

## Findings

- FedAvg client-mean macro F1 was 0.6376 and benign Bluetooth recall was 0.0161. FedPer had 0.7180 specificity, fixed protocol groups 0.9827, equal weighting 0.9592, and local-only 0.9965. All adaptive checkpoints used singleton groups. The evidence concerns the matched training schedule.
- Platt scaling raised specificity from 0.0161 to 0.8370 and lowered Brier score from 0.3891 to 0.1007. Attack recall changed from 0.9982 to 0.9084. Temperature scaling lowered Brier score and preserved predictions. Full fine-tuning improved both recalls relative to the output-only adjustment.
- Within Bluetooth, attack SHAP rank agreement averaged 0.9843. The FedAvg Wi-Fi–Bluetooth comparison averaged 0.2450. Trunk CKA averaged 0.9720. These measures describe different comparison objects and leave the underlying mechanism open.
- Feature groups eliminate exact cross-partition vector matches by construction. The rounded screen flags possible near matches. The protocol shortcut classifier averaged 0.9423 accuracy. Capture-independent generalization remains untested.

## Verification

- Seven pytest regression tests and six Streamlit tabs passed. The legacy schema and result-completeness audits also passed.
- Recomputed metrics match 294 client prediction files. Every selected round minimizes the logged validation criterion. Zero-penalty FedProx gives the FedAvg scores, and the matched pooled models use equal optimizer step counts.
- Clean seed-42 FedAvg and FedPer reruns reproduced the sample IDs and logits of all six client files bit for bit.
- The generated tables and numerical macros matched the saved outputs. That build resolved 41 bibliography entries and produced a 15-page IEEE Access PDF. MiKTeX used substitute fonts and reported two template title-block width warnings and a minor table-width warning.
- The release and Overleaf archives passed ZIP integrity checks. The reproducibility archive contains code, configs, manifests, checkpoints, logits, per-label reports, SHAP outputs and documentation. It excludes raw and derived traffic tables.

## Author and data dependencies

- Supply the original extractor, source capture and device/session identifiers, plus feature-unit documentation for grouped independent evaluation and exact reconstruction of historical sampling.
- Confirm applicable author biographies, photographs, funding, competing interests, institutional declarations, licence, release tag and public repository link.
- Identify the intended supervisor XAI paper by title or persistent identifier. A verified federated IDS paper from the supervisor is already cited. Optional SCAFFOLD and FedNova empirical controls were outside the retained compute budget.

## Follow-up verification on 19 November 2026

- Verified the experiment-to-publication order in Git. The grouped training implementation precedes the run records, which in turn precede the manuscript rewrite and final checks. The revised source is committed on the local `revision/ieee-access` branch. A read-only check of the GitHub remote found `main` at the earlier revision and no published branch with the revision name.
- Reran the seven regression tests, dashboard smoke test, numerical generation check, and metrics validation on all 294 client prediction files. All five primary split manifest digests matched, and the Bluetooth calibration fitting and selection groups were disjoint subsets of validation. All 19 archived JSON files matched the corresponding original outputs.
- Fresh seed-42 FedAvg and FedPer training reproduced the sample IDs, binary labels, logits and probabilities in all six client prediction files exactly.
- The README now explains branch visibility and links to the implementation and evidence. The paper explains how validation-selected checkpoints became test predictions and corrects stale wording about training shares and target-inclusive adaptation. [Guide checklist](guide_checklist.md)
- Rebuilt the 15-page PDF with a 182-word rendered abstract, 41 resolved references and no broken cross-references. The new opening reports the measured five-seed values from the generated numerical macros.
