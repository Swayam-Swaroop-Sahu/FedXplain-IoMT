# Prior manuscript revision audit

The primary results now use grouped train, validation and test partitions. This file records the earlier row-split response and its historical source locations. See [TRACEABILITY](TRACEABILITY.md) and [FINAL_REPORT](FINAL_REPORT.md) for current status.

## Build and evidence

The 11-page IEEE Access manuscript compiled with pdfLaTeX and BibTeX. The final log had zero fatal errors and zero unresolved citations or references. PDF text extraction found zero question marks. All 11 pages were rendered for inspection. The supplied class substitutes fonts and emits underfull boxes and two 9.27-point title-block box warnings. The abstract has 184 whitespace-delimited words and six keywords.

The schema validator, all-run completeness audit and six-tab dashboard smoke test passed. The main training sweep was retained. Both centralized checkpoints reproduced the archived pooled confusion counts, and both FedAvg checkpoints reproduced the Bluetooth confusion matrices. The separate transfer experiment has its own checkpoints and evaluation population.

The Overleaf archive contains the synchronized editable sources, bibliography, class, figure assets, generated bibliography and compiled PDF. Build inside the extracted archive with pdfLaTeX, BibTeX, then pdfLaTeX twice. The original timestamped manuscript backup is in this directory. Paper sources remain local under the repository’s existing ignore rules.

The evidence archive contains three row manifests with CSV hashes, train/test and SHAP selections, label visibility, exact feature overlap, feature order, class confusion counts, foreign benign controls and all 378 adaptation trials. Rebuild it with python -m scripts.review_supplement. The two additional analyses run with python -m src.calibration_probe and python -m src.benign_shap_probe. The historical JSON schema keys remain available for older readers.

## Critical wording changes

| Before | After |
|---|---|
| “strictly prohibit pooling” | Identifiable health information is subject to processing and disclosure conditions. Updates may still disclose information. |
| “40 physical IoMT devices” | The source testbed used 25 physical and 15 simulated devices. Results concern a subset of derived records. |
| “output calibration failure rather than representational divergence” | Similar input attribution ranks coexist with different operating points. A scalar adjustment on frozen logits improved Brier score and benign recall, with lower attack recall. |
| “cross-client” agreement above 0.88 | Within Bluetooth, FedPer versus FedAvg yielded 0.9888 twice. Across Wi-Fi and Bluetooth, FedAvg yielded 0.3176 and 0.2514, with zero top-five overlap. |
| “Noise Floor Calibration” | The absolute difference describes the two observed runs for each specific metric. |
| “Zero-Day Threat Generalization” | Foreign attack features are presented to a target client lacking those labels locally. Report recall and source-matched detection margin. |
| Centralized “ceiling” | Pooled-training comparator evaluated separately by client and on the pooled test population. |
| Individual runs described as monotonic | The mean decreased. Seed 7 rose from 0.0322 to 0.0343 over the final step. |

## Review checklist

Numbers follow the 72 detailed corrections in the supplied review. The location identifies the principal response in the revised manuscript. Entries identify remaining evidence and author dependencies.

| Review item | Manuscript section and source line | Disposition |
|---|---|---|
| 1. Abstract opening | Introduction, [line 54](../access.tex#L54) | Qualified the legal claim, cited primary EU and HHS sources, identified pooled feature inputs and the limits of parameter sharing. |
| 2. Abstract protocol description | Introduction, [line 62](../access.tex#L62) | Separated the traffic domains, layers, within-domain prevalence and aggregation weights. HCI is a capture interface. A Bluetooth radio version is unavailable. |
| 3. Abstract dataset description | Benchmark and partitions, [line 86](../access.tex#L86) | Corrected device provenance and documented subset size, sampling caps, source pools and per-protocol labels. Original row-to-device and source-file linkage remains unavailable. |
| 4. Abstract causal inference | Introduction, [line 64](../access.tex#L64) | Replaced novelty and causal claims with three evaluation questions and two distinct SHAP comparison types. The local operating point is tested separately. |
| 5. Abstract correlation claim | Introduction, [line 64](../access.tex#L64) | Replaced novelty and causal claims with three evaluation questions and two distinct SHAP comparison types. The local operating point is tested separately. |
| 6. Abstract length and definitions | Abstract, [line 40](../access.tex#L40) | 184 abstract words and six keywords. |
| 7. L opening paragraph | Introduction, [line 54](../access.tex#L54) | Qualified the legal claim, cited primary EU and HHS sources, identified pooled feature inputs and the limits of parameter sharing. |
| 8. L regulatory citations and privacy | Introduction, [line 54](../access.tex#L54) | Qualified the legal claim, cited primary EU and HHS sources, identified pooled feature inputs and the limits of parameter sharing. |
| 9. L “Section I A” protocol list | Introduction, [line 62](../access.tex#L62) | Separated the traffic domains, layers, within-domain prevalence and aggregation weights. HCI is a capture interface. A Bluetooth radio version is unavailable. |
| 10. L Bluetooth bullet | Introduction, [line 62](../access.tex#L62) | Separated the traffic domains, layers, within-domain prevalence and aggregation weights. HCI is a capture interface. A Bluetooth radio version is unavailable. |
| 11. L disconnected silos and aggregate metrics | Introduction, [line 62](../access.tex#L62) | Separated the traffic domains, layers, within-domain prevalence and aggregation weights. HCI is a capture interface. A Bluetooth radio version is unavailable. |
| 12. L Wi-Fi dominance | Introduction, [line 62](../access.tex#L62) | Separated the traffic domains, layers, within-domain prevalence and aggregation weights. HCI is a capture interface. A Bluetooth radio version is unavailable. |
| 13. L clinical consequences | Discussion, [line 367](../access.tex#L367) | Limited consequences to potential operational risks. No clinical outcomes were observed. |
| 14. R “Section I B” novelty and diagnosis | Introduction, [line 64](../access.tex#L64) | Replaced novelty and causal claims with three evaluation questions and two distinct SHAP comparison types. The local operating point is tested separately. |
| 15. R contribution bullets | Introduction, [line 64](../access.tex#L64) | Replaced novelty and causal claims with three evaluation questions and two distinct SHAP comparison types. The local operating point is tested separately. |
| 16. L zero-day contribution | Cross-protocol attack transfer, [line 342](../access.tex#L342) | Identified foreign feature domains, target model and private head, train-fitted target scaler and fixed threshold. Supplement tabulates train/test label visibility and support by source. |
| 17. L data-scarcity and release claims | Data and code availability, [line 380](../access.tex#L380) | Added repository and supplement locations. The adaptation section separates target-inclusive and excluded pretraining. A public release identifier and license require author action. |
| 18. L Section II A FedProx | Related Work, [line 69](../access.tex#L69) | Distinguished the penalty from a hard bound, adaptive clustering from fixed groups, and previous four-client federated SHAP work from the present protocol partitioning. |
| 19. R clustered learning continuation | Related Work, [line 69](../access.tex#L69) | Distinguished the penalty from a hard bound, adaptive clustering from fixed groups, and previous four-client federated SHAP work from the present protocol partitioning. |
| 20. R Sections II B and II D | Related Work, [line 69](../access.tex#L69) | Distinguished the penalty from a hard bound, adaptive clustering from fixed groups, and previous four-client federated SHAP work from the present protocol partitioning. |
| 21. R Section II C and architecture opening | Benchmark and partitions, [line 86](../access.tex#L86) | Corrected device provenance and documented subset size, sampling caps, source pools and per-protocol labels. Original row-to-device and source-file linkage remains unavailable. |
| 22. L architecture item 1 | Benchmark and partitions, [line 86](../access.tex#L86) | Corrected device provenance and documented subset size, sampling caps, source pools and per-protocol labels. Original row-to-device and source-file linkage remains unavailable. |
| 23. L architecture item 2 | Benchmark and partitions, [line 90](../access.tex#L90) | Reconstructed train/test row membership and counted exact 45-feature matches across splits. Independent capture validation remains outstanding. |
| 24. L architecture item 4 | Attribution and detection metrics, [line 182](../access.tex#L182) | Specified model output, sample/background sizes and source splits, version, model mode and paired sampling. Supplement reconstructs row positions. Added a benign Bluetooth check. |
| 25. L architecture item 5 | Within-client and cross-client explanations, [line 250](../access.tex#L250) | Rewrote interpretation around the observed comparison. Full ranks can agree even when leading features and classification behavior differ. Added benign SHAP and retained local-only results. |
| 26. R Section III B privacy and threats | Model and training, [line 177](../access.tex#L177) | Delimited local retention and separated security of the federation from network events being detected. |
| 27. R Section III C and Table 1 | Benchmark and partitions, [line 86](../access.tex#L86) | Corrected device provenance and documented subset size, sampling caps, source pools and per-protocol labels. Original row-to-device and source-file linkage remains unavailable. |
| 28. R Table 1 details | Benchmark and partitions, [line 93](../access.tex#L93) | Included exact training and test class counts and defined Bluetooth windows. |
| 29. R Bluetooth preprocessing | Benchmark and partitions, [line 88](../access.tex#L88) | Documented zero-filled fields and the repurposed LLC field. The supplement lists ordered features and variation. A complete unit-level dictionary and a protocol-shortcut ablation remain open. |
| 30. Figure 1 local training | Benchmark and partitions, [line 136](../access.tex#L136) | Rebuilt the architecture from configuration with three local epochs and ten rounds. Labeled local data, model exchanges, private heads, fixed clusters and offline pooled training. |
| 31. Figure 1 data flow and outputs | Benchmark and partitions, [line 136](../access.tex#L136) | Rebuilt the architecture from configuration with three local epochs and ten rounds. Labeled local data, model exchanges, private heads, fixed clusters and offline pooled training. |
| 32. Figure 1 caption and convergence | Benchmark and partitions, [line 136](../access.tex#L136) | Rebuilt the architecture from configuration with three local epochs and ten rounds. Labeled local data, model exchanges, private heads, fixed clusters and offline pooled training. |
| 33. L “Section III D” model specification | Model and training, [line 147](../access.tex#L147) | Defined hidden widths, elementwise activation, dropout, attack probability and operating threshold. Removed unsupported latency claims. |
| 34. R “Section III E” opening | Model and training, [line 177](../access.tex#L177) | Specified six paradigms, Adam, state resets, batches, update counts, full participation and final-round selection. Original independent validation curves and compute-matched controls are unavailable. |
| 35. R centralised and local controls | Model and training, [line 177](../access.tex#L177) | Specified six paradigms, Adam, state resets, batches, update counts, full participation and final-round selection. Original independent validation curves and compute-matched controls are unavailable. |
| 36. Original architecture diagram | Benchmark and partitions, [line 136](../access.tex#L136) | Rebuilt the architecture from configuration with three local epochs and ten rounds. Labeled local data, model exchanges, private heads, fixed clusters and offline pooled training. |
| 37. L FedAvg and optimiser | Model and training, [line 177](../access.tex#L177) | Specified six paradigms, Adam, state resets, batches, update counts, full participation and final-round selection. Original independent validation curves and compute-matched controls are unavailable. |
| 38. L Equations 5 to 7 | Model and training, [line 151](../access.tex#L151) | Defined uploaded local weights, full-participation weights and the all-parameter squared penalty. Described the four tested settings. |
| 39. L FedPer and clustered training | Model and training, [line 170](../access.tex#L170) | Defined common initial heads, persistence, joint local training, shared arrays, local inference, cluster sample totals and aggregation. |
| 40. L training budget and convergence | Model and training, [line 177](../access.tex#L177) | Specified six paradigms, Adam, state resets, batches, update counts, full participation and final-round selection. Original independent validation curves and compute-matched controls are unavailable. |
| 41. L to R SHAP protocol | Attribution and detection metrics, [line 182](../access.tex#L182) | Specified model output, sample/background sizes and source splits, version, model mode and paired sampling. Supplement reconstructs row positions. Added a benign Bluetooth check. |
| 42. R Equations 8 to 11 | Attribution and detection metrics, [line 186](../access.tex#L186) | Defined Jaccard, average-rank Spearman, cosine, ranges, undefined cases and deterministic ties. Verified the tie rule reproduces all historical top-five lists. Background uncertainty remains unmeasured. |
| 43. R Equation 12 and uncertainty rule | Attribution and detection metrics, [line 198](../access.tex#L198) | Defined a metric-specific descriptive difference. Removed the two-seed uncertainty decision rule. |
| 44. Figure 2 and caption | Detection by traffic domain, [line 238](../access.tex#L238) | Corrected the panels and caption and specified observed two-seed ranges. |
| 45. L “Section IV A” result reporting | Detection by traffic domain, [line 206](../access.tex#L206) | Added confusion counts, support, attack recall and false alarms. Replayed centralized checkpoints and verified sums against both published pooled matrices. |
| 46. L “Section IV B” monotonicity | FedProx penalty sweep, [line 242](../access.tex#L242) | Restricted monotonicity to the four means, reported the seed-7 reversal and labeled the categorical axis. Gradient conflict and client-weight dominance remain hypotheses. |
| 47. R gradient mechanism paragraph | FedProx penalty sweep, [line 242](../access.tex#L242) | Restricted monotonicity to the four means, reported the seed-7 reversal and labeled the categorical axis. Gradient conflict and client-weight dominance remain hypotheses. |
| 48. R Section IV C wording | Within-client and cross-client explanations, [line 250](../access.tex#L250) | Rewrote interpretation around the observed comparison. Full ranks can agree even when leading features and classification behavior differ. Added benign SHAP and retained local-only results. |
| 49. R cross-model SHAP results | Within-client and cross-client explanations, [line 253](../access.tex#L253) | Added a separate within-client model table with the three metrics, seed, output and sampling protocol. |
| 50. R representation conclusion and FedPer recovery | Within-client and cross-client explanations, [line 250](../access.tex#L250) | Rewrote interpretation around the observed comparison. Full ranks can agree even when leading features and classification behavior differ. Added benign SHAP and retained local-only results. |
| 51. Figure 3 | Within-client and cross-client explanations, [line 302](../access.tex#L302) | Regenerated the heatmap for all penalties and removed the uncertainty decision rule. Stored JSON provides all numerical cells. |
| 52. Table 2 centralised row | Detection by traffic domain, [line 206](../access.tex#L206) | Added confusion counts, support, attack recall and false alarms. Replayed centralized checkpoints and verified sums against both published pooled matrices. |
| 53. Table 2 rounding and reporting | Attribution and detection metrics, [line 202](../access.tex#L202) | Documented averaging and precision. Detection means use unrounded client metrics. Historical pairwise SHAP metrics are serialized to four decimals. |
| 54. Figure 4 | FedProx penalty sweep, [line 242](../access.tex#L242) | Restricted monotonicity to the four means, reported the seed-7 reversal and labeled the categorical axis. Gradient conflict and client-weight dominance remain hypotheses. |
| 55. R Equations 13 to 16 | Personalization and fixed groups, [line 306](../access.tex#L306) | Defined a positive reference gap and recovered fraction, corrected the displayed seed-7 subtraction, and retained the per-seed percentages. |
| 56. R explanatory consistency paragraph | Within-client and cross-client explanations, [line 250](../access.tex#L250) | Rewrote interpretation around the observed comparison. Full ranks can agree even when leading features and classification behavior differ. Added benign SHAP and retained local-only results. |
| 57. Table 3 | Attribution and detection metrics, [line 186](../access.tex#L186) | Defined Jaccard, average-rank Spearman, cosine, ranges, undefined cases and deterministic ties. Verified the tie rule reproduces all historical top-five lists. Background uncertainty remains unmeasured. |
| 58. Figure 5 axis and caption | Personalization and fixed groups, [line 317](../access.tex#L317) | Defined the all-pair average and both error bars. Recomputed plot coordinates from full SHAP vectors and removed optimality claims. |
| 59. L trade-off interpretation | Personalization and fixed groups, [line 317](../access.tex#L317) | Defined the all-pair average and both error bars. Recomputed plot coordinates from full SHAP vectors and removed optimality claims. |
| 60. L Section IV E clustering conclusion | Personalization and fixed groups, [line 313](../access.tex#L313) | Reported cluster outcomes and the remaining within-cluster sharing. |
| 61. R Section IV F attack transfer | Cross-protocol attack transfer, [line 342](../access.tex#L342) | Identified foreign feature domains, target model and private head, train-fitted target scaler and fixed threshold. Supplement tabulates train/test label visibility and support by source. |
| 62. R attack-recall claims | Cross-protocol attack transfer, [line 344](../access.tex#L344) | Presented recall with source-matched benign flag rates and detection margins for all four methods. Supplement includes attack-wise seed-specific support and metrics. |
| 63. L sample-efficiency experiment | Adaptation with limited local data, [line 362](../access.tex#L362) | Distinguished target-inclusive and excluded pretraining. Documented all seven budgets, three draws per split seed, matched adaptation steps and fresh-model controls in the supplement. |
| 64. L six trials and claimed convergence | Adaptation with limited local data, [line 362](../access.tex#L362) | Distinguished target-inclusive and excluded pretraining. Documented all seven budgets, three draws per split seed, matched adaptation steps and fresh-model controls in the supplement. |
| 65. L “Section V A” clinical interpretation | Discussion, [line 367](../access.tex#L367) | Limited consequences to potential operational risks. No clinical outcomes were observed. |
| 66. L “Section V B” item 1 | Attribution and detection metrics, [line 198](../access.tex#L198) | Defined a metric-specific descriptive difference. Removed the two-seed uncertainty decision rule. |
| 67. L “Section V B” items 2 and 3 | Limits of inference, [line 372](../access.tex#L372) | Identified saved-run versus cross-platform reproducibility. Removed the trajectory claim. Checkpoint-level replication was performed for the central and Bluetooth FedAvg baselines. |
| 68. R “Section V C” optimisation methods | Limits of inference, [line 372](../access.tex#L372) | Corrected the first-order control variate and update normalization descriptions. No new empirical drift-correction claim. |
| 69. R adaptive thresholding | Retrospective Bluetooth calibration probe, [line 321](../access.tex#L321) | Distinguished temperature, intercept and threshold changes. Measured specificity, sensitivity, probability scores and reliability after Platt scaling. |
| 70. R limitations and conclusion | Conclusion, [line 378](../access.tex#L378) | Bound conclusions to one offline subset and report specificity gains alongside lost sensitivity and the stronger local comparator. |
| 71. R Acknowledgment and end matter | Data and code availability, [line 386](../access.tex#L386) | Removed the empty acknowledgment and added four affiliation-based biographies with template photo placeholders. Authors must confirm details. |
| 72. Reference list | Data and code availability, [line 384](../access.tex#L384) | Replaced the unmatched survey and corrected the dataset, KDD, UNSW, Bot-IoT, CICIoT2023 and clustered-learning records. BibTeX orders references by citation. Comprehensive publisher correction/retraction clearance remains open. |

## Mathematical corrections

- R1, data, widths, dropout and outputs. [Manuscript line 140](../access.tex#L140)
- R2, local loss and full-participation weights. [Manuscript line 147](../access.tex#L147)
- R3, all-parameter proximal objective. [Manuscript line 155](../access.tex#L155)
- R4, mean absolute attribution. [Manuscript line 182](../access.tex#L182)
- R5, top-five Jaccard and tie rule. [Manuscript line 186](../access.tex#L186)
- R6, tie-aware Spearman. [Manuscript line 190](../access.tex#L190)
- R7, cosine and seed spread. [Manuscript line 194](../access.tex#L194)
- R8, binary metrics and aggregation. [Manuscript line 202](../access.tex#L202)
- R9, gap and recovered fraction. [Manuscript line 306](../access.tex#L306)
- R10, all-pair plot and both ranges. [Manuscript line 317](../access.tex#L317)
- R11, calibration versus threshold decisions. [Manuscript line 321](../access.tex#L321)
- R12, trunk, head and clusters, with scaler and budget definitions in Methods. [Manuscript line 170](../access.tex#L170)

## Submission dependencies

The Wi-Fi splits contain 4,329 and 4,417 test rows whose complete feature vector matches a training vector. Original capture/session identifiers, extractor source and a fully verified feature-unit dictionary are needed for a capture-independent replication. Further split seeds, prospectively held-out calibration, matched threshold/head/fine-tuning controls and background-sampling sensitivity would strengthen mechanism and generalization claims. The current paper limits its claims accordingly.

Author-approved biographies, portraits, applicable declarations, public release terms and final publisher checks remain open. Six DOI metadata records were compared against Crossref. The cross-client comparison, baseline replay and local evidence checks above use the current study files rather than historical backups.

Primary records used include the [UNB dataset description](https://www.unb.ca/cic/datasets/iomt-dataset-2024.html), [dataset publication](https://doi.org/10.1016/j.iot.2024.101351), [four-client federated SHAP study](https://doi.org/10.3390/fi17060234), [HHS research guidance](https://www.hhs.gov/hipaa/for-professionals/special-topics/research/index.html), [EU regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj) and [IEEE Access submission guidelines](https://ieeeaccess.ieee.org/authors/submission-guidelines/).

## Expanded manuscript verification

The revised package contains 35 cited works, an unchanged 184-word abstract, three new figures, all 19 attack labels, separate detection tables for each seed and a dictionary of 45 ordered features. Replaying the 14 archived federated configurations across three clients reproduced all 42 confusion matrices. The MQTT attack fraction is 63.2%. All experiments use the fixed training schedule documented in the sensitivity report. The recorded contrasts identify seed and proximal-penalty variation only. The professor-specific search verified a relevant 2025 federated IDS publication coauthored by Dr. Amali. Its citation describes hyperparameter optimization. An XAI-specific publication by the professor was not independently identified.
