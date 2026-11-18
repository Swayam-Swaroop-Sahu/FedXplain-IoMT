# Audit before retraining

The supplied 18-page PDF was read in full in the preceding workspace revision. The supervisor appendix is reproduced in guide_review.md. The starting sources include the expanded IEEE Access manuscript, two distinct archived SHAP background protocols, study checkpoints and raw-to-feature documentation, but no original captures or capture IDs were present in the workspace.

## Data

The existing inputs are data/study/{wifi,mqtt,bluetooth}_study.csv. Per the data notes, Wi-Fi and MQTT came from official precomputed tables with labels collapsed by source filename and sampled with nominal caps of 100,000 and 50,000. The raw Bluetooth HCI H4 captures were described as grouped into connection-handle and one-second windows, then mapped into 45 columns. The exact extraction code, minimum allocation, realized indices and original source membership are unavailable. The legacy pipeline performs an 80/20 binary-stratified row split after independent protocol sampling from the assembled subsets, fits one scaler per client on its training rows, and replaces nonfinite values with zero. Shared benign source pools and missing grouping identifiers leave dependence unresolved.

| Client | Seed | Train benign | Train attack | Test benign | Test attack | Attack labels |
|---|---:|---:|---:|---:|---:|---:|
| wifi | 42 | 2,891 | 77,103 | 723 | 19,276 | 13 |
| wifi | 7 | 2,891 | 77,103 | 723 | 19,276 | 13 |
| mqtt | 42 | 14,711 | 25,286 | 3,678 | 6,322 | 5 |
| mqtt | 7 | 14,711 | 25,286 | 3,678 | 6,322 | 5 |
| bluetooth | 42 | 7,688 | 5,388 | 1,923 | 1,347 | 1 |
| bluetooth | 7 | 7,688 | 5,388 | 1,923 | 1,347 | 1 |

The inventory has 19 distinct non-benign labels, including BT-DoS, plus the shared Benign label. The derived counts sum to 99,993 Wi-Fi, 49,997 MQTT and 16,346 Bluetooth. Legacy exact-match counts are in results_v1_archive/exact_match_analysis.json. Feature names, documented analogues and empirical all-zero flags are in results_v1_archive/feature_dictionary.json. Units and formulas for custom Bluetooth summaries remain unverified. The LLC slot was repurposed as an ACL packet fraction. Zero-filled protocol indicator columns motivate a shortcut measurement.

## Training and interpretation

Legacy loaders use shuffled batches of 64, Adam at 0.001, binary cross-entropy on sigmoid probabilities and a 64/32-unit shared architecture with dropout 0.2. Federated clients participate each round and reset optimizer state on each call. Local and pooled models keep optimizer state across epochs; their budgets are 30 and 10 epochs respectively. The historical study used ten rounds of three epochs. Exact training initialization parity is not asserted for the historical runners. FedProx adds a half-scaled squared L2 penalty across every trainable parameter. FedPer shares the first four parameter arrays and retains each locally updated classification head. The fixed model groups isolate Wi-Fi and combine MQTT and Bluetooth. A legacy adaptive runner chooses an update-similarity split after a fixed warm-up, which differs from Sattler's norm-gated recursive procedure.

Legacy DeepExplainer explains sigmoid probabilities in evaluation mode, using 100 benign training rows and 200 held-out attack rows. FedAvg/FedPer share paired sampling on a given protocol and seed. The historical benign probe explains 200 Bluetooth benign rows. A cross-client comparison can change both the input distribution and the background.

## Manuscript evidence and discrepancies

The immediately preceding quantitative ledger is scripts/collect_paper_numbers.py. All detection, calibration, transfer and within-client SHAP tables match archived outputs. Decimal calibration scores in other drafts differ from the JSON. The current draft already corrected equation (15), pooled metrics in the centralized row, the Figure 5 x-axis, the number of methods and the seed-7 reversal. Main scores still refer to historical row splits. The working copy preserves them until the new experiment finishes.

## Environment

Windows 11, Python 3.14.7 and a 12-logical-processor Intel host. Historical execution used PyTorch 2.14.1, Flower 1.39.0, SHAP 0.53.0, scikit-learn 1.9.1 and SciPy 1.18.1. The exact available Python environment is in requirements.lock. Older results use seeds 42 and 7.
