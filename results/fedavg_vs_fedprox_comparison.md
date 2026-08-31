# FedAvg vs. FedProx: Performance and SHAP Divergence Comparison

## 1. Final Round F1 Scores (Round 5/5)

| Client | FedAvg F1 | FedProx (μ=0.01) F1 | Δ (Prox − Avg) |
| :--- | :---: | :---: | :---: |
| **Wi-Fi** | 0.9874 | 0.9874 | 0.0000 |
| **MQTT** | 0.8607 | 0.8990 | **+0.0383** |
| **Bluetooth** | 0.7198 | 0.8058 | **+0.0860** |
| **Macro Avg** | 0.8560 | 0.8974 | **+0.0414** |

> [!WARNING]
> Bluetooth F1 is below the 0.85 reliability threshold for both algorithms. SHAP explanations for Bluetooth should be interpreted with caution.

> [!WARNING]
> **Wi-Fi F1 is inflated by severe class imbalance.** The Wi-Fi test set contains only 36 benign samples (3.6%) vs. 963 attack samples (96.4%). Both models misclassify 21 attack samples as benign (benign precision = 0.6111). The high binary F1 (0.9874) reflects strong attack-class recall (0.9782) but masks poor benign-class precision. Macro F1 (averaging benign and attack F1 equally) is 0.8604, a more honest measure for this imbalanced test set.

---

## 2. SHAP Divergence Metrics Comparison

Top-5 feature explanation overlap (Jaccard index) between client pairs:

| Client Pair | FedAvg Jaccard | FedProx Jaccard | Δ (Prox − Avg) |
| :--- | :---: | :---: | :---: |
| **wifi-mqtt** | 0.6667 | 0.4286 | -0.2381 |
| **wifi-bluetooth** | 0.1111 | 0.1111 | 0.0000 |
| **mqtt-bluetooth** | 0.1111 | 0.0000 | -0.1111 |

*Lower Jaccard indicates higher explanation divergence across clients.*

---

## 3. Shared Top-5 Features Breakdown

- **wifi-mqtt**:
  - *FedAvg*: `rst_count`, `ack_flag_number`, `psh_flag_number`, `Magnitue` (4 shared)
  - *FedProx*: `rst_count`, `psh_flag_number`, `ack_flag_number` (3 shared)
- **wifi-bluetooth**:
  - *FedAvg*: `Magnitue` (1 shared)
  - *FedProx*: `Magnitue` (1 shared — from Wi-Fi side only; BT top-5 does not include `Magnitue` in FedProx top-5)
- **mqtt-bluetooth**:
  - *FedAvg*: `Magnitue` (1 shared)
  - *FedProx*: None (0 shared)

---

## 4. Individual Client Top-5 SHAP Features

| Rank | Wi-Fi (FedAvg) | Wi-Fi (FedProx) | MQTT (FedAvg) | MQTT (FedProx) | BT (FedAvg) | BT (FedProx) |
|:---:|:---|:---|:---|:---|:---|:---|
| 1 | rst_count | rst_count | rst_count | rst_count | Magnitue | Magnitue |
| 2 | ack_flag_number | ack_flag_number | psh_flag_number | psh_flag_number | AVG | Tot sum |
| 3 | psh_flag_number | psh_flag_number | HTTP | ack_flag_number | Number | AVG |
| 4 | Variance | Variance | ack_flag_number | Max | Max | Tot size |
| 5 | Magnitue | Magnitue | Magnitue | syn_flag_number | Tot sum | Number |

---

## 5. Interpretation

FedProx (μ=0.01) improves F1 on the minority clients (MQTT: +3.8pp, Bluetooth: +8.6pp) while maintaining identical Wi-Fi performance — consistent with FedProx's design goal of stabilizing heterogeneous clients via proximal regularization.

**Wi-Fi F1 identity investigation**: The identical Wi-Fi F1 (0.987421 to 6 decimal places) is NOT because the models are similar — the L2 norm of the full weight difference between FedAvg and FedProx global models is **2.403** (non-trivial), with max prediction probability difference of **0.289** across test samples. Despite these weight differences, the threshold binarization at 0.5 produces identical binary predictions (0 of 999 predictions differ). This means the two models reached the same Wi-Fi decision boundary via different weight-space paths. The FedProx improvement at μ=0.01 is driven entirely by MQTT and Bluetooth, not Wi-Fi.

On the explanability side, FedProx produces slightly *higher* explanation divergence (lower Jaccard overlap) on two of three client pairs (wifi-mqtt: 0.67→0.43; mqtt-bluetooth: 0.11→0.00). This suggests that proximal regularization, while improving client-level F1, may allow each client's local decision logic to specialize more distinctly rather than converging toward the dominant client's feature preferences.

However, this is strictly ONE run with ONE seed (seed=42) and ONE value of μ, serving only as a preliminary signal rather than a validated finding. The full study will systematically test whether proximal regularization impacts explanatory divergence using multi-seed repetitions, μ parameter sweeps, and statistical noise-floor calibration.

---

## 6. Known Limitations

- **Single seed**: All results use `random_state=42`. No multi-seed noise-floor estimation.
- **Bluetooth F1 < 0.85**: Both algorithms underperform on Bluetooth, making SHAP explanations for that client less reliable.
- **Wi-Fi F1 inflated by class imbalance**: Wi-Fi test set is 96.4% attack. Binary F1 (0.9874) masks poor benign precision (0.6111, 21 false negatives out of 54 benign predictions). Macro F1 is 0.8604.
- **Sample imbalance**: Wi-Fi dominates aggregation with 50% of total samples, creating structural bias in the global model.
- **μ=0.01 was not tuned**: This value was selected as a lower-intensity alternative to μ=0.1 (which was too aggressive). A proper μ sweep would be needed for the full study.
- **Prior μ=0.1 investigation**: Initial FedProx runs used μ=0.1, which produced proximal regularization exceeding 100% of BCE loss for Wi-Fi. This over-regularization was diagnosed via the proximal term magnitude analysis in the diagnosis report. μ=0.01 was selected as a correction.

---

## 7. Wi-Fi Confusion Matrix

Both FedAvg and FedProx (μ=0.01) produce **identical** confusion matrices on the Wi-Fi test set (0 prediction differences across 999 samples):

|  | Predicted Benign | Predicted Attack |
|:---|:---:|:---:|
| **Actual Benign** (n=36) | 33 (TN) | 3 (FP) |
| **Actual Attack** (n=963) | 21 (FN) | 942 (TP) |

| Class | Precision | Recall |
|:---|:---:|:---:|
| Benign | 0.6111 | 0.9167 |
| Attack | 0.9968 | 0.9782 |

---

## 8. Reproducibility Guardrails Added

To prevent future silent staleness bugs (like the stale FedAvg baselines diagnosed in this investigation):

1. **Feature count assertion** (`src/utils.py`): `load_and_preprocess()` now asserts that the resulting feature count equals the expected constant (45). Any preprocessing or CSV schema change that alters the feature set will raise an immediate, clear error rather than silently producing incomparable results.
2. **Cross-protocol input_dim consistency check** (`src/centralized.py`, `src/run_federated.py`, `src/run_federated_prox.py`): Each runner script now asserts that `get_input_dim()` returns the same value for all three protocols before training begins.
