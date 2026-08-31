# FedXplain-IoMT Proof-of-Concept (POC) Summary

## 1. Overview
This proof-of-concept implements a lightweight federated learning pipeline for binary attack detection across three protocol-grounded Internet of Medical Things (IoMT) clients (Wi-Fi, MQTT, and Bluetooth) derived from the CICIoMT2024 dataset. A unified 3-layer multi-layer perceptron (IoMTMLP) was trained centrally across all pooled protocol data as an initial benchmark, and subsequently in a federated architecture using manual Federated Averaging (FedAvg) aggregation over 5 communication rounds with 2 local epochs per client. A second federated run using FedProx (μ=0.01) was conducted to compare the effect of proximal regularization on both performance and model explanations. Post-training SHAP explanations were extracted for held-out attack traffic at each client using DeepExplainer to compute feature attribution distributions and assess cross-client explanation divergence.

---

## 2. Centralized vs. Federated Performance

| Training Paradigm | Evaluation Scope | F1-Score |
| :--- | :--- | :---: |
| **Centralized Baseline** (10 epochs) | Combined Pooled Test Set | **0.9875** |
| **Federated FedAvg** (Round 5) | Client 0: Wi-Fi Test Set | 0.9874 |
| **Federated FedAvg** (Round 5) | Client 1: MQTT Test Set | 0.8607 |
| **Federated FedAvg** (Round 5) | Client 2: Bluetooth Test Set | 0.7198 |
| **Federated FedAvg** (Round 5) | **Macro Average across Clients** | **0.8560** |
| **Federated FedProx μ=0.01** (Round 5) | Client 0: Wi-Fi Test Set | 0.9874 |
| **Federated FedProx μ=0.01** (Round 5) | Client 1: MQTT Test Set | 0.8990 |
| **Federated FedProx μ=0.01** (Round 5) | Client 2: Bluetooth Test Set | 0.8058 |
| **Federated FedProx μ=0.01** (Round 5) | **Macro Average across Clients** | **0.8974** |

> **Note:** Bluetooth F1 is below 0.85 for both federated algorithms, flagged as unreliable for SHAP interpretation. Wi-Fi F1 (0.9874) is inflated by severe class imbalance in the test set (96.4% attack, only 3.6% benign); benign-class precision is 0.6111 and macro F1 is 0.8604.

---

## 3. Cross-Client Explanation Divergence

SHAP explanations were generated using protocol-specific benign traffic as background references (50 samples) and evaluated against held-out attack samples (up to 100 samples). Top-5 feature overlap was quantified using the Jaccard similarity index:

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

### FedAvg Global Model

| Client Pair | Shared Top-5 Features | Jaccard Index |
| :--- | :--- | :---: |
| **wifi-mqtt** | rst_count, ack_flag_number, psh_flag_number, Magnitue | **0.6667** |
| **wifi-bluetooth** | Magnitue | **0.1111** |
| **mqtt-bluetooth** | Magnitue | **0.1111** |

### FedProx Global Model (μ=0.01)

| Client Pair | Shared Top-5 Features | Jaccard Index |
| :--- | :--- | :---: |
| **wifi-mqtt** | rst_count, psh_flag_number, ack_flag_number | **0.4286** |
| **wifi-bluetooth** | Magnitue | **0.1111** |
| **mqtt-bluetooth** | None | **0.0000** |

*Lower Jaccard indicates higher explanation divergence across clients.*

---

## 4. Visual Evidence of Divergence

### FedAvg Global Model
![Cross-Client SHAP Explanation Divergence - FedAvg](../results/shap_cross_client_comparison.png)

### FedProx Global Model (μ=0.01)
![Cross-Client SHAP Explanation Divergence - FedProx](../results/shap_cross_client_comparison_fedprox.png)

Features that repeat across clients maintain consistent colors across subplots to illustrate visual alignment between IP-based protocols (Wi-Fi, MQTT) and sharp divergence in Bluetooth traffic.

---

## 5. Interpretation

The proof-of-concept demonstrates clear, visible divergence in top SHAP explanatory features across heterogeneous IoMT client protocols, most notably between network-layer clients (Wi-Fi and MQTT, which prioritize TCP flag metrics like `rst_count` and `ack_flag_number`) and link-layer clients (Bluetooth, which relies almost exclusively on flow statistics such as `Magnitue`, `AVG`, and `Tot sum`).

FedProx (μ=0.01) improves macro-average F1 by +4.1pp over FedAvg (0.8974 vs. 0.8560), primarily benefiting the minority clients (MQTT: +3.8pp, Bluetooth: +8.6pp). On the explanability side, FedProx produces slightly higher explanation divergence (lower Jaccard overlap on 2 of 3 client pairs), suggesting proximal regularization may allow more protocol-specific feature specialization.

Because this pilot represents a single-seed run on a compact dataset subset without multi-seed noise-floor validation, these findings should be treated strictly as an initial demonstration rather than statistically definitive conclusions. Nonetheless, the observed variation confirms that uniform global weights produce protocol-contingent local decision logic, motivating the comprehensive multi-seed study and noise-floor calibration described in the primary project methodology.

**Caveats**: Wi-Fi's binary F1 (0.9874) masks poor benign-class precision (0.6111) due to extreme test-set class imbalance (36 benign vs. 963 attack). The identical Wi-Fi F1 between FedAvg and FedProx is a coincidence of threshold binarization: the two models have a weight-space L2 distance of 2.403 and max prediction probability difference of 0.289, but produce identical binary predictions on this test set. FedProx's improvement is driven entirely by MQTT and Bluetooth.

---

## 6. Reproducibility Guardrails

To prevent future silent staleness bugs:
1. `src/utils.py`: `load_and_preprocess()` asserts feature count equals 45 (the documented CICIoMT2024 schema).
2. `src/centralized.py`, `src/run_federated.py`, `src/run_federated_prox.py`: Each asserts `get_input_dim()` is consistent across all three protocols before training.
