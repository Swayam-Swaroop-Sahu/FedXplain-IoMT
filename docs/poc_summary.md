# FedXplain-IoMT Proof-of-Concept (POC) Summary

## 1. Overview
This proof-of-concept implements a lightweight federated learning pipeline for binary attack detection across three protocol-grounded Internet of Medical Things (IoMT) clients (Wi-Fi, MQTT, and Bluetooth) derived from the CICIoMT2024 dataset. A unified 3-layer multi-layer perceptron (IoMTMLP) was trained centrally across all pooled protocol data as an initial benchmark, and subsequently in a federated architecture using manual Federated Averaging (FedAvg) aggregation over 5 communication rounds with 2 local epochs per client. Post-training model explanations were extracted for held-out attack traffic at each client using Kernel/Deep SHAP to compute feature attribution distributions and assess whether clients converge on diverging explanatory representations despite sharing a single global model.

---

## 2. Centralized vs. Federated Performance

Both the centralized baseline and federated FedAvg global model achieve strong attack detection capabilities across the POC subsets without architectural changes or hyperparameter tuning.

| Training Paradigm | Evaluation Scope | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Centralized Baseline** (10 epochs) | Combined Pooled Test Set | 0.9810 | 0.9855 | 0.9894 | **0.9875** |
| **Federated FedAvg** (Round 5) | Client 0: Wi-Fi Test Set | — | — | — | 0.9829 |
| **Federated FedAvg** (Round 5) | Client 1: MQTT Test Set | — | — | — | 0.9915 |
| **Federated FedAvg** (Round 5) | Client 2: Bluetooth Test Set | — | — | — | 0.9938 |
| **Federated FedAvg** (Round 5) | **Macro Average across Clients** | — | — | — | **0.9894** |

---

## 3. Cross-Client Explanation Divergence

SHAP explanations were generated using protocol-specific benign traffic as background references (50 samples) and evaluated against held-out attack samples (up to 100 samples). Top-5 feature overlap was quantified using the Jaccard similarity index:

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

| Client Pair | Shared Top-5 Features | Jaccard Index |
| :--- | :--- | :---: |
| **wifi-mqtt** | Magnitue, ack_flag_number, psh_flag_number, rst_count | **0.6667** |
| **wifi-bluetooth** | Magnitue | **0.1111** |
| **mqtt-bluetooth** | Magnitue | **0.1111** |

*Lower Jaccard indicates higher explanation divergence across clients.*

---

## 4. Visual Evidence of Divergence

![Cross-Client SHAP Explanation Divergence](../results/shap_cross_client_comparison.png)

Features that repeat across clients (such as `rst_count`, `ack_flag_number`, `psh_flag_number`, and `Magnitue`) maintain consistent colors across subplots to illustrate visual alignment between IP-based protocols and sharp divergence in Bluetooth traffic.

---

## 5. Interpretation

The proof-of-concept demonstrates clear, visible divergence in top SHAP explanatory features across heterogeneous IoMT client protocols, most notably between network-layer clients (Wi-Fi and MQTT, which prioritize TCP flag metrics like `rst_count` and `ack_flag_number`) and link-layer clients (Bluetooth, which relies almost exclusively on flow statistics such as `Magnitue`, `AVG`, and `Tot sum`). Because this pilot represents a single-seed run on a compact subset of the dataset without multi-seed noise-floor validation, these findings should be treated strictly as an initial demonstration rather than statistically definitive conclusions. Nonetheless, the observed variation confirms that uniform global weights produce protocol-contingent local decision logic, motivating the comprehensive multi-seed study and noise-floor calibration described in the primary project methodology.
