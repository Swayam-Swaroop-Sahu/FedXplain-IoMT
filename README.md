# FedXplain-IoMT: Federated Learning & Explainable AI for Heterogeneous IoMT Intrusion Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-1.5%2B-gold.svg)](https://flower.ai/)
[![SHAP](https://img.shields.io/badge/SHAP-0.42%2B-brightgreen.svg)](https://shap.readthedocs.io/)
[![Dataset](https://img.shields.io/badge/Dataset-CICIoMT2024-purple.svg)](https://www.unb.ca/cic/datasets/iomt-dataset-2024.html)

**FedXplain-IoMT** is a research framework investigating **cross-client explanation divergence** in Federated Learning (FL) for Internet of Medical Things (IoMT) intrusion detection. 

While conventional federated evaluation focuses exclusively on global accuracy and F1 scores, FedXplain-IoMT evaluates whether a converged global model relies on **fundamentally different decision logic** across clients operating on heterogeneous network protocols (Wi-Fi, MQTT, Bluetooth). Using model-agnostic feature attributions (SHAP) and set-theoretic overlap metrics (Jaccard Index), this project provides quantitative and visual evidence of explanatory drift in privacy-preserving medical device environments.

---

## Table of Contents
- [1. Motivation & Research Problem](#1-motivation--research-problem)
- [2. System Architecture](#2-system-architecture)
- [3. Dataset: CICIoMT2024](#3-dataset-ciciomt2024)
- [4. Methods & Algorithms](#4-methods--algorithms)
  - [Model Architecture](#model-architecture)
  - [Centralized Baseline](#centralized-baseline)
  - [Federated Learning (FedAvg & FedProx)](#federated-learning-fedavg--fedprox)
  - [Post-Hoc Explainability (SHAP & Jaccard)](#post-hoc-explainability-shap--jaccard)
- [5. Experimental Results](#5-experimental-results)
  - [Model Detection Performance](#model-detection-performance)
  - [SHAP Feature Attribution Divergence](#shap-feature-attribution-divergence)
  - [FedAvg vs. FedProx Comparison](#fedavg-vs-fedprox-comparison)
- [6. Visualizations](#6-visualizations)
- [7. Installation & Setup](#7-installation--setup)
- [8. Reproduction Guide](#8-reproduction-guide)
- [9. Repository Structure](#9-repository-structure)
- [10. Scope & Next Steps](#10-scope--next-steps)

---

## 1. Motivation & Research Problem

In clinical healthcare environments, IoMT devices (infusion pumps, patient monitors, telemetry sensors) communicate across diverse physical and application protocols:
- **Wi-Fi** (high-throughput transport for telemetry and imaging)
- **MQTT** (lightweight publish-subscribe telemetry for resource-constrained bedside sensors)
- **Bluetooth / BLE** (short-range point-to-point peripheral communication)

Due to strict privacy regulations (HIPAA, GDPR), raw telemetry from hospital partitions cannot be pooled centrally. Federated learning enables distributed training across hospital silos without sharing patient data.

### The Explanation Divergence Problem
Standard evaluation assumes that achieving high classification metrics (e.g., F1 > 0.98) across all clients implies consistent model behavior. However:
1. **Protocol Asymmetry**: IP-based attacks exhibit distinct TCP/IP flag anomalies, whereas Bluetooth attacks manifest purely in packet frequency and flow timing.
2. **Global Representation vs. Local Logic**: A single global neural network parameter set ($\theta^*$) may achieve uniform predictive accuracy while relying on disjoint feature subsets when explaining predictions locally.
3. **Clinical Trust**: If a hospital security analyst cannot verify consistent explanatory rationales across device subnets, automated containment decisions cannot be trusted.

---

## 2. System Architecture

```
                       +-------------------------------+
                       |   Federated Server / Aggregator|
                       |       (FedAvg / FedProx)       |
                       +---------------+---------------+
                                       |
                   Global Weights (w_t)| Aggregate (w_{t+1})
                                       v
         +-----------------------------+-----------------------------+
         |                             |                             |
         v                             v                             v
+-----------------+           +-----------------+           +-----------------+
|   Client 0      |           |   Client 1      |           |   Client 2      |
|  Protocol: Wi-Fi|           |  Protocol: MQTT |           |Protocol: BLE/BT |
| Local Training  |           | Local Training  |           | Local Training  |
| 50 Benign Ref   |           | 50 Benign Ref   |           | 50 Benign Ref   |
| 100 Attack Eval |           | 100 Attack Eval |           | 100 Attack Eval |
+--------+--------+           +--------+--------+           +--------+--------+
         |                             |                             |
         | Local SHAP                  | Local SHAP                  | Local SHAP
         v                             v                             v
+-----------------+           +-----------------+           +-----------------+
| Top-10 Attribs  |           | Top-10 Attribs  |           | Top-10 Attribs  |
+--------+--------+           +--------+--------+           +--------+--------+
         |                             |                             |
         +-----------------------------+-----------------------------+
                                       |
                                       v
                     +-----------------------------------+
                     | Cross-Client Divergence Analysis  |
                     |  - Top-5 Jaccard Similarity Index |
                     |  - Multi-Client Alignment Plots   |
                     +-----------------------------------+
```

---

## 3. Dataset: CICIoMT2024

The study utilizes the **CICIoMT2024** benchmark (Canadian Institute for Cybersecurity, 40 IoMT devices, 18 attack types).

### Protocol Partitions
- **Wi-Fi**: TCP/IP-based attacks (DDoS/DoS via ICMP/SYN/TCP/UDP, ARP Spoofing, Reconnaissance, Benign).
- **MQTT**: Broker-targeted attacks (Connect Flood, Publish Flood, Malformed Data, Benign).
- **Bluetooth**: Extracted from raw Bluetooth HCI H4 `.pcap` captures (Benign vs. BT-DoS). Non-applicable TCP/IP fields are zeroed out while flow timing and statistical metrics are preserved.

### Feature Schema (45 Network Flow Features)
- **Header & Timing**: `Header_Length`, `Duration`, `Rate`, `Srate`, `Drate`, `IAT`
- **TCP Flags & Counts**: `fin_flag_number`, `syn_flag_number`, `rst_flag_number`, `psh_flag_number`, `ack_flag_number`, `ece_flag_number`, `cwr_flag_number`, `ack_count`, `syn_count`, `fin_count`, `rst_count`
- **Protocol Flags**: `HTTP`, `HTTPS`, `DNS`, `Telnet`, `SMTP`, `SSH`, `IRC`, `TCP`, `UDP`, `DHCP`, `ARP`, `ICMP`, `IGMP`, `IPv`, `LLC`
- **Statistical Aggregates**: `Tot sum`, `Min`, `Max`, `AVG`, `Std`, `Tot size`, `Number`, `Magnitue`, `Radius`, `Covariance`, `Variance`, `Weight`

### Preprocessing Pipeline (`src/utils.py`)
1. Drops identifier/timestamp columns (`id`, `ip`, `time`, case-insensitive).
2. Binary target mapping: `BENIGN` $\rightarrow 0$, all attack classes $\rightarrow 1$.
3. Stratified 80/20 train-test split (`random_state=42`).
4. `StandardScaler` fitted strictly on training data and applied to test data.
5. Missing or infinite values handled cleanly via `nan_to_num`.

---

## 4. Methods & Algorithms

### Model Architecture (`src/model.py`)
A unified, lightweight 3-layer Multilayer Perceptron (`IoMTMLP`):
$$\text{Linear}(45, 64) \rightarrow \text{ReLU} \rightarrow \text{Dropout}(0.2) \rightarrow \text{Linear}(64, 32) \rightarrow \text{ReLU} \rightarrow \text{Linear}(32, 1) \rightarrow \text{Sigmoid}$$

### Centralized Baseline (`src/centralized.py`)
- Concatenates training and testing partitions across all three protocols into a pooled dataset.
- Trains for 10 epochs using Adam ($\text{lr} = 0.001$), batch size 64, and Binary Cross-Entropy loss ($\text{BCELoss}$).

### Federated Learning (`src/run_federated.py` & `src/run_federated_prox.py`)
- **Flower NumPyClient (`src/client.py`)**: Manages local data loaders and parameter updates.
- **Communication Schedule**: 5 rounds, 2 local epochs per client per round, batch size 64.
- **FedAvg**: Aggregates updated weights element-wise weighted by client dataset size $n_k$:
  $$w_{t+1} = \sum_{k=1}^K \frac{n_k}{N} w_{t+1}^k$$
- **FedProx**: Incorporates a proximal regularization penalty directly into the client loss objective to restrict local drift from the global parameters $w_t$:
  $$\mathcal{L}_{\text{FedProx}}(w; w_t) = \mathcal{L}_{\text{BCE}}(w) + \frac{\mu}{2} \|w - w_t\|_2^2 \quad (\mu = 0.1)$$

### Post-Hoc Explainability (`src/explain.py` & `src/explain_prox.py`)
- **Explainer**: `shap.DeepExplainer` applied to the converged global PyTorch model.
- **Reference (Background) Set**: 50 benign samples ($y = 0$) sampled from client training sets.
- **Evaluation Set**: Up to 100 attack samples ($y = 1$) sampled from client test sets.
- **Importance Metric**: Mean absolute SHAP value per feature:
  $$\bar{\phi}_j = \frac{1}{M} \sum_{i=1}^M |\phi_{i, j}|$$
- **Divergence Quantification**: Top-5 feature sets $S_A, S_B$ compared via Jaccard similarity index:
  $$J(S_A, S_B) = \frac{|S_A \cap S_B|}{|S_A \cup S_B|}$$
  *A lower Jaccard index signifies greater explanatory divergence across clients.*

---

## 5. Experimental Results

### Model Detection Performance

All paradigms achieve high attack detection performance on the held-out test partitions:

| Training Paradigm | Evaluation Scope | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Centralized Baseline** (10 epochs) | Pooled Combined Test Set | **0.9810** | **0.9855** | **0.9894** | **0.9875** |
| **Federated FedAvg** (Round 5) | Wi-Fi Client Test Set | — | — | — | 0.9829 |
| **Federated FedAvg** (Round 5) | MQTT Client Test Set | — | — | — | 0.9915 |
| **Federated FedAvg** (Round 5) | Bluetooth Client Test Set | — | — | — | 0.9938 |
| **Federated FedAvg** (Round 5) | **Macro Average across Clients** | — | — | — | **0.9894** |
| **Federated FedProx** ($\mu=0.1$, Round 5) | Wi-Fi Client Test Set | — | — | — | 0.9874 |
| **Federated FedProx** ($\mu=0.1$, Round 5) | MQTT Client Test Set | — | — | — | 0.9106 |
| **Federated FedProx** ($\mu=0.1$, Round 5) | Bluetooth Client Test Set | — | — | — | 0.7721 |
| **Federated FedProx** ($\mu=0.1$, Round 5) | **Macro Average across Clients** | — | — | — | **0.8900** |

---

### SHAP Feature Attribution Divergence

#### Top-5 Features Identified per Client (FedAvg)
- **Wi-Fi Client**: `rst_count`, `ack_flag_number`, `psh_flag_number`, `Variance`, `Magnitue`
- **MQTT Client**: `rst_count`, `psh_flag_number`, `HTTP`, `ack_flag_number`, `Magnitue`
- **Bluetooth Client**: `Magnitue`, `AVG`, `Number`, `Max`, `Tot sum`

#### Top-5 Features Identified per Client (FedProx, $\mu = 0.1$)
- **Wi-Fi Client**: `rst_count`, `psh_flag_number`, `ack_flag_number`, `Variance`, `Magnitue`
- **MQTT Client**: `rst_count`, `psh_flag_number`, `Max`, `Variance`, `syn_count`
- **Bluetooth Client**: `Magnitue`, `Tot sum`, `Tot size`, `Number`, `AVG`

---

### FedAvg vs. FedProx Comparison

Comparing top-5 Jaccard overlap between algorithms on identical data splits:

| Client Pair | FedAvg Jaccard | FedProx Jaccard ($\mu=0.1$) | Difference ($\Delta$) | Shared Features (FedAvg $\rightarrow$ FedProx) |
| :--- | :---: | :---: | :---: | :--- |
| **wifi-mqtt** | **0.6667** | **0.4286** | -0.2381 | {`Magnitue`, `ack`, `psh`, `rst`} $\rightarrow$ {`Variance`, `psh`, `rst`} |
| **wifi-bluetooth** | **0.1111** | **0.1111** | 0.0000 | {`Magnitue`} $\rightarrow$ {`Magnitue`} |
| **mqtt-bluetooth** | **0.1111** | **0.0000** | -0.1111 | {`Magnitue`} $\rightarrow$ *None* |

#### Empirical Takeaways
1. **Strong Protocol Divide**: Both algorithms show very low overlap ($J \le 0.11$) between IP clients (Wi-Fi, MQTT) and non-IP clients (Bluetooth). While IP-based attacks trigger transport flag counts (`rst_count`, `psh_flag_number`), Bluetooth attacks are driven entirely by packet volume and timing statistics (`AVG`, `Number`, `Tot sum`).
2. **Impact of Proximal Regularization**: In this initial run, FedProx ($\mu = 0.1$) did not decrease divergence; rather, it reduced overlap on Wi-Fi/MQTT (from 0.6667 to 0.4286) and MQTT/Bluetooth (from 0.1111 to 0.0000).

---

## 6. Visualizations

### Cross-Client SHAP Divergence (FedAvg)
![Cross-Client SHAP Explanation Divergence (FedAvg)](results/shap_cross_client_comparison.png)

*Recurring features across clients maintain consistent colors (e.g., light blue for `rst_count`, yellow for `psh_flag_number`, green for `Magnitue`), highlighting the visual alignment between Wi-Fi and MQTT and the complete divergence of Bluetooth.*

### Cross-Client SHAP Divergence (FedProx)
![Cross-Client SHAP Explanation Divergence (FedProx)](results/shap_cross_client_comparison_fedprox.png)

---

## 7. Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/Swayam-Swaroop-Sahu/FedXplain-IoMT.git
cd FedXplain-IoMT

pip install -r requirements.txt
```

### Dependency Note
If running on Windows CPU environments with PyTorch 2.5+, install standard binary wheels:
```bash
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
```

---

## 8. Reproduction Guide

Run the full end-to-end pipeline in order:

### Step 1: Centralized Baseline Benchmark
Trains the pooled MLP model for 10 epochs and prints classification metrics:
```bash
python src/centralized.py
```
*Output: Saves model to `models/centralized_baseline.pth`.*

### Step 2: Federated Learning (FedAvg)
Simulates 5 communication rounds across Wi-Fi, MQTT, and Bluetooth clients:
```bash
python src/run_federated.py
```
*Output: Saves model to `models/fedavg_global.pth`.*

### Step 3: SHAP Explanations & Divergence for FedAvg
Computes feature attributions and generates comparison plots:
```bash
python src/explain.py
```
*Outputs: Generates `results/shap_top10_{protocol}.png`, `results/shap_cross_client_comparison.png`, and `results/divergence_report.md`.*

### Step 4: Federated Learning with FedProx
Simulates 5 rounds with proximal penalty ($\mu = 0.1$):
```bash
python src/run_federated_prox.py
```
*Output: Saves model to `models/fedprox_global.pth`.*

### Step 5: SHAP Explanations & Divergence for FedProx
Computes attributions on the FedProx model and creates comparative metrics:
```bash
python src/explain_prox.py
```
*Outputs: Generates `results/shap_top10_{protocol}_fedprox.png`, `results/shap_cross_client_comparison_fedprox.png`, `results/divergence_report_fedprox.md`, and `results/fedavg_vs_fedprox_comparison.md`.*

---

## 9. Repository Structure

```
FedXplain-IoMT/
├── .gitignore                          # Excludes raw data, checkpoints, logs
├── README.md                           # Main project documentation & findings
├── requirements.txt                    # Project dependencies
├── data/
│   ├── README.md                       # CICIoMT2024 dataset & extraction details
│   ├── poc/                            # Proof-of-concept sampled subsets
│   │   ├── wifi_poc.csv                # 4,995 rows, 14 classes
│   │   ├── mqtt_poc.csv                # 2,998 rows, 6 classes
│   │   └── bluetooth_poc.csv           # 1,999 rows, 2 classes (HCI H4)
│   └── study/                          # Full study datasets (scaled for research)
│       ├── wifi_study.csv              # 99,993 rows
│       ├── mqtt_study.csv              # 49,997 rows
│       └── bluetooth_study.csv         # 16,346 rows
├── docs/
│   └── poc_summary.md                  # Concise research summary report
├── models/                             # Saved model weights (.pth) [gitignored]
│   ├── centralized_baseline.pth
│   ├── fedavg_global.pth
│   └── fedprox_global.pth
├── results/                            # Generated plots, tables, reports
│   ├── divergence_report.md            # FedAvg Jaccard similarity table
│   ├── divergence_report_fedprox.md    # FedProx Jaccard similarity table
│   ├── fedavg_vs_fedprox_comparison.md # Side-by-side Jaccard comparison
│   ├── shap_cross_client_comparison.png
│   ├── shap_cross_client_comparison_fedprox.png
│   ├── shap_top10_wifi.png
│   ├── shap_top10_mqtt.png
│   ├── shap_top10_bluetooth.png
│   ├── shap_top10_wifi_fedprox.png
│   ├── shap_top10_mqtt_fedprox.png
│   └── shap_top10_bluetooth_fedprox.png
└── src/                                # Core codebase
    ├── __init__.py
    ├── client.py                       # Flower NumPyClient with proximal support
    ├── model.py                        # IoMTMLP PyTorch neural network
    ├── utils.py                        # Seed fixing, preprocessing, dimension utils
    ├── centralized.py                  # Centralized baseline training script
    ├── run_federated.py                # FedAvg orchestration runner
    ├── run_federated_prox.py           # FedProx orchestration runner
    ├── explain.py                      # SHAP analysis pipeline for FedAvg
    └── explain_prox.py                 # SHAP analysis pipeline for FedProx
```

---

## 10. Scope & Next Steps

### Proof-of-Concept Pilot Scope
The current findings represent an exploratory single-seed pilot ($seed = 42$, 5 rounds, POC subsets) conducted to validate pipeline mechanics and test for observable divergence. **These results serve as preliminary empirical signals, not final claims of statistical significance.**

### Planned Full Study
1. **Multi-Seed Validation**: 10+ random seeds across both POC and full `data/study/` datasets.
2. **Noise Floor Calibration**: Measuring baseline explanation variation caused strictly by sampling noise within identical protocols to determine whether cross-protocol divergence exceeds stochastic noise.
3. **$\mu$ Hyperparameter Grid**: Evaluating FedProx across $\mu \in \{0.001, 0.01, 0.1, 0.5, 1.0\}$.
4. **Alternative Explainers & Aggregators**: Benchmarking Integrated Gradients, KernelSHAP, FedOpt, and SCAFFOLD.
5. **Defense-Ready Metrics**: Developing automated explanation alignment loss terms during federated aggregation.

---

## Citation & Attribution
If you utilize this framework or the extracted CICIoMT2024 partitions, please cite the underlying dataset:
- **CICIoMT2024**: Canadian Institute for Cybersecurity, University of New Brunswick.
- **Repository**: [Swayam-Swaroop-Sahu/FedXplain-IoMT](https://github.com/Swayam-Swaroop-Sahu/FedXplain-IoMT)