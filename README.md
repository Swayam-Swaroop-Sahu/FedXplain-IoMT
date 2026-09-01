# FedXplain-IoMT: Federated Learning & Explainable AI for Heterogeneous Medical IoT

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-1.5%2B-gold.svg)](https://flower.ai/)
[![SHAP](https://img.shields.io/badge/SHAP-0.42%2B-brightgreen.svg)](https://shap.readthedocs.io/)
[![Dataset](https://img.shields.io/badge/Dataset-CICIoMT2024-purple.svg)](https://www.unb.ca/cic/datasets/iomt-dataset-2024.html)
[![Dashboard](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

**FedXplain-IoMT** is an empirical research framework investigating **cross-client explanation divergence** in Federated Learning (FL) for Internet of Medical Things (IoMT) intrusion detection.

While conventional federated evaluation focuses exclusively on global accuracy and F1 scores, FedXplain-IoMT evaluates whether a converged global model relies on **fundamentally disjoint decision logic** across clients operating on heterogeneous physical network protocols (Wi-Fi, MQTT, Bluetooth). Using model-agnostic feature attributions (SHAP) and set-theoretic overlap metrics (Jaccard Index), this project provides quantitative and visual evidence of explanatory drift in privacy-preserving medical device environments.

---

## Table of Contents
- [1. Research Motivation & Problem Statement](#1-research-motivation--problem-statement)
- [2. Dataset: CICIoMT2024](#2-dataset-ciciomt2024)
- [3. System Architecture](#3-system-architecture)
- [4. Methodology & Algorithms](#4-methodology--algorithms)
- [5. Experimental Results](#5-experimental-results)
  - [POC-Scale Results (Initial Pilot)](#poc-scale-results-initial-pilot)
  - [Study-Scale Results (Extended Multi-Round Pilot)](#study-scale-results-extended-multi-round-pilot)
  - [Class-Imbalance Diagnostic & Rigor](#class-imbalance-diagnostic--rigor)
  - [Preliminary Noise Floor Analysis (n=2 Seeds)](#preliminary-noise-floor-analysis-n2-seeds)
  - [What This Does NOT Yet Show](#what-this-does-not-yet-show)
- [6. Interactive Research Dashboard](#6-interactive-research-dashboard)
- [7. Reproducibility Guide](#7-reproducibility-guide)
- [8. Project Status & Roadmap](#8-project-status--roadmap)
- [9. Repository Structure](#9-repository-structure)
- [10. Citation & Attribution](#10-citation--attribution)

---

## 1. Research Motivation & Problem Statement

In clinical healthcare environments, IoMT devices (infusion pumps, patient monitors, telemetry sensors) communicate across diverse physical and transport protocols:
- **Wi-Fi** (high-throughput transport for telemetry and diagnostic imaging)
- **MQTT** (lightweight publish-subscribe telemetry for bedside patient monitors)
- **Bluetooth / BLE** (short-range point-to-point peripheral communication)

Due to strict healthcare privacy regulations (HIPAA, GDPR), raw telemetry from hospital subnets cannot be pooled centrally. Federated Learning enables distributed model training across hospital silos without transmitting patient data.

### The Explanation Divergence Problem
Standard evaluation assumes that achieving high classification metrics (e.g., F1 > 0.98) across all clients implies consistent model behavior. However:
1. **Protocol Asymmetry**: IP-based attacks exhibit distinct TCP/IP flag anomalies, whereas Bluetooth attacks manifest purely in packet frequency and flow timing.
2. **Global Representation vs. Local Logic**: A single global neural network parameter set ($\theta^*$) may achieve uniform predictive accuracy while relying on disjoint feature subsets when explaining predictions locally.
3. **Clinical Trust**: If a hospital cybersecurity analyst on a Wi-Fi ward observes attacks flagged by TCP reset anomalies while a Bluetooth ward flags predictions via volumetric flow aggregates, automated containment policies cannot be trusted without explanation alignment.

---

## 2. Dataset: CICIoMT2024

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

---

## 3. System Architecture

```
                       +-------------------------------+
                       |   Central Server / Orchestrator|
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
| 100 Benign Ref  |           | 100 Benign Ref  |           | 100 Benign Ref  |
| 200 Attack Eval |           | 200 Attack Eval |           | 200 Attack Eval |
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
                     |  - 2-Seed Stability Noise Floor   |
                     +-----------------------------------+
```

---

## 4. Methodology & Algorithms

### Centralized Configuration (`src/config.py`)
All experimental parameters are strictly managed via `src/config.py`:
- `N_ROUNDS = 10`
- `N_LOCAL_EPOCHS = 3`
- `SEEDS = [42, 7]`
- `MU_VALUES = [0.01, 0.1]`
- `DATA_DIR = "data/study"`

### Model Architecture (`src/model.py`)
A unified, lightweight 3-layer Multilayer Perceptron (`IoMTMLP`):
$$\text{Linear}(45, 64) \rightarrow \text{ReLU} \rightarrow \text{Dropout}(0.2) \rightarrow \text{Linear}(64, 32) \rightarrow \text{ReLU} \rightarrow \text{Linear}(32, 1) \rightarrow \text{Sigmoid}$$

### Federated Optimization
- **Flower NumPyClient (`src/client.py`)**: Executes local mini-batch SGD on per-protocol partitions.
- **FedAvg (`src/run_federated.py`)**: Weighted aggregation based on client sample counts $n_k$:
  $$w_{t+1} = \sum_{k=1}^K \frac{n_k}{N} w_{t+1}^k$$
- **FedProx (`src/run_federated_prox.py`)**: Penalizes local drift from global parameters $w_t$:
  $$\mathcal{L}_{\text{FedProx}}(w; w_t) = \mathcal{L}_{\text{BCE}}(w) + \frac{\mu}{2} \|w - w_t\|_2^2$$

### Post-Hoc Explainability (`src/explain.py`)
- **Explainer**: `shap.DeepExplainer` on the converged global neural network.
- **Reference Set**: 100 benign samples ($y = 0$) per client.
- **Evaluation Set**: Up to 200 attack samples ($y = 1$) per client.
- **Divergence Metric**: Pairwise Jaccard similarity of Top-5 feature sets $S_A, S_B$:
  $$J(S_A, S_B) = \frac{|S_A \cap S_B|}{|S_A \cup S_B|}$$

---

## 5. Experimental Results

### POC-Scale Results (Initial Pilot)
*Evaluated on `data/poc` subsets (5 rounds, 2 local epochs, seed=42):*

| Algorithm | Wi-Fi Macro F1 | MQTT Macro F1 | BT Macro F1 | Macro Avg F1 | Top-5 Jaccard (Wi-Fi/MQTT) | Top-5 Jaccard (Wi-Fi/BT) | Top-5 Jaccard (MQTT/BT) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized Baseline** | — | — | — | **0.9875** | — | — | — |
| **FedAvg** | 0.8604 | 0.8607 | 0.7198 | **0.8560** | **0.6667** | **0.1111** | **0.1111** |
| **FedProx ($\mu=0.01$)** | 0.8604 | 0.8990 | 0.8058 | **0.8974** | **0.4286** | **0.1111** | **0.0000** |

---

### Study-Scale Results (Extended Multi-Round Pilot)
*Evaluated on full `data/study` datasets (166,000+ flows, 10 rounds, 3 local epochs, seed=42):*

| Algorithm | Wi-Fi Macro F1 | MQTT Macro F1 | BT Macro F1 | Macro Avg F1 | Top-5 Jaccard (Wi-Fi/MQTT) | Top-5 Jaccard (Wi-Fi/BT) | Top-5 Jaccard (MQTT/BT) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized Baseline** | — | — | — | **0.9794** | — | — | — |
| **FedAvg** | 0.9353 | 0.8351 | 0.6106 | **0.7937** | **0.2500** | **0.0000** | **0.0000** |
| **FedProx ($\mu=0.01$)** | 0.9138 | 0.9250 | 0.3379 | **0.7256** | **0.4286** | **0.1111** | **0.2500** |
| **FedProx ($\mu=0.1$)** | 0.8928 | 0.8664 | 0.3343 | **0.6979** | **0.6667** | **0.2500** | **0.2500** |

---

### Class-Imbalance Diagnostic & Rigor
During rigorous auditing, we identified that **Wi-Fi binary F1 (>0.99) is inflated by class imbalance** (96.4% attack samples, 3.6% benign).
- Under FedAvg (Study scale), Wi-Fi yields:
  - **Attack Precision**: 0.9968 | **Attack Recall**: 0.9935 | **Binary F1**: 0.9951
  - **Benign Precision**: 0.8399 | **Benign Recall**: 0.9142 | **Macro F1**: 0.9353
- **Methodological Standard**: Macro F1 and per-class precision/recall are now reported by default across all experiments to prevent false optimism.

---

### Preliminary Noise Floor Analysis (n=2 Seeds)
Comparing Top-5 feature attribution stability between random seeds ($seed = 42$ vs $seed = 7$):
- **FedAvg**: Seed-to-seed top-5 Jaccard overlap = **0.4286** across all three clients.
- **FedProx ($\mu=0.01$)**: Wi-Fi overlap = **0.6667**, MQTT overlap = **0.4286**, Bluetooth overlap = **0.6667**.
- **FedProx ($\mu=0.1$)**: Wi-Fi overlap = **0.6667**, MQTT overlap = **0.2500**, Bluetooth overlap = **0.2500**.

---

### What This Does NOT Yet Show
To maintain strict scientific integrity:
1. **Preliminary Seed Count**: $n=2$ seeds provide a directional stability check, not a fully converged 10-seed confidence interval.
2. **Hyperparameter Grid**: Explored $\mu \in \{0.01, 0.1\}$. A continuous logarithmic sweep remains for the full publication study.
3. **No Statistical Significance Claims**: Findings demonstrate visible explanatory divergence under protocol heterogeneity, motivating the full formal study.

---

## 6. Interactive Research Dashboard

An interactive Streamlit dashboard is included for live presentations and inspection:

```bash
pip install streamlit plotly pandas
streamlit run app.py
```

### Dashboard Features:
1. **Overview & Architecture**: Interactive system diagram and problem framing.
2. **Detection Performance**: Filterable performance table with Macro F1 vs Binary F1 comparisons.
3. **Explanation Divergence**: Interactive Plotly bar charts of SHAP attributions + Jaccard heatmap.
4. **FedAvg vs FedProx Comparison**: Side-by-side performance and attribution drift metrics.
5. **Methodology & Limitations**: Transparent checklist of verified vs planned items.

*See [`docs/ui_guide.md`](docs/ui_guide.md) for full dashboard documentation.*

---

## 7. Reproducibility Guide

All results can be regenerated from scratch using the centralized configuration:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the complete study experiment suite (Centralized + 6 FL runs + SHAP + JSON export)
python src/run_study_experiments.py

# 3. Launch the dashboard
streamlit run app.py
```

---

## 8. Project Status & Roadmap

- [x] Phase 1: Shared preprocessing & centralized baseline benchmark.
- [x] Phase 2: Multi-client federated training with Flower (FedAvg).
- [x] Phase 3: Post-hoc explainability pipeline with DeepExplainer & Jaccard index.
- [x] Phase 4: FedProx proximal term integration ($\mu=0.01, 0.1$).
- [x] Phase 5: Diagnostic robustness check (Wi-Fi class imbalance, weight norms, guardrails).
- [x] Phase 6: Scaled-up study run on `data/study` with centralized `config.py` and 2-seed pilot.
- [x] Phase 7: Interactive Streamlit research dashboard (`app.py`).
- [ ] Phase 8 (Full Study): 10-seed noise-floor calibration & full logarithmic $\mu$ sweep.
- [ ] Phase 9 (Full Study): Advanced aggregators (SCAFFOLD, FedNova, FedOpt) and gradient-based explainers.

---

## 9. Repository Structure

```
FedXplain-IoMT/
├── .gitignore                          # Git tracking exclusions
├── README.md                           # Master project documentation
├── requirements.txt                    # Project dependencies
├── app.py                              # Interactive Streamlit dashboard
├── data/
│   ├── README.md                       # CICIoMT2024 dataset documentation
│   ├── poc/                            # POC subsets (~10k rows total)
│   └── study/                          # Full study datasets (~166k rows total)
├── docs/
│   ├── poc_summary.md                  # POC pilot summary report
│   └── ui_guide.md                     # Dashboard presentation guide
├── models/                             # Saved model weights (.pth) [gitignored]
├── results/                            # Generated experiment artifacts
│   ├── study_results.json              # Master JSON payload for dashboard
│   ├── divergence_report.md            # POC FedAvg divergence report
│   ├── divergence_report_fedprox.md    # POC FedProx divergence report
│   ├── fedavg_vs_fedprox_comparison.md # POC comparative analysis
│   └── *.png                           # SHAP attribution visualizations
└── src/                                # Core codebase
    ├── __init__.py
    ├── config.py                       # Centralized experiment configuration
    ├── client.py                       # Flower NumPyClient with proximal regularization
    ├── model.py                        # IoMTMLP neural network architecture
    ├── utils.py                        # Preprocessing, caching & 45-feature guardrails
    ├── centralized.py                  # Centralized baseline benchmark
    ├── run_federated.py                # FedAvg orchestration runner
    ├── run_federated_prox.py           # FedProx orchestration runner
    ├── run_study_experiments.py        # Master suite for study experiments
    ├── explain.py                      # SHAP explainability pipeline
    └── explain_prox.py                 # FedProx explainability wrapper
```

---

## 10. Citation & Attribution

If you utilize this framework or the extracted CICIoMT2024 partitions, please cite:
- **CICIoMT2024 Dataset**: Canadian Institute for Cybersecurity, University of New Brunswick.
- **Repository**: [Swayam-Swaroop-Sahu/FedXplain-IoMT](https://github.com/Swayam-Swaroop-Sahu/FedXplain-IoMT)