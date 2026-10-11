# FedXplain-IoMT: Federated Learning & Explainable AI for Heterogeneous Medical IoT

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-1.5%2B-gold.svg)](https://flower.ai/)
[![SHAP](https://img.shields.io/badge/SHAP-0.53.0%2B-brightgreen.svg)](https://shap.readthedocs.io/)
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
  - [Final Study Results](#final-study-results)
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

![FedXplain-IoMT System Architecture](docs/system_architecture.png)

*Figure 1. End-to-end FedXplain-IoMT system architecture: Phase 1 (Federated Training across heterogeneous protocol clients with FedAvg/FedProx aggregation) and Phase 2 (Post-Training Explainability & Cross-Client Divergence Analysis via SHAP and Jaccard similarity).*

---

## 4. Methodology & Algorithms

### Centralized Configuration (`src/config.py`)
All experimental parameters are strictly managed via `src/config.py`:
- `N_ROUNDS = 10`
- `N_LOCAL_EPOCHS = 3`
- `SEEDS = [42, 7]`
- `MU_VALUES = [0.001, 0.01, 0.05, 0.1]`
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
- **FedPer (`src/run_fedper.py`)**: Personalized FL where the first 4 parameter arrays (shared trunk) are aggregated, while the last 2 arrays (private output head) remain local.

### Post-Hoc Explainability (`src/explain.py`)
- **Explainer**: `shap.DeepExplainer` on the converged global neural network.
- **Reference Set**: 100 benign samples ($y = 0$) per client.
- **Evaluation Set**: Up to 200 attack samples ($y = 1$) per client.
- **Divergence Metric**: Pairwise Jaccard similarity of Top-5 feature sets $S_A, S_B$:
  $$J(S_A, S_B) = \frac{|S_A \cap S_B|}{|S_A \cup S_B|}$$

---

## 5. Experimental Results

### Final Study Results
All numbers are read directly from `results/study_results.json` (single source of truth). The study covers **2 seeds (42, 7)** — sample counts are too small for statistical inference, so every claim below is *directional*, and every between-method divergence gap is read against the **seed-to-seed noise floor** (the same statistic recomputed across seeds; a gap smaller than its noise-floor cell is within seed noise). `figures/` is regenerated with `python scripts/make_figures.py`; completeness of the results file itself is verified with `python scripts/audit_results.py`.

Methods compared: **FedAvg** (shared model), **FedProx** (shared model + proximal term, mu in {0.001, 0.01, 0.05, 0.1}), **FedPer** (shared trunk + private per-client head), **Local-only** (no sharing, control). Centralized pooling is the upper reference. Cluster-FL is an empty slot pending a teammate's port.

See [`docs/final_results.md`](docs/final_results.md) for the complete result tables and figures.

---

## 6. Interactive Research Dashboard

An interactive Streamlit dashboard is included for live presentations and inspection:

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Dashboard Features:
1. **Executive Summary**: One-sentence thesis; 4 KPI cards computed from the JSON; Figure 4 embedding; two-column "What we found / What we didn't prove yet".
2. **Detection**: Unified comparison table (FedAvg, FedProx×4 mu, FedPer, Local-only, Centralized); Bluetooth benign-recall table with any cell < 0.5 auto-flagged; detailed per-seed metrics table; Bluetooth collapse callout.
3. **Explanation Divergence**: Aggregator selector covering ALL methods (fedavg, fedprox×4, fedper, local-only, Cluster-FL); top-10 SHAP bar chart + Jaccard heatmap; full-metrics table (Jaccard + Spearman + cosine per pair, both seeds) from `divergence_summary` PLUS noise-floor reference row; divergence_scope caption; for fedper runs show `fedper_vs_fedavg`/`fedper_vs_local_only` per-client tables; selecting Cluster-FL (or any missing run) renders a graceful placeholder.
4. **Methods Compared**: Centerpiece = Bluetooth benign recall vs mu curve (x categorical: FedAvg(μ=0), 0.001, 0.01, 0.05, 0.1; two lines one per seed; local-only as horizontal teal dashed reference line; FedPer as navy dotted reference); secondary chart = avg macro F1 per method (two-seed mean, error bars = seed spread); short written findings.
5. **Methodology & Limitations**: 2 seeds (directional), 10 federated + 2 local-only + 2 FedPer runs, full 45-dim SHAP vectors, noise floor across ALL aggregators; Cluster-FL as "in progress — port pending" placeholder with pointer to documented schema; seed-to-seed stability table extended to all aggregators; architecture diagram (`docs/system_architecture.png`).

*See [`docs/ui_guide.md`](docs/ui_guide.md) for full dashboard documentation.*

---

## 7. Reproducibility Guide

All results can be regenerated from scratch using the centralized configuration:

```bash
# 1. Install dependencies (exact pinned versions, Python 3.14)
pip install -r requirements.txt

# 2. Run the complete experiment suite (Centralized + FL runs + local-only control + SHAP + JSON export)
python src/run_study_experiments.py

# 3. (Optional) Verify completeness of results
python scripts/audit_results.py

# 4. Generate figures
python scripts/make_figures.py

# 5. Launch the dashboard
streamlit run app.py
```

### Cluster-FL Slot (for teammates)
A teammate's Cluster-FL port will drop runs at `results/cluster_fl_results.json` following the documented schema (`CLUSTER_FL_RUN_SCHEMA` in `src/run_study_experiments.py`) and merge with `python src/merge_cluster_fl.py` — no code changes needed. Until then the dashboard and the divergence summary show an explicit placeholder (`slot empty - cluster_fl results pending`).

### Environment Pinning
`requirements.txt` contains **exact pinned versions** (`==`) of every dependency. All reported numbers in this README and in `results/study_results.json` were generated with these versions on **Python 3.14** (CPU-only; the model is a small MLP so no GPU is required). Re-running the suite with different dependency versions may produce slightly different numbers due to library-level numerical changes (e.g., PyTorch kernel updates, pandas/scikit-learn behavioral changes), even with identical seeds.

---

## 8. Project Status & Roadmap

- [x] Phase 1: Shared preprocessing & centralized baseline benchmark.
- [x] Phase 2: Multi-client federated training with Flower (FedAvg).
- [x] Phase 3: Post-hoc explainability pipeline with DeepExplainer & Jaccard index.
- [x] Phase 4: FedProx proximal term integration ($\mu=0.001, 0.01, 0.05, 0.1$).
- [x] Phase 5: Diagnostic robustness check (Wi-Fi class imbalance, weight norms, guardrails).
- [x] Phase 6: Scaled-up study run on `data/study` with centralized `config.py` and 2-seed pilot.
- [x] Phase 7: Interactive Streamlit research dashboard (`app.py`).
- [ ] Phase 8 (Full Study): 10-seed noise-floor calibration & full logarithmic $\mu$ sweep.
- [ ] Phase 9 (Full Study): Advanced aggregators (SCAFFOLD, FedNova, FedOpt) and gradient-based explainers.
- [ ] Phase 10: Teammate's Cluster-FL port (see `docs/final_results.md` for slot documentation).

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
│   │   ├── bluetooth_poc.csv
│   │   ├── mqtt_poc.csv
│   │   └── wifi_poc.csv
│   └── study/                          # Full study datasets (~166k rows total)
│       ├── bluetooth_study.csv
│       ├── mqtt_study.csv
│       └── wifi_study.csv
├── docs/
│   ├── agent_fix_report.md             # Agent fix report with completion pass
│   ├── final_results.md                # Final result tables and figures
│   ├── poc_summary.md                  # POC pilot summary report
│   ├── system_architecture.png         # System architecture diagram
│   └── ui_guide.md                     # Dashboard presentation guide
├── figures/                            # Generated publication figures
│   ├── fig1_detection_macro_f1.png
│   ├── fig2_spearman_divergence_heatmap.png
│   ├── fig3_bluetooth_mu_sweep.png
│   └── fig4_tradeoff_scatter.png
├── models/                             # Saved model weights (.pth) [gitignored]
├── results/                            # Generated experiment artifacts
│   ├── audit_results.py                # Completeness checker
│   ├── backfill_completions.py         # Idempotent backfiller
│   ├── divergence_report.md            # POC FedAvg divergence report
│   ├── divergence_report_fedprox.md    # POC FedProx divergence report
│   ├── fedavg_vs_fedprox_comparison.md # POC comparative analysis
│   ├── make_figures.py                 # Figure generation script
│   ├── shap_*.png                      # SHAP attribution visualizations
│   ├── study_results.json              # Master JSON payload for dashboard
│   └── test_dashboard.py               # Dashboard verification script
├── scripts/
│   ├── audit_results.py                # Completeness matrix verification
│   ├── backfill_completions.py         # Idempotent backfill for missing pieces
│   ├── make_figures.py                 # Regenerate all figures
│   └── test_dashboard.py               # Streamlit dashboard test script
└── src/                                # Core codebase
    ├── __init__.py
    ├── config.py                       # Centralized experiment configuration
    ├── centralized.py                  # Centralized baseline benchmark
    ├── client.py                       # Flower NumPyClient with proximal regularization
    ├── explain.py                      # SHAP explainability pipeline
    ├── explain_prox.py                 # FedProx explainability wrapper
    ├── merge_cluster_fl.py             # Merge teammate-produced Cluster-FL runs
    ├── model.py                        # IoMTMLP neural network architecture
    ├── run_federated.py                # FedAvg orchestration runner
    ├── run_federated_prox.py           # FedProx orchestration runner
    ├── run_fedper.py                   # FedPer (personalized FL) orchestration runner
    ├── run_local_only.py               # Local-only (no-sharing) control baseline
    ├── run_study_experiments.py        # Master suite for study experiments
    └── utils.py                        # Preprocessing, caching & 45-feature guardrails
```

---

## 10. Citation & Attribution

If you utilize this framework or the extracted CICIoMT2024 partitions, please cite:
- **CICIoMT2024 Dataset**: Canadian Institute for Cybersecurity, University of New Brunswick.
- **Repository**: [Swayam-Swaroop-Sahu/FedXplain-IoMT](https://github.com/Swayam-Swaroop-Sahu/FedXplain-IoMT)