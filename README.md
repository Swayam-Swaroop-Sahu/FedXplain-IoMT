# FedXplain-IoMT: Diagnosing Cross-Protocol Explanation Divergence in Federated Intrusion Detection for Medical IoT

[![Python 3.14](https://img.shields.io/badge/Python-3.14.7-blue.svg)](https://www.python.org/)
[![PyTorch 2.14](https://img.shields.io/badge/PyTorch-2.14.1-ee4c2c.svg)](https://pytorch.org/)
[![Flower 1.39](https://img.shields.io/badge/Flower-1.39.0-gold.svg)](https://flower.ai/)
[![SHAP 0.53](https://img.shields.io/badge/SHAP-0.53.0-brightgreen.svg)](https://shap.readthedocs.io/)
[![Dataset: CICIoMT2024](https://img.shields.io/badge/Dataset-CICIoMT2024-purple.svg)](https://www.unb.ca/cic/datasets/iomt-dataset-2024.html)
[![Submission: IEEE Access](https://img.shields.io/badge/Submission-IEEE%20Access-orange.svg)](#citation)

**FedXplain-IoMT** is an empirical research framework investigating **cross-client explanation divergence** in Federated Learning (FL) for Internet of Medical Things (IoMT) intrusion detection, introducing cross-protocol explanation consistency as an essential evaluation axis because conventional global accuracy and macro F1 metrics obscure catastrophic localized client collapses.

---

## Key Findings

Evaluating federated models across heterogeneous physical communication protocols (Wi-Fi, MQTT, Bluetooth) reveals an acute failure mode: global aggregation drives minority-protocol benign classification to near-total failure, even while global detection metrics appear robust.

### Benchmark Summary Table (Ground Truth)

All metrics report evaluations across held-out test partitions ($n=2$ random seeds $\{42, 7\}$) verified against `results/study_results.json`.

| Method | Avg Macro F1 (Mean) | Wi-Fi F1 (Mean) | MQTT F1 (Mean) | Bluetooth F1 (Mean) | Bluetooth Benign Recall (Seed 42 / Seed 7 \| Mean) | Key Takeaway |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Centralized** (Ceiling) | 0.9792 | — | — | — | 0.9681 / 0.9692 \| **0.9687** | Pooled 10-epoch training upper bound |
| **Local-Only** (Control) | 0.9708 | 0.9406 | 0.9828 | 0.9891 | 0.9984 / 0.9990 \| **0.9987** | Isolated per-client training; zero sharing |
| **Cluster-FL** | 0.9665 | 0.9389 | 0.9777 | 0.9827 | 0.9906 / 0.9948 \| **0.9927** | Protocol clustering ({Wi-Fi: 0}, {MQTT, BT: 1}) |
| **FedPer** | 0.9241 | 0.9387 | 0.9345 | 0.8990 | 0.8258 / 0.8721 \| **0.8489** | Shared trunk + private classification head |
| **FedAvg** | 0.7988 | 0.9354 | 0.8314 | 0.6298 | 0.3666 / 0.4238 \| **0.3952** | Baseline FL; catastrophic Bluetooth collapse |
| **FedProx** ($\mu=0.001$) | 0.7680 | 0.9242 | 0.9347 | 0.4451 | 0.2090 / 0.1092 \| **0.1591** | Proximal penalty initiates monotonic collapse |
| **FedProx** ($\mu=0.01$) | 0.7179 | 0.9216 | 0.9027 | 0.3295 | 0.0364 / 0.0333 \| **0.0349** | Severe benign suppression |
| **FedProx** ($\mu=0.05$) | 0.7017 | 0.9093 | 0.8759 | 0.3198 | 0.0192 / 0.0322 \| **0.0257** | Benign recall falls below 3% |
| **FedProx** ($\mu=0.1$) | 0.6959 | 0.9022 | 0.8712 | 0.3143 | 0.0073 / 0.0343 \| **0.0208** | 97%–99% of benign packets falsely flagged |

### Core Empirical Insights

1. **Catastrophic Bluetooth Benign Recall Collapse under FedAvg**: Standard FedAvg degrades Bluetooth benign recall from 0.9987 (Local-only) down to **0.3666** (seed 42) and **0.4238** (seed 7), averaging 0.3952. Over 60% of legitimate Bluetooth medical traffic is falsely flagged as malicious attacks, dropping Bluetooth macro F1 to 0.6298 and overall average macro F1 to 0.7988.
2. **Monotonic Failure of FedProx Proximal Regularization**: Sweeping proximal penalty $\mu \in \{0.001, 0.01, 0.05, 0.1\}$ does not alleviate client drift; it monotonically deepens the collapse. At $\mu=0.1$, Bluetooth benign recall plummets to **0.0073** (seed 42) and **0.0343** (seed 7). Proximal anchoring tightly binds the Bluetooth client's parameters to the global model, which is dominated by majority Wi-Fi gradients (79,994 training flows, 96.4% attack), effectively suppressing minority benign representations.
3. **SHAP Explanation Diagnosis (Calibration vs. Representation)**: Post-hoc attribution using `shap.DeepExplainer` reveals that the collapse is an **output calibration failure**, not a representation divergence. Cross-client feature attribution rank correlation remains exceptionally strong: the Spearman correlation between FedPer and FedAvg feature importance vectors on the Bluetooth client is **$\rho = 0.9888$** across both seeds (and $\ge 0.8856$ across all clients). Both models leverage identical underlying feature importance rankings; however, shared output layer weights shift the classification decision threshold toward the majority attack class.
4. **FedPer Restores Local Calibration**: By decoupling the model into a shared representation trunk (layers 1–2, 4 parameter arrays) and a personalized classification head (layer 3, 2 parameter arrays), FedPer recovers **72.7%** (seed 42) and **77.9%** (seed 7) of the lost Bluetooth benign-recall gap (reaching 0.8258 / 0.8721). FedPer achieves **0.9241** average macro F1 (95.2% of the local-only ceiling) while maintaining high cross-client explanation consistency ($\rho \ge 0.88$ against FedAvg).
5. **Cluster-FL Macro-Segregation**: Segregating clients into protocol-aligned clusters ({Wi-Fi: 0}, {MQTT: 1, Bluetooth: 1}) completely prevents inter-protocol gradient conflict, attaining **0.9665** average macro F1 and **0.9927** Bluetooth benign recall.

---

## Quickstart

### 1. Installation

Set up a Python 3.14 environment (tested on Python 3.14.7, 64-bit CPU) and install dependencies:

```bash
pip install -r requirements.txt
```

*Core pinned packages*: `torch==2.14.1`, `flwr==1.39.0`, `shap==0.53.0`, `scikit-learn==1.9.1`, `pandas==3.0.6`, `scipy==1.18.1`, `streamlit==1.65.0`, `plotly==7.1.0`.

### 2. Launch the Research Dashboard

FedXplain-IoMT provides an offline, interactive Streamlit dashboard for exploring results, attributions, and figures:

```bash
python -m streamlit run app.py
```

*Note: Use `python -m streamlit run app.py` rather than running bare `python app.py` to ensure proper Streamlit execution context.*

### 3. Running Experiment Modules Directly

Individual learning baselines can be executed directly from the repo root:

- **Centralized Ceiling Benchmark**:
  ```bash
  python src/centralized.py
  ```
- **Standard Federated Averaging (FedAvg)**:
  ```bash
  python src/run_federated.py
  ```
- **Proximal Regularized FL (FedProx)**:
  ```bash
  python src/run_federated_prox.py
  ```
- **Personalized Federated Learning (FedPer)**:
  ```bash
  python src/run_fedper.py
  ```
- **Isolated Local-Only Control**:
  ```bash
  python src/run_local_only.py
  ```
- **Full Master Suite Execution**:
  ```bash
  python -m src.run_study_experiments
  ```
- **Zero-Day & Data-Scarcity Generalization Study** (writes only `results/generalization_results.json`):
  ```bash
  python -m src.run_generalization_study
  ```
  Or run it interactively, cell by cell, in `notebooks/generalization_study.ipynb` (needs `pip install ipykernel nbformat` in the same environment).

---

## Generalization Experiments: Zero-Day & Data Scarcity

Local-only beats every federated method on each client's *own* test set (0.9708 vs 0.7988 avg macro F1 for FedAvg). These experiments test whether FL helps where Local-only should be weakest. They are run by `src/run_generalization_study.py`, which never modifies `results/study_results.json` or existing checkpoints. Models are loaded from `models/` if present; otherwise they are retrained with the study's seeds and settings into `models/generalization/`.

### Experiment 1 — Zero-day (novel attack) detection

Each target client's model is scored on attack types that never appear in its training data. These are taken from the other protocols' **held-out test splits** (same seed), so no FL model has seen those exact flows. Inputs are standardized with the target client's own scaler.

**Control:** benign traffic from the same foreign protocols is scored the same way. *Detection margin* = unseen-attack recall − foreign-benign flag rate. A margin ≤ 0 means the model is flagging unfamiliar traffic, not recognising the attack.

| Target | Method | Unseen-attack recall | Foreign benign flagged | Margin | In-dist macro F1 |
|:---|:---|:---:|:---:|:---:|:---:|
| MQTT | Local-only | 0.449 | 0.021 (Wi-Fi) | +0.430 | 0.983 |
| MQTT | FedAvg | **0.922** | 0.338 (Wi-Fi) | **+0.606** | 0.847 |
| MQTT | FedPer | 0.690 | 0.180 (Wi-Fi) | +0.523 | 0.928 |
| MQTT | Cluster-FL | 0.454 | 0.021 (Wi-Fi) | +0.435 | 0.981 |
| Bluetooth | Local-only | 0.228 | 0.79 (Wi-Fi/MQTT) | −0.567 | 0.989 |
| Bluetooth | FedAvg | 0.981 | 1.00 | −0.019 | 0.636 |
| Bluetooth | FedPer | 0.836 | 0.99 | −0.149 | 0.891 |
| Bluetooth | Cluster-FL | 0.449 | 0.12 | **+0.328** | 0.983 |

(Two-seed means; seeds 42 and 7.)

- **MQTT:** sharing gives real zero-day coverage. FedAvg detects unseen Wi-Fi attacks (ICMP/UDP floods, Ping Sweep, ARP spoofing) that MQTT Local-only misses entirely, but its in-distribution F1 drops from 0.983 to 0.847. FedPer sits between the two. Cluster-FL matches Local-only because MQTT is clustered with Bluetooth, not Wi-Fi.
- **Bluetooth:** the high recall of FedAvg and FedPer is not evidence of attack knowledge, because they flag nearly all foreign traffic, benign included. Only Cluster-FL has a positive margin, and it varies between seeds.

### Experiment 2 — Small-clinic sample efficiency

For each client, the training split is subsampled (stratified) to N ∈ {50, 100, 250, 500, 1000, 5000, full}, with 3 draws per seed. The scaler is refit on the N samples, and every model is scored on the full held-out test split. Three arms get the same optimizer budget:

- `local_only`: a fresh IoMTMLP with all 5,057 parameters trained.
- `fedper_head`: the frozen FedPer trunk plus a fresh 33-parameter head. The target client was part of that federation with its full data, so this is an **optimistic upper bound**.
- `fedper_head_unseen`: the frozen trunk from a FedPer federation **without** the target client (a late-joining clinic) plus a fresh head. This is the fair comparison.

Results are being re-run with a larger optimizer budget (≥ 5,000 steps). The first run's 200-step floor left the head-only arms under-trained.

---

## Repository Structure

```
FedXplain-IoMT/
├── app.py                          # Offline interactive 6-tab Streamlit research dashboard
├── access.tex                      # IEEE Access journal article LaTeX source
├── requirements.txt                # Pinned package dependencies (Python 3.14.7)
├── README.md                       # Master project documentation
├── data/
│   ├── README.md                   # Dataset partition documentation and provenance
│   ├── poc/                        # Sampled POC subsets (~10k rows total)
│   └── study/                      # Full study partitions (~166k rows total)
│       ├── wifi_study.csv          # Wi-Fi partition (99,993 rows; 45 features + label)
│       ├── mqtt_study.csv          # MQTT partition (49,997 rows; 45 features + label)
│       └── bluetooth_study.csv     # Bluetooth partition (16,346 rows; 45 features + label)
├── docs/
│   ├── agent_fix_report.md         # Historical QA audit log & environment divergence notes
│   ├── final_results.md            # Comprehensive tabular results and figure catalog
│   ├── qa_findings.md              # Systematic metric correctness audit evidence
│   └── system_architecture.png     # End-to-end framework architecture diagram
├── figures/
│   ├── fig1_detection_macro_f1.png          # Per-client detection quality across methods
│   ├── fig2_spearman_divergence_heatmap.png # Cross-client Spearman rank correlation heatmap
│   ├── fig3_bluetooth_mu_sweep.png          # Bluetooth benign recall vs FedProx mu
│   └── fig4_tradeoff_scatter.png            # Detection performance vs explanation consistency
├── notebooks/
│   └── generalization_study.ipynb  # Interactive runner for the zero-day / data-scarcity study
├── results/
│   ├── study_results.json          # Master JSON payload (single source of truth)
│   ├── generalization_results.json # Zero-day & data-scarcity results (dashboard Tab 6)
│   ├── cluster_fl_results.json     # Teammate Cluster-FL payload for merging
│   └── shap_*.png                  # Pre-computed SHAP summary plots
├── scripts/
│   ├── audit_results.py            # Completeness matrix validator across all runs
│   ├── backfill_completions.py     # Idempotent derived-key backfill utility
│   ├── make_figures.py             # Generates all four 300-DPI publication figures
│   ├── test_dashboard.py           # Headless Streamlit AppTest verification suite
│   └── validate_results.py         # Schema, value-range, and type-assertion validator
└── src/
    ├── __init__.py                 # Package marker
    ├── centralized.py              # Centralized pooled-data training baseline (ceiling)
    ├── client.py                   # Flower client with custom local training and FedProx loss
    ├── config.py                   # Central hyperparameter configuration and constants
    ├── explain.py                  # SHAP DeepExplainer orchestration and metric computation
    ├── explain_prox.py             # FedProx explainability wrapper
    ├── merge_cluster_fl.py         # Utility to merge Cluster-FL into master results JSON
    ├── model.py                    # IoMTMLP PyTorch architecture definition
    ├── run_federated.py            # FedAvg experiment runner
    ├── run_federated_prox.py       # FedProx experiment runner across mu sweep
    ├── run_fedper.py               # FedPer personalized head experiment runner
    ├── run_generalization_study.py # Zero-day & data-scarcity experiments (parallel results file)
    ├── run_local_only.py           # Local-only isolated client runner (control baseline)
    ├── run_study_experiments.py    # Master batch orchestrator for all methods and seeds
    └── utils.py                    # Data ingestion, train/test split, scaling & guardrails
```

---

## Reproducibility & Verification

The repository enforces automated verification gates to ensure complete reproducibility and schema integrity. All five gate scripts must exit with return code `0`:

| Script | Purpose & Verification Scope | Pass Condition |
|:---|:---|:---:|
| `scripts/validate_results.py` | Validates JSON schema structure, data types, metric ranges $[0, 1]$, and asserts required keys for all federated and local runs. | `Exit 0 (PASS)` |
| `scripts/audit_results.py` | Confirms experimental completeness across all methods (Centralized, FedAvg, FedProx, FedPer, Cluster-FL, Local-only), seeds $\{42, 7\}$, and pairwise client attribution matrices. | `Exit 0 (PASS)` |
| `scripts/test_dashboard.py` | Executes headless Streamlit `AppTest` simulation, verifying that all tabs, selector widgets, Plotly figures, and metric containers render without exceptions. | `Exit 0 (PASS)` |
| `scripts/make_figures.py` | Re-generates all four 300-DPI publication-quality PNG figures (`fig1`–`fig4`) directly from `results/study_results.json`. | `Exit 0 (PASS)` |
| `scripts/backfill_completions.py` | Idempotently populates derived summary statistics, pairwise divergence spreads, and noise-floor comparisons without mutating existing raw results. | `Exit 0 (PASS)` |

To run the complete gate pipeline in PowerShell or Bash:

```bash
python scripts/validate_results.py
python scripts/audit_results.py
python scripts/test_dashboard.py
python scripts/make_figures.py
```

---

## Dataset: CICIoMT2024

The study benchmarks on the **CICIoMT2024** dataset (Canadian Institute for Cybersecurity, University of New Brunswick), captured from 40 real IoMT devices. The study partitions cover **19 distinct attack label values** (13 Wi-Fi, 5 MQTT, 1 Bluetooth) spanning DoS, DDoS, Reconnaissance, ARP spoofing, and protocol-specific exploits (MQTT flood/malformed-data attacks and a Bluetooth DoS class).

### Protocol Partition Statistics

Each protocol represents an isolated clinical network silo. Partitions use an 80/20 stratified train/test split (`stratify=y`):

| Protocol Silo | File | Total Flows | Train Flows (80%) | Test Flows (20%) | Attack Flows (%) | Benign Flows (%) | Notes |
|:---|:---|:---:|:---:|:---:|:---:|:---|:---|
| **Wi-Fi** | `wifi_study.csv` | 99,993 | 79,994 | 19,999 | 96.4% | 3.6% | High-throughput clinical telemetry |
| **MQTT** | `mqtt_study.csv` | 49,997 | 39,997 | 10,000 | 63.2% | 36.8% | Bedside monitors & telemetry brokers |
| **Bluetooth** | `bluetooth_study.csv` | 16,346 | 13,076 | 3,270 | 41.2% | 58.8% | Extracted from HCI H4 `.pcap`; non-applicable TCP/IP flags zeroed |
| **Total** | — | **166,336** | **133,067** | **33,269** | — | — | Exactly 45 flow features asserted |

---

## Architectural Guardrails & Rules

The codebase enforces four strict technical invariants:

1. **45-Feature Assertion**: Every data partition must contain exactly 45 network flow features (stripped of non-generalizable network identifiers such as IP addresses, MAC addresses, and timestamps). Any missing or superfluous column triggers an immediate assertion failure in `src/utils.py`.
2. **Train-Only Scaler Fitting**: `StandardScaler` is fitted strictly on the training partition ($80\%$) and applied via `.transform()` to the test partition ($20\%$). Fitting scalers on pooled or test data is strictly prohibited to prevent data snooping.
3. **Offline JSON-Driven Dashboard**: The Streamlit user interface (`app.py`) functions purely as a visualization renderer reading from `results/study_results.json` (and `results/generalization_results.json` for the Zero-Day & Data Scarcity tab). It never triggers background training, SHAP computations, or model inference at render time.
4. **Strict Results Schema**: All experiment scripts write directly to or merge with `results/study_results.json` adhering to the schema checked by `validate_results.py`. Hand-edited or unverifiable metrics are rejected.

---

## Hardware & Environment Specifications

- **Python Version**: 3.14.7 (64-bit)
- **Deep Learning Framework**: PyTorch 2.14.1 (CPU-only execution)
- **Federated Orchestration**: Flower 1.39.0
- **Explainability**: SHAP 0.53.0 (`shap.DeepExplainer`)
- **Seeds Tested**: $n=2$ seeds ($\text{seed} \in \{42, 7\}$)
- **Cross-Platform Numerical Note**: As documented in `docs/agent_fix_report.md`, minor floating-point divergence ($\le \pm 0.1$ macro F1) can manifest across disparate OS/BLAS/MKL linear algebra implementations due to differences in CPU SIMD vectorization. All claims in this repository are directional and evaluated against calibrated seed-to-seed noise floors.

---

## Citation

If you utilize this codebase, the experimental protocol, or the processed CICIoMT2024 partitions, please cite:

```bibtex
@software{sahu2026fedxplain,
  author    = {Sahu, Swayam Swaroop},
  title     = {FedXplain-IoMT: Diagnosing Cross-Protocol Explanation Divergence in Federated Intrusion Detection for Medical IoT},
  year      = {2026},
  url       = {https://github.com/Swayam-Swaroop-Sahu/FedXplain-IoMT}
}
```

Please also cite the underlying CICIoMT2024 benchmark:

```bibtex
@article{dadkhah2024towards,
  author    = {Dadkhah, Sajjad and Mahdikhani, Hamid and Danso, Philip K. and Zohourian, Amirhossein and Truong, Kevin A. and Ghorbani, Ali A.},
  title     = {Towards the Development of a Realistic Multidimensional IoMT Dataset for Machine Learning-Based Intrusion Detection},
  journal   = {IEEE Access},
  volume    = {12},
  pages     = {31718--31733},
  year      = {2024},
  doi       = {10.1109/ACCESS.2024.3371077}
}
```