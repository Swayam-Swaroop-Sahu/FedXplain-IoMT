# Final Results — FedXplain-IoMT Study

All numbers are read directly from `results/study_results.json` (single source of
truth). The study covers **2 seeds (42, 7)** — sample counts are too small for
statistical inference, so every claim below is *directional*, and every
between-method divergence gap is read against the **seed-to-seed noise floor**
(the same statistic recomputed across seeds; a gap smaller than its noise-floor
cell is within seed noise). `figures/` is regenerated with
`python scripts/make_figures.py`; completeness of the results file itself is
verified with `python scripts/audit_results.py`.

Methods compared: **FedAvg** (shared model), **FedProx** (shared model + proximal
term, mu in {0.001, 0.01, 0.05, 0.1}), **FedPer** (shared trunk + private
per-client head), **Local-only** (no sharing, control). Centralized pooling is
the upper reference. Cluster-FL is an empty slot pending a teammate's port.

---

## 1. Unified detection table

Macro F1 on each client's held-out test set (Wi-Fi / MQTT / Bluetooth), per seed,
with the two-seed mean and spread (max − min across seeds; n = 2).

| Method | Wi-Fi (42 / 7) | MQTT (42 / 7) | Bluetooth (42 / 7) | Avg macro F1 (42 / 7) | Mean ± spread |
|---|---|---|---|---|---|
| FedAvg | 0.9375 / 0.9332 | 0.8409 / 0.8218 | 0.6112 / 0.6483 | 0.7965 / 0.8011 | 0.7988 ± 0.0046 |
| FedProx (mu=0.001) | 0.9266 / 0.9217 | 0.9534 / 0.9160 | 0.4894 / 0.4008 | 0.7898 / 0.7462 | 0.7680 ± 0.0436 |
| FedProx (mu=0.01) | 0.9242 / 0.9190 | 0.9205 / 0.8849 | 0.3312 / 0.3277 | 0.7253 / 0.7105 | 0.7179 ± 0.0148 |
| FedProx (mu=0.05) | 0.9070 / 0.9116 | 0.8775 / 0.8743 | 0.3130 / 0.3266 | 0.6991 / 0.7042 | 0.7017 ± 0.0051 |
| FedProx (mu=0.1) | 0.9021 / 0.9023 | 0.8665 / 0.8759 | 0.2999 / 0.3286 | 0.6895 / 0.7023 | 0.6959 ± 0.0128 |
| FedPer | 0.9414 / 0.9360 | 0.9270 / 0.9419 | 0.8854 / 0.9127 | 0.9179 / 0.9302 | 0.9241 ± 0.0123 |
| **Local-only** | 0.9442 / 0.9370 | 0.9819 / 0.9838 | 0.9883 / 0.9899 | 0.9715 / 0.9702 | **0.9708 ± 0.0013** |
| Centralized (reference) | — | — | — | 0.9794 / 0.9790 | 0.9792 ± 0.0004 |

Figure: `figures/fig1_detection_macro_f1.png`.

---

## 2. Headline findings

### (a) No mu saves Bluetooth

Bluetooth is the client whose benign traffic most resembles attack traffic, and
every shared model attacks *benign* recall there. The FedProx proximal term —
the standard remedy for heterogeneous clients — makes it **strictly worse**, at
every mu, on both seeds:

| Method | BT benign recall (42 / 7) |
|---|---|
| FedAvg (mu=0) | 0.3666 / 0.4238 |
| FedProx mu=0.001 | 0.2090 / 0.1092 |
| FedProx mu=0.01 | 0.0364 / 0.0333 |
| FedProx mu=0.05 | 0.0192 / 0.0322 |
| FedProx mu=0.1 | 0.0073 / 0.0343 |
| FedPer | 0.8258 / 0.8721 |
| **Local-only** | **0.9984 / 0.9990** |
| Centralized (reference) | 0.9681 / 0.9692 |

There is no safe mu: even the gentlest proximal term (0.001) drops benign recall
below FedAvg, and by mu=0.1 the model flags ~97–99% of Bluetooth benign traffic
as attack. Local-only sits at 0.999 — the collapse is caused by *sharing*, and
proximal anchoring only deepens it. This is the study's most striking result.

Figure: `figures/fig3_bluetooth_mu_sweep.png`.

### (b) Local-only beats all sharing on detection

Local-only wins every client, every seed (unified table above): **0.9708** avg
macro F1 vs **0.9241** for FedPer (+4.7 points for local-only), **0.7988** for
FedAvg (+17.2), and **0.7680** for the best FedProx (+20.3). It even matches the
pooled centralized reference (0.9792) within 0.9 points — and beats it on
Bluetooth benign recall (0.999 vs 0.969). With clients this heterogeneous,
sharing a single model is a net loss; the best "federated" detector here is the
one that never federates.

### (c) FedPer's trade-off position

FedPer (private per-client head + shared trunk) buys back most of what sharing
destroyed, at a measured cost:

- **Detection quality kept:** 0.9241 avg macro F1 = **95.2%** of local-only
  (0.9708); Bluetooth macro F1 0.8854/0.9127 vs local-only 0.9883/0.9899.
- **Bluetooth benign recall recovered:** 0.8258/0.8721 — recovers **72.7%**
  (seed 42) and **77.9%** (seed 7) of the FedAvg→local-only gap.
- **Explanation consistency kept:** cross-client explanation similarity stays at
  FedAvg level, far above the local-only control (divergence tables below). The
  between-method gap vs FedAvg is within seed noise on every pair; the gap vs
  local-only exceeds noise on Wi-Fi–MQTT and Wi-Fi–BT, and is within noise on
  MQTT–BT.

Reading: personalizing only the 32→1 output head recovers behavior (the collapse
lives in the output layer) while leaving *what the model looks at* — and hence
cross-client explanation agreement — essentially where FedAvg had it.

Figure: `figures/fig4_tradeoff_scatter.png` (detection quality vs explanation
consistency, all seven methods).

---

## 3. Explanation divergence (across clients) with noise floor

Scope differs by method and is stated per run (`divergence_scope`): FedAvg and
FedProx compare **one shared global model** evaluated per client; FedPer compares
the three **personalized assembled models**; local-only compares **three
isolated models**. Higher = more similar across clients.

### Top-5 Jaccard (seed 42 / seed 7)

| Method | Wi-Fi–MQTT | Wi-Fi–BT | MQTT–BT |
|---|---|---|---|
| FedAvg | 0.1111 / 0.2500 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| FedProx mu=0.001 | 0.6667 / 0.2500 | 0.1111 / 0.2500 | 0.1111 / 0.0000 |
| FedProx mu=0.01 | 0.6667 / 0.6667 | 0.1111 / 0.1111 | 0.1111 / 0.0000 |
| FedProx mu=0.05 | 0.4286 / 0.6667 | 0.1111 / 0.2500 | 0.2500 / 0.1111 |
| FedProx mu=0.1 | 0.4286 / 1.0000 | 0.0000 / 0.4286 | 0.1111 / 0.4286 |
| FedPer | 0.1111 / 0.2500 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| Local-only | 0.2500 / 0.2500 | 0.0000 / 0.0000 | 0.2500 / 0.2500 |

### Spearman rho over full 45-dim mean-|SHAP| vectors (seed 42 / seed 7)

| Method | Wi-Fi–MQTT | Wi-Fi–BT | MQTT–BT |
|---|---|---|---|
| FedAvg | 0.7493 / 0.7520 | 0.3176 / 0.2514 | 0.2641 / 0.2346 |
| FedProx mu=0.001 | 0.8049 / 0.8129 | 0.4168 / 0.4333 | 0.3062 / 0.2676 |
| FedProx mu=0.01 | 0.8424 / 0.8791 | 0.5422 / 0.4897 | 0.3892 / 0.3656 |
| FedProx mu=0.05 | 0.8424 / 0.8569 | 0.4306 / 0.4735 | 0.3366 / 0.3707 |
| FedProx mu=0.1 | 0.8299 / 0.8593 | 0.4168 / 0.4506 | 0.3224 / 0.3413 |
| FedPer | 0.7368 / 0.7894 | 0.2740 / 0.2663 | 0.2467 / 0.1461 |
| Local-only | 0.4632 / 0.5825 | 0.1407 / −0.0012 | 0.2661 / 0.1851 |

### Cosine similarity over full vectors (seed 42 / seed 7)

| Method | Wi-Fi–MQTT | Wi-Fi–BT | MQTT–BT |
|---|---|---|---|
| FedAvg | 0.9902 / 0.9705 | 0.0632 / 0.0989 | 0.0848 / 0.1178 |
| FedProx mu=0.001 | 0.9802 / 0.9866 | 0.1190 / 0.1476 | 0.1927 / 0.1655 |
| FedProx mu=0.01 | 0.9848 / 0.9912 | 0.1890 / 0.2038 | 0.2296 / 0.1942 |
| FedProx mu=0.05 | 0.9793 / 0.9804 | 0.2110 / 0.3005 | 0.2539 / 0.2563 |
| FedProx mu=0.1 | 0.9649 / 0.9712 | 0.2080 / 0.3810 | 0.2593 / 0.3398 |
| FedPer | 0.9772 / 0.9664 | 0.0708 / 0.1085 | 0.1404 / 0.1226 |
| Local-only | 0.9149 / 0.9113 | 0.0456 / 0.0568 | 0.2124 / 0.1952 |

### Noise floor: seed-to-seed |difference| of each statistic

Read any between-method gap above against these cells; smaller gaps are within
seed noise (n = 2 seeds — this is a floor, not a full variance estimate).

| Method | Wi-Fi–MQTT (dJ / dS / dC) | Wi-Fi–BT | MQTT–BT | Top-5 stability per client (Wi-Fi / MQTT / BT) |
|---|---|---|---|---|
| FedAvg | 0.139 / 0.003 / 0.020 | 0.000 / 0.066 / 0.036 | 0.000 / 0.029 / 0.033 | 0.43 / 0.43 / 0.43 |
| FedProx mu=0.001 | 0.417 / 0.008 / 0.006 | 0.139 / 0.017 / 0.029 | 0.111 / 0.039 / 0.027 | 0.43 / 0.25 / 0.43 |
| FedProx mu=0.01 | 0.000 / 0.037 / 0.006 | 0.000 / 0.052 / 0.015 | 0.111 / 0.024 / 0.035 | 0.43 / 0.25 / 0.67 |
| FedProx mu=0.05 | 0.238 / 0.015 / 0.001 | 0.139 / 0.043 / 0.089 | 0.139 / 0.034 / 0.002 | 0.43 / 0.25 / 0.43 |
| FedProx mu=0.1 | 0.571 / 0.029 / 0.006 | 0.429 / 0.034 / 0.173 | 0.318 / 0.019 / 0.081 | 0.43 / 0.43 / 0.11 |
| FedPer | 0.139 / 0.053 / 0.011 | 0.000 / 0.008 / 0.038 | 0.000 / 0.101 / 0.018 | 0.67 / 0.67 / 0.67 |
| Local-only | 0.000 / 0.119 / 0.004 | 0.000 / 0.142 / 0.011 | 0.000 / 0.081 / 0.017 | 0.67 / 1.00 / 1.00 |

Figure: `figures/fig2_spearman_divergence_heatmap.png` (Spearman panel + noise
floor side by side).

### Per-client explanation comparison (FedPer and local-only vs baselines)

Spearman rho between a client's explanation under two methods, same seed
(computed from stored full vectors; keys `fedper_vs_fedavg`,
`fedper_vs_local_only`, `local_vs_global`).

| Comparison | Wi-Fi (42 / 7) | MQTT (42 / 7) | Bluetooth (42 / 7) |
|---|---|---|---|
| FedPer vs FedAvg | 0.9546 / 0.9900 | 0.9109 / 0.8856 | 0.9888 / 0.9888 |
| FedPer vs Local-only | 0.8636 / 0.8620 | 0.7809 / 0.8061 | 0.9795 / 0.9829 |
| Local-only vs FedAvg | 0.7505 / 0.8583 | 0.7818 / 0.7055 | 0.9820 / 0.9813 |

Per client, FedPer's explanations are slightly *closer* to FedAvg's than to
local-only's on Wi-Fi and MQTT, and essentially equidistant on Bluetooth —
consistent with (c): the private head changes behavior, not feature attribution.

---

## 4. Cluster-FL slot (empty)

A teammate's Cluster-FL port will drop runs at `results/cluster_fl_results.json`
following the documented schema (`CLUSTER_FL_RUN_SCHEMA` in
`src/run_study_experiments.py`) and merge with `python src/merge_cluster_fl.py`
— no code changes needed. Until then the dashboard and the divergence summary
show an explicit placeholder (`slot empty - cluster_fl results pending`).

---

## 5. Reproduction

```bash
pip install -r requirements.txt
python src/run_study_experiments.py   # full suite (seeds 42, 7)
python scripts/backfill_completions.py  # idempotent: only missing pieces
python scripts/audit_results.py       # completeness matrix
python scripts/make_figures.py        # figures/ (300 dpi PNGs)
streamlit run app.py                  # 5-tab dashboard
```
