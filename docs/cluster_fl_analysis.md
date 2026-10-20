# Cluster-FL Scientific Analysis

This document provides the rigorous scientific analysis of the Cluster-FL experimental results integrated into `results/study_results.json`. All metrics and divergence numbers derive strictly from the merged study payload across seeds 42 and 7 without retraining.

---

## 1. Bluetooth Benign Recall Rescue

The primary vulnerability identified in the FedXplain-IoMT benchmark is the catastrophic collapse of Bluetooth benign recall under global federated aggregation (FedAvg achieves only 0.3666 on seed 42 and 0.4238 on seed 7, flagging over 57–63% of benign Bluetooth packets as malicious).

Cluster-FL partitions clients by cosine similarity of model updates after warm-up:
- **Cluster 0**: Wi-Fi (large-volume, high-throughput protocol)
- **Cluster 1**: MQTT + Bluetooth (lightweight IoT sensor & diagnostic protocols)

### Rescue Comparison Table

| Method | Seed 42 | Seed 7 | Two-Seed Mean | Abs. Gain vs FedAvg | Gap Recovery vs Local-only |
|---|:---:|:---:|:---:|:---:|:---:|
| **FedAvg (baseline)** | 0.3666 | 0.4238 | 0.3952 | — | 0.00% |
| **FedProx ($\mu=0.1$)** | 0.0073 | 0.0343 | 0.0208 | −0.3744 | −59.4% (worsens collapse) |
| **FedPer (head personalization)** | 0.8258 | 0.8721 | 0.8489 | +0.4537 | 72.67% / 77.94% |
| **Cluster-FL (clustered FL)** | **0.9906** | **0.9948** | **0.9927** | **+0.5975** | **98.77% / 99.28%** |
| **Local-only (ceiling)** | 0.9984 | 0.9990 | 0.9987 | +0.6035 | 100.00% |

### Key Takeaway
Cluster-FL delivers an almost complete rescue of Bluetooth benign recall, recovering **98.8% (seed 42)** and **99.3% (seed 7)** of the gap between FedAvg and the unconstrained local-only ceiling. This significantly outperforms FedPer (+0.1438 higher benign recall), demonstrating that separating structurally discordant client data distributions at the cluster level is more effective for local calibration than sharing a trunk with isolated output heads.

---

## 2. Overall Detection Quality (Macro F1)

Under Cluster-FL, the individual clients achieve high detection quality without being dominated by Wi-Fi parameter updates.

### Summary Across Methods

| Method | Seed 42 | Seed 7 | Two-Seed Mean | Seed Spread (|42 − 7|) |
|---|:---:|:---:|:---:|:---:|
| **FedAvg** | 0.7966 | 0.8011 | 0.7988 | 0.0045 |
| **FedProx ($\mu=0.001$)** | 0.7898 | 0.7462 | 0.7680 | 0.0436 |
| **FedPer** | 0.9179 | 0.9302 | 0.9241 | 0.0123 |
| **Cluster-FL** | **0.9666** | **0.9664** | **0.9665** | **0.0002** |
| **Local-only (ceiling)** | 0.9715 | 0.9702 | 0.9708 | 0.0012 |
| **Centralized (reference)** | 0.9794 | 0.9790 | 0.9792 | 0.0004 |

### Protocol-Specific Macro F1 Breakdown

| Protocol | FedAvg | FedPer | Cluster-FL | Local-only |
|---|:---:|:---:|:---:|:---:|
| **Wi-Fi** | 0.9354 | 0.9387 | 0.9389 | 0.9406 |
| **MQTT** | 0.8314 | 0.9345 | 0.9777 | 0.9828 |
| **Bluetooth** | 0.6298 | 0.8990 | 0.9827 | 0.9891 |

Cluster-FL achieves an overall mean macro F1 of **0.9665**, placing it within **0.0043** of the local-only ceiling (**0.9708**) and significantly above FedPer (**0.9241**, +0.0424) and FedAvg (**0.7988**, +0.1677). Notably, Cluster-FL exhibits remarkable stability between seeds (spread of only 0.0002).

---

## 3. Explanation Divergence vs Noise Floor

To avoid confusing random seed variance with genuine behavioral shifts, every cross-method difference is evaluated against the seed-to-seed noise floor spread $| \text{seed42} - \text{seed7} |$ stored in `preliminary_noise_floor`.

### Evaluation Table: Cluster-FL vs Baselines

| Pair & Metric | Cluster-FL Mean | FedAvg Mean | FedPer Mean | Ref. Noise Floor | vs FedAvg Status | vs FedPer Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Wi-Fi–MQTT** | | | | | | |
| Top-5 Jaccard | 0.1111 | 0.1805 | 0.1805 | 0.1389 | WITHIN-NOISE ($\Delta = 0.0694$) | WITHIN-NOISE ($\Delta = 0.0694$) |
| Spearman $\rho$ | 0.6511 | 0.7507 | 0.7631 | 0.0350 / 0.0526 | **FINDING** ($\Delta = 0.0996$) | **FINDING** ($\Delta = 0.1120$) |
| Cosine Sim. | 0.9410 | 0.9804 | 0.9718 | 0.0206 / 0.0206 | **FINDING** ($\Delta = 0.0393$) | **FINDING** ($\Delta = 0.0308$) |
| **Wi-Fi–Bluetooth** | | | | | | |
| Top-5 Jaccard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | WITHIN-NOISE ($\Delta = 0.0000$) | WITHIN-NOISE ($\Delta = 0.0000$) |
| Spearman $\rho$ | 0.2029 | 0.2845 | 0.2702 | 0.0662 / 0.0462 | **FINDING** ($\Delta = 0.0816$) | **FINDING** ($\Delta = 0.0673$) |
| Cosine Sim. | 0.0657 | 0.0811 | 0.0897 | 0.0357 / 0.0377 | WITHIN-NOISE ($\Delta = 0.0154$) | WITHIN-NOISE ($\Delta = 0.0239$) |
| **MQTT–Bluetooth** | | | | | | |
| Top-5 Jaccard | 0.1111 | 0.0000 | 0.0000 | 0.0000 | **FINDING** ($\Delta = 0.1111$) | **FINDING** ($\Delta = 0.1111$) |
| Spearman $\rho$ | 0.2702 | 0.2494 | 0.1964 | 0.0671 / 0.1006 | WITHIN-NOISE ($\Delta = 0.0209$) | WITHIN-NOISE ($\Delta = 0.0738$) |
| Cosine Sim. | 0.2047 | 0.1013 | 0.1315 | 0.0330 / 0.0178 | **FINDING** ($\Delta = 0.1034$) | **FINDING** ($\Delta = 0.0732$) |

### Divergence Findings Analysis
1. **Cross-Cluster Divergence (Wi-Fi vs MQTT/BT)**: Wi-Fi is isolated in Cluster 0 while MQTT and Bluetooth share Cluster 1. As a direct consequence, Spearman $\rho$ between Wi-Fi and MQTT drops from 0.7507 (FedAvg) to 0.6511 (Cluster-FL), which is a statistically meaningful **FINDING** exceeding the noise floor (0.0350). Similarly, Wi-Fi vs Bluetooth Spearman drops from 0.2845 to 0.2029 (**FINDING**).
2. **Intra-Cluster Alignment (MQTT vs Bluetooth)**: Because MQTT and Bluetooth are aggregated together in Cluster 1, their explanation similarity increases: Top-5 Jaccard increases from 0.0000 to 0.1111 (**FINDING**, shared feature: `Number`), and Cosine similarity doubles from 0.1013 (FedAvg) to 0.2047 (Cluster-FL), exceeding the noise floor (0.0330) as a robust **FINDING**.

---

## 4. Scientific Verdict

Clustering clients by update similarity successfully rescues the Bluetooth minority protocol without needing private client heads, achieving **0.9906 / 0.9948** benign recall and recovering **98.8–99.3%** of the gap to local-only. By segregating high-volume Wi-Fi from MQTT and Bluetooth, Cluster-FL eliminates the gradient interference that drove the Bluetooth collapse, reaching **0.9665** overall macro F1 (within 0.43 points of the local-only ceiling). However, this rescue operates by partitioning the explanation space rather than unifying it: explanation agreement between clusters drops significantly below FedAvg, while intra-cluster explanation alignment between MQTT and Bluetooth increases.
