# Preliminary FedAvg vs. FedProx SHAP Divergence Comparison

## 1. Divergence Metrics Comparison

The table below compares top-5 SHAP feature explanation overlap (measured via Jaccard index) between the FedAvg global model and the FedProx ($\mu = 0.1$) global model across identical client data splits.

| Client Pair | FedAvg Jaccard | FedProx Jaccard | Difference ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **wifi-mqtt** | 0.6667 | 0.4286 | -0.2381 |
| **wifi-bluetooth** | 0.1111 | 0.1111 | 0.0000 |
| **mqtt-bluetooth** | 0.1111 | 0.0000 | -0.1111 |

*Note: Difference calculated as FedProx Jaccard minus FedAvg Jaccard. Lower Jaccard indicates higher explanation divergence across clients.*

---

## 2. Shared Top-5 Features Breakdown

- **wifi-mqtt**:
  - *FedAvg*: `Magnitue`, `ack_flag_number`, `psh_flag_number`, `rst_count` (4 shared)
  - *FedProx*: `Variance`, `psh_flag_number`, `rst_count` (3 shared)
- **wifi-bluetooth**:
  - *FedAvg*: `Magnitue` (1 shared)
  - *FedProx*: `Magnitue` (1 shared)
- **mqtt-bluetooth**:
  - *FedAvg*: `Magnitue` (1 shared)
  - *FedProx*: `None` (0 shared)

---

## 3. Interpretation

Based on this single run, FedProx ($\mu = 0.1$) yielded lower Jaccard overlap scores on two of the three client pairs compared to FedAvg (reflecting slightly higher explanation divergence), while maintaining identical low overlap on Wi-Fi vs. Bluetooth. However, this is strictly ONE run with ONE seed ($seed = 42$) and ONE un-tuned value of $\mu$, serving only as a preliminary signal rather than a validated finding; the full study will systematically test whether proximal regularization systematically impacts explanatory divergence using multi-seed repetitions, $\mu$ parameter sweeps, and statistical noise-floor calibration.
