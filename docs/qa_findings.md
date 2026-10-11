# FedXplain-IoMT: Quality Assurance & Correctness Audit Report

**Date:** 2026-10-11  
**Auditor:** Senior Quality Assurance Engineer  
**Scope:** Correctness audit across all mathematical metrics, data pipelines, federated algorithms, results integrity, Streamlit dashboard, publication figures, and documentation.  
**Seeds:** Exactly 2 seeds (42, 7).  
**Cluster-FL Scope:** Out of scope (external teammate ownership; audited only that the placeholder slot exists, is documented, and does not cause dashboard/suite failures).

---

## 1. Executive Summary & Audit Matrix

Every subsystem was audited against the specifications in Part A (A1–A6). All critical mathematical computations, data integrity boundaries, and federated algorithms have been proven correct.

| Audit Section | Check | Status | Severity | Notes |
|---|---|:---:|:---:|---|
| **A1. Metric Correctness** | Confusion Matrix Recomputations (Macro F1, Precision, Recall) | **PASS** | None | 42/42 client metric sets match raw $tn, fp, fn, tp$ bit-for-bit. |
| | Top-5 Jaccard Similarity ($|A \cap B| / |A \cup B|$) | **PASS** | None | 42/42 client pair sets match mathematical definition. |
| | Spearman $\rho$ and Cosine on Full 45-dim Vectors | **PASS** | None | Computed on complete 45-dim attributions; names & ordering 100% aligned. |
| | Noise Floor Between-Seed Comparison Check | **PASS** | None | Seed-to-seed top-5 agreement is strictly between seed 42 and seed 7. |
| | Noise Floor Label Assignment Order in JSON | **FINDING 1** | Low | Subkeys `"seed42_top5"` and `"seed7_top5"` swapped due to ascending seed sort. |
| **A2. Data Pipeline** | StandardScaler Test-Data Leakage | **PASS** | None | `fit_transform` strictly on train, `transform` on test; no leakage. |
| | Stratification Verification | **PASS** | None | Verified `stratify=y`; train/test class ratios identical to within rounding. |
| | 45-Feature Guardrail Assertion | **PASS** | None | Injected column-drop test; `AssertionError` fired immediately with error message. |
| | Documented Train Sample Counts | **PASS** | None | Wi-Fi: 79,994; MQTT: 39,997; Bluetooth: 13,076 (100% match). |
| | Cache Key Freshness / Mutation Safety | **PASS** | None | Key contains `(protocol, data_dir, test_size, seed)`; returns cloned tensors. |
| **A3. Federated Logic** | Aggregation Weights Sum to 1 | **PASS** | None | $\sum_k \frac{n_k}{\sum_j n_j} = 1.0$ across FedAvg, FedProx, and FedPer. |
| | FedPer Trunk / Private Head Parameter Isolation | **PASS** | None | Only 4 trunk arrays uploaded and averaged; private head never uploaded. |
| | FedProx Detached Global Weights | **PASS** | None | Proximal loss uses cloned & detached tensors from global model. |
| | Local-Only Cross-Client Weight Isolation | **PASS** | None | 30 isolated epochs per client; zero weight flow across clients. |
| | Determinism Check (FedAvg Seed 42, 10 rounds) | **PASS** | None | Re-run produced $\Delta = 0.000000\text{e}+00$ across all metrics and confusion matrices. |
| **A4. Results File Integrity** | JSON Schema Validator (`validate_results.py`) | **PASS** | None | Passes with exit code 0; validates all keys, types, value bounds, and vector shapes. |
| | NaN / Infinity Verification | **PASS** | None | 0 NaN, Infinity, or -Infinity values in `study_results.json`. |
| **A5. Dashboard** | Streamlit AppTest Execution | **PASS** | None | 0 exceptions across all 5 tabs; all 5 dataframes render real non-dummy data. |
| | 5 Hand Cross-Checks against `study_results.json` | **PASS** | None | All 5 cross-checks match bit-for-bit. |
| | Cluster-FL Empty Slot Graceful Placeholder | **PASS** | None | Selecting Cluster-FL renders informative placeholder box; no crashes. |
| | Dashboard Centralized Display Inconsistency | **FINDING 2** | Low | Tab 1 BT table and Tab 3 text used `'bluetooth_benign_recall'` instead of fallback. |
| **A6. Figures & Report** | Figure Regeneration via `scripts/make_figures.py` | **PASS** | None | All 4 publication PNGs regenerated; byte-identical to `figures/`. |
| | Report Spot-Checks (docs/final_results.md) | **PASS** | None | 10/10 spot-checks match `study_results.json` numbers exactly. |

---

## 2. Detailed Findings & Issue Log

### Finding 1: Preliminary Noise Floor Seed Key Label Inversion in `study_results.json`
- **Issue:** In `results/study_results.json` under `preliminary_noise_floor`, the nested dictionary keys `"seed42_top5"` and `"seed7_top5"` (as well as `"seed42_jaccard"` and `"seed7_jaccard"` in `pairwise_divergence_spread`) have inverted values: `"seed42_top5"` stores the top-5 feature list from the seed 7 run, and `"seed7_top5"` stores the list from seed 42.
- **Severity:** Low (Cosmetic metadata naming; no metrics or conclusions affected).
- **Root Cause:** In `src/run_study_experiments.py` line 174 (`compute_noise_floor()`):
  ```python
  runs = sorted(runs, key=lambda r: r["seed"])
  if len(runs) < 2:
      continue
  r42, r7 = runs[0], runs[1]
  ```
  Because `SEEDS = [42, 7]`, numerical ascending sorting places `7` before `42`. Thus `runs[0]` is seed 7 and `runs[1]` is seed 42. Unpacking `r42, r7 = runs[0], runs[1]` mistakenly assigns the seed 7 run to variable `r42` and seed 42 to variable `r7`.
  Because the Jaccard overlap $\frac{|A \cap B|}{|A \cup B|}$ and spread $|j_1 - j_2|$ are symmetric set/arithmetic operations, the computed noise floor values (`seed_to_seed_jaccard`, `abs_diff`, `full_metric_spread`) are 100% mathematically correct between-seed comparisons. Only the labels on the intermediate lists were swapped.
- **Proposed Fix:** Use explicit seed indexing:
  ```python
  by_seed = {r["seed"]: r for r in runs}
  if 42 not in by_seed or 7 not in by_seed:
      continue
  r42, r7 = by_seed[42], by_seed[7]
  ```
  Update `src/run_study_experiments.py`, and run `scripts/backfill_completions.py` to refresh `results/study_results.json`.
- **Verification:** Run `scratch/audit_a1.py`: all 20 noise floor label mismatches drop to 0.

---

### Finding 2: Streamlit Dashboard Centralized Reference Benign Recall Field Fallback in `app.py`
- **Issue:** In `app.py`, the Bluetooth benign-recall table (Tab 1, line 532) and the written findings text (Tab 3, line 997) attempt to read `cents_run.get("bluetooth_benign_recall")`. Because centralized training is a pooled model, its evaluation metrics are stored under `"benign_recall"` (not per-protocol). Consequently, the dashboard renders `"—"` in Tab 1 and `0.0000 / 0.0000` in Tab 3 for Centralized benign recall, instead of the true reference values `0.9681 / 0.9692`. In addition, line 998 references `fedper_bt_mean` for FedPer's avg macro F1, displaying 0.8490 (87%) instead of 0.9241 (95%).
- **Severity:** Low (Display inconsistency in dashboard UI only; results file and published report `docs/final_results.md` already have the correct numbers).
- **Root Cause:** In `app.py`:
  - Line 532: `val = cents_run.get("bluetooth_benign_recall")`
  - Line 997: `cents.get('42', {}).get('bluetooth_benign_recall', 0.0)`
  - Line 998: `Its avg macro F1 ({fmt(fedper_bt_mean)}) is {fedper_bt_mean / local_mean * 100:.0f}%`
- **Proposed Fix:**
  - In lines 532 and 997, fallback to `"benign_recall"`: `cents_run.get("bluetooth_benign_recall", cents_run.get("benign_recall"))`.
  - In line 998, compute and display `fedper_f1_mean` (the true average macro F1 across seeds, 0.9241, representing 95% of local-only).
- **Verification:** Run `AppTest` on `app.py`: verify that Centralized renders `0.9681` and `0.9692`, and FedPer macro F1 text renders `0.9241` (95%).

---

## 3. Evidence Log

### A1. Metric Recomputation Audit
- Verified every run $\times$ client metric entry in `results/study_results.json` against mathematical formulas:
  - $\text{benign\_precision} = \frac{tn}{tn + fn}$
  - $\text{benign\_recall} = \frac{tn}{tn + fp}$
  - $\text{attack\_precision} = \frac{tp}{tp + fp}$
  - $\text{attack\_recall} = \frac{tp}{tp + fn}$
  - $\text{f1\_macro} = \frac{F1_{\text{benign}} + F1_{\text{attack}}}{2}$
  - $\text{accuracy} = \frac{tp + tn}{tp + tn + fp + fn}$
  - Top-5 Jaccard: $J(A, B) = \frac{|A \cap B|}{|A \cup B|}$
  - Spearman $\rho$: $\text{scipy.stats.spearmanr}(a, b)$ on full 45-dimensional mean-$|\text{SHAP}|$ vectors.
  - Cosine: $\frac{a \cdot b}{\|a\|_2 \|b\|_2}$ on full 45-dimensional vectors.
- Result: **0 mismatches** across all 42 client metric evaluations and 42 pairwise divergence evaluations.

### A2. Data Pipeline & Preprocessing Audit
- Test data leakage into `StandardScaler`: Inspected `src/utils.py` lines 137–140:
  ```python
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train).astype(np.float32)
  X_test = scaler.transform(X_test).astype(np.float32)
  ```
  `fit_transform` is strictly executed on `X_train`. Zero leakage.
- Stratification:
  - Wi-Fi: Train benign ratio = 0.036140, Test benign ratio = 0.036152 ($\Delta = 1.16\times 10^{-5}$)
  - MQTT: Train benign ratio = 0.367803, Test benign ratio = 0.367800 ($\Delta = 2.59\times 10^{-6}$)
  - Bluetooth: Train benign ratio = 0.587947, Test benign ratio = 0.588073 ($\Delta = 1.26\times 10^{-4}$)
- Guardrail assertion:
  - Injected temporary CSV with 44 features into `wifi_study.csv`.
  - Result: `AssertionError: Feature count mismatch for protocol 'wifi': expected 45, got 44...` fired as expected.
- Train sample counts match documented splits:
  - Wi-Fi: 79,994 (Train) / 19,999 (Test) $\rightarrow$ matches 79,994
  - MQTT: 39,997 (Train) / 10,000 (Test) $\rightarrow$ matches 39,997
  - Bluetooth: 13,076 (Train) / 3,270 (Test) $\rightarrow$ matches 13,076
- Cache keys: `(protocol.lower().strip(), os.path.abspath(data_dir), test_size, random_state)`. All 4 dimensions present.

### A3. Determinism Re-Run Evidence
Ran `train_fedavg(seed=42, n_rounds=10, n_epochs=3)` under pinned requirements:
```
Protocol: WIFI
  Macro F1:      stored=0.937525, rerun=0.937525, diff=0.000000e+00
  Binary F1:     stored=0.995433, rerun=0.995433, diff=0.000000e+00
  Benign Recall: stored=0.889350, rerun=0.889350, diff=0.000000e+00
  Attack Recall: stored=0.995020, rerun=0.995020, diff=0.000000e+00
  Confusion Matrix: tn=643, fp=80, fn=96, tp=19180 (exact match)

Protocol: MQTT
  Macro F1:      stored=0.840945, rerun=0.840945, diff=0.000000e+00
  Binary F1:     stored=0.902929, rerun=0.902929, diff=0.000000e+00
  Benign Recall: stored=0.646275, rerun=0.646275, diff=0.000000e+00
  Attack Recall: stored=0.992407, rerun=0.992407, diff=0.000000e+00
  Confusion Matrix: tn=2377, fp=1301, fn=48, tp=6274 (exact match)

Protocol: BLUETOOTH
  Macro F1:      stored=0.611241, rerun=0.611241, diff=0.000000e+00
  Binary F1:     stored=0.686972, rerun=0.686972, diff=0.000000e+00
  Benign Recall: stored=0.366615, rerun=0.366615, diff=0.000000e+00
  Attack Recall: stored=0.996288, rerun=0.996288, diff=0.000000e+00
  Confusion Matrix: tn=705, fp=1218, fn=5, tp=1342 (exact match)

STATUS: PASS (Deterministic bit-for-bit reproduction)
```

### A4. Schema & Integrity Validation Evidence
Ran `python scripts/validate_results.py`:
- 0 NaN, Infinity, or -Infinity values.
- All 12 federated runs, 2 centralized runs, and 2 local-only runs validated.
- All metric values $\in [0, 1]$, all confusion matrix values $\ge 0$, all feature vectors length $== 45$.
- Result: `[PASS] All schema and integrity checks passed successfully (0 errors).` (Exit code: 0).

### A5. Dashboard AppTest & Cross-Check Evidence
- `streamlit.testing.v1.AppTest` executed:
  - Tabs rendered: 5/5
  - Dataframes rendered: 5/5 (real non-dummy data)
  - Exceptions: 0
  - Cluster-FL slot selection: Renders pending placeholder (`st.info`) with teammate instructions, 0 exceptions.
- 5 Mandatory Hand Cross-Checks against `results/study_results.json`:
  1. Wi-Fi FedAvg seed 42 macro F1: Dashboard = **0.9375**, JSON = **0.9375** (Exact match)
  2. Bluetooth benign recall FedProx $\mu=0.01$ seed 42: Dashboard = **0.0364**, JSON = **0.0364** (Exact match)
  3. Jaccard Wi-Fi–MQTT FedAvg seed 42: Dashboard = **0.1111**, JSON = **0.1111** (Exact match)
  4. Spearman Wi-Fi–MQTT FedAvg seed 42: Dashboard = **0.7493**, JSON = **0.7493** (Exact match)
  5. Local-only Bluetooth seed 42 macro F1: Dashboard = **0.9883**, JSON = **0.9883** (Exact match)

### A6. Figure & Report Verification Evidence
- Ran `python scripts/make_figures.py`:
  - `figures/fig1_detection_macro_f1.png`: Byte-identical to committed file.
  - `figures/fig2_spearman_divergence_heatmap.png`: Byte-identical to committed file.
  - `figures/fig3_bluetooth_mu_sweep.png`: Byte-identical to committed file.
  - `figures/fig4_tradeoff_scatter.png`: Byte-identical to committed file.
- Spot-checked 10 values in `docs/final_results.md` against `results/study_results.json`: 10/10 exact matches (100%).

---

## 4. Status Summary
**System QA status: PASS WITH FIXES**  
- Critical (P0): 0  
- High (P1): 0  
- Medium (P2): 0  
- Low (P3): 2 (Finding 1: noise floor label swap in JSON; Finding 2: dashboard UI display fallback)
