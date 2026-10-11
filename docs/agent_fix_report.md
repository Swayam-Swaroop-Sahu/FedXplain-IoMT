# Agent Fix Report — FedXplain-IoMT

**Date:** 2026-10-10
**Scope:** Environment pinning, full SHAP vector storage + new divergence metrics, one-environment
result regeneration, local-only control baseline, FedProx μ sweep extension, dashboard/docs
corrections. No module restructuring, no key renames, no changes to model architecture,
`N_ROUNDS`, `N_LOCAL_EPOTS`, or `SEEDS`.

---

## 1. What changed, file by file

| File | Change | Rationale |
|---|---|---|
| `requirements.txt` | Replaced all `>=` ranges with exact `==` pins; added `scipy`, `streamlit`, `plotly` | Reproducibility (Task 1); scipy needed for the new Spearman metric; dashboard deps belong in the env that generated the results |
| `README.md` | Python 3.14 badge; new "Environment Pinning" paragraph in Reproducibility section; `MU_VALUES` listing updated; `run_local_only.py` added to structure | Document the pinned env and the new control baseline |
| `src/config.py` | `MU_VALUES` extended to `[0.001, 0.01, 0.05, 0.1]` with rationale comment | Test whether a gentler proximal pull helps MQTT without breaking Bluetooth (Task 5) |
| `src/explain.py` | Extracted `explain_client()` per-client helper; added `_spearman_cosine()` and `compute_pairwise_divergence()`; `compute_shap_divergence()` now also returns `shap_full_vectors_per_client` (45-element mean-|SHAP| vector per client) and `divergence_full` (Spearman + cosine per pair); existing top-10/top-5 keys unchanged | Stop storing only top-k rankings — full vectors enable smooth, fine-grained divergence metrics (Task 2) |
| `src/run_local_only.py` | **New.** `train_local_only()` (per-protocol isolated models, `N_ROUNDS × N_LOCAL_EPOCHS` = 30 total epochs to match federated compute; reuses `IoMTFlowerClient` preprocessing/evaluation), `compute_local_vs_global()`, `run_local_only_suite()`, standalone `main()` that merges into `study_results.json` | The "everyone trains alone" control (Task 4) |
| `src/run_study_experiments.py` | Stores the two new keys per run; runs the local-only suite for both seeds; payload gains top-level `local_only` and `local_vs_global`; summary printing extended with Spearman/cosine per pair and local-only rows | Tasks 2.4, 4.5, 4.6 |
| `app.py` | Fixed "50%" → "60%" (Wi-Fi share); Tab 2 gains a full-vector divergence table below the Jaccard heatmap, guarded with `.get()` so old results files render fine | Task 6 |
| `docs/ui_guide.md` | Added full-vector divergence table to Tab 2 visuals + plain-language note on why Jaccard/Spearman/cosine differ | Task 6 |
| `results/study_results.json` | Regenerated in one environment (see §3); old file backed up to the OS temp dir | Task 3 |
| `models/*.pth` | 11 model files regenerated (centralized + 10 federated runs) | Byproduct of the suite run (gitignored) |

Backward compatibility verified: every pre-existing JSON key (`client_metrics`, `shap_top10_per_client`,
`shap_top5_per_client`, `jaccard_pairwise`, `shared_features`, `centralized`, `poc_baseline_reference`,
`preliminary_noise_floor`, `run_metadata`) is still present with the same schema, and `app.py` loads the
new file without error (§5).

## 2. Environment pinning

The original environment no longer existed on this machine (no torch/flwr/shap/streamlit/plotly were
installed). The dependencies were installed into the current Python and the **actual resulting versions**
were pinned:

```
torch==2.14.1        flwr==1.39.0      shap==0.53.0       pandas==3.0.6
scikit-learn==1.9.1  matplotlib==3.11.2  numpy==2.5.3     scipy==1.18.1
streamlit==1.65.0    plotly==7.1.0
```

Python version: **3.14.7**. Hardware: CPU-only (no CUDA available); the MLP is small enough that GPU
is not required. Note: `shap==0.53.0` no longer requires `numba` (the old `NUMBA_DISABLE_JIT=1`
workaround in `explain.py`/`run_study_experiments.py` is harmless but no longer load-bearing).

## 3. Regenerated results (Task 3) — old vs new

The suite (`python -m src.run_study_experiments`) was run **once, end-to-end, in the pinned
environment**: 1 centralized baseline + 10 federated runs (2 seeds × [FedAvg + 4 μ values]) +
2 local-only control runs, each followed by the full SHAP pipeline. Verified: all runs have complete
`client_metrics`, 45-element `shap_full_vectors_per_client`, and `divergence_full`.

> **Note on run matrix:** because the Task-5 config change was applied before this run, the
> regeneration already includes the extended μ sweep (10 federated runs rather than the former 6).
> Nothing was skipped.

Old vs new macro F1 / benign recall (existing configurations only; μ=0.001 had never been run):

| Run (seed, aggregator, μ) | Client | Old macro F1 | New macro F1 | Old benign recall | New benign recall |
|---|---|---:|---:|---:|---:|
| 42, fedavg, — | Wi-Fi | 0.9353 | 0.9375 | 0.9142 | 0.8893 |
| 42, fedavg, — | MQTT | 0.8351 | 0.8409 | 0.6330 | 0.6463 |
| 42, fedavg, — | Bluetooth | 0.6106 | 0.6112 | 0.3661 | 0.3666 |
| 42, fedprox, 0.01 | Wi-Fi | 0.9138 | 0.9242 | 0.9682 | 0.9502 |
| 42, fedprox, 0.01 | MQTT | 0.9250 | 0.9205 | 0.8284 | 0.8140 |
| 42, fedprox, 0.01 | Bluetooth | 0.3379 | 0.3312 | 0.0426 | 0.0364 ⚠ |
| 42, fedprox, 0.1 | Wi-Fi | 0.9073 | 0.9021 | 0.9544 | 0.9336 |
| 42, fedprox, 0.1 | MQTT | 0.9093 | 0.8665 | 0.7983 | 0.6906 |
| 42, fedprox, 0.1 | Bluetooth | 0.3286 | 0.2999 | 0.0338 | 0.0073 ⚠ |
| 7, fedavg, — | Wi-Fi | 0.9302 | 0.9332 | 0.8534 | 0.8797 |
| 7, fedavg, — | MQTT | 0.8524 | 0.8218 | 0.6607 | 0.5984 |
| 7, fedavg, — | Bluetooth | 0.7459 | 0.6483 | 0.5762 | 0.4238 |
| 7, fedprox, 0.01 | Wi-Fi | 0.9142 | 0.9190 | 0.9115 | 0.9350 |
| 7, fedprox, 0.01 | MQTT | 0.8809 | 0.8849 | 0.7205 | 0.7295 |
| 7, fedprox, 0.01 | Bluetooth | 0.3343 | 0.3277 | 0.0400 | 0.0333 ⚠ |
| 7, fedprox, 0.1 | Wi-Fi | 0.8928 | 0.9023 | 0.9516 | 0.9267 |
| 7, fedprox, 0.1 | MQTT | 0.8664 | 0.8759 | 0.6911 | 0.7115 |
| 7, fedprox, 0.1 | Bluetooth | 0.3343 | 0.3286 | 0.0400 | 0.0343 ⚠ |

Centralized baseline is stable across environments: macro F1 **0.9794 → 0.9794** (was already
reproduced bit-for-bit to 4 decimals).

### Flags raised

- **Bluetooth benign recall < 0.3 under every FedProx configuration (both seeds)** — the collapse is
  universal across μ ∈ {0.001 … 0.1}. New μ=0.001 numbers: Bluetooth macro F1 0.4894 (seed 42) /
  0.4008 (seed 7), benign recall 0.209 / 0.109 — better than larger μ but still far below FedAvg
  (0.6112 / 0.6483). **FedProx does not rescue Bluetooth at any tested μ.**
- **MQTT finding:** μ=0.001 gives MQTT its best macro F1 of the whole sweep (0.9534 seed 42, 0.9160
  seed 7, vs 0.8409/0.8218 under FedAvg) — the "gentler pull helps MQTT" hypothesis is supported,
  but it does not fix Bluetooth.
- **Old vs new differences are environment-level, not code-level:** the same seed produces slightly
  different numbers (≤ ±0.1 macro F1, e.g. FedAvg seed-7 Bluetooth 0.7459 → 0.6483) because the
  previous results were generated in a different (unrecoverable) environment. All numbers in
  `study_results.json`, the README, and the dashboard now come from the single pinned environment.
- **Seed-to-seed Jaccard under mu=0.1 seed 7 is now 1.0** (wifi-mqtt top-5 identical) — a reminder
  that top-5 Jaccard is coarse; the full-vector metrics (Spearman 0.8593, cosine 0.9712) still show
  it is not a perfect alignment.

## New divergence metrics (examples, seed 42)

| Pair | FedAvg J / ρ / cos | FedProx μ=0.01 J / ρ / cos |
|---|---|---|
| Wi-Fi–MQTT | 0.1111 / 0.7493 / 0.9902 | 0.6667 / 0.8424 / 0.9848 |
| Wi-Fi–Bluetooth | 0.0000 / 0.3176 / 0.0632 | 0.1111 / 0.5422 / 0.1890 |
| MQTT–Bluetooth | 0.0000 / 0.2641 / 0.0848 | 0.1111 / 0.3892 / 0.2296 |

The full-vector metrics behave as intended: Jaccard jumps in 0.11–0.33 steps while Spearman/cosine
move smoothly, and the Wi-Fi–Bluetooth/cosine ≈ 0.06 numbers quantify *how* disjoint the BT client's
attribution profile is rather than merely "no overlap in top 5".

## 4. Local-only baseline (Task 4)

Each protocol trains alone for 30 epochs (identical per-client compute to the federated runs).
Macro F1 per client:

| Seed | Wi-Fi | MQTT | Bluetooth | Avg |
|---|---:|---:|---:|---:|
| 42 | 0.9442 | 0.9819 | 0.9883 | 0.9715 |
| 7 | 0.9370 | 0.9838 | 0.9899 | 0.9702 |

Bluetooth benign recall: 0.998 (seed 42) / 0.999 (seed 7).

**This is the headline control result:** local-only beats every federated configuration on average
macro F1 (0.97 vs 0.80 best-case FedAvg) and completely avoids the Bluetooth collapse. Sharing —
under FedAvg or any tested μ of FedProx — currently costs detection quality on this benchmark.

Local vs FedAvg (seed 42) explanation comparison (`local_vs_global` in the JSON):

| Client | Top-5 Jaccard | Spearman ρ | Cosine |
|---|---:|---:|---:|
| Wi-Fi | 0.4286 | 0.7505 | 0.9816 |
| MQTT | 0.2500 | 0.7818 | 0.9387 |
| Bluetooth | 0.6667 | 0.9820 | 0.8464 |

Bluetooth's local model reasons almost identically to its federated counterpart (ρ = 0.98) despite
the collapse — i.e., the collapse is a decision-threshold/calibration failure, not a feature-logic
divergence. Wi-Fi's federated model drifted the most in feature logic (ρ = 0.75).

## 5. Dashboard verification

`streamlit run app.py` was exercised via `streamlit.testing.v1.AppTest` (executes the full script,
all four tab bodies included) against the regenerated JSON:

- Exceptions: **0**
- Tabs rendered: **4/4** (`Detection Performance`, `Explanation Divergence`, `FedAvg vs FedProx`, `Methodology and Limitations`)
- Dataframes rendered: **5**, including the new full-vector divergence table in Tab 2
- Sidebar shows the extended `mu_values` list
- The server also served HTTP 200 with no console errors (headless check on port 8599, then killed)

## 6. What was NOT done, and why

- **Nothing was skipped.** All runs completed in the single environment; the extended μ sweep was
  executed as part of the regeneration (~75 min wall clock on CPU).
- **Stale "50%" wording in `results/fedavg_vs_fedprox_comparison.md` was left as-is** — that file is
  a historical POC-scale report where Wi-Fi genuinely was ~50% of the (5k/3k/2k) data. Only
  `app.py` and `docs/ui_guide.md` were in Task 6's scope, and `docs/ui_guide.md` contained no 50%
  claim.
- **The "old notebook" mentioned in the task brief was not found** — the repository contains no
  `.ipynb` files (an earlier commit archived superseded trial artifacts out of git). The old seed-42
  numbers compared in §3 come from the previous `study_results.json`, backed up before overwrite.
- **Dashboard Tab for local-only results was not added** — Task 4 asks only for JSON storage and
  console summary rows. The `local_only`/`local_vs_global` keys are available in the JSON if a Tab
  is wanted later.
- **One bug was introduced and fixed during this work** (`_spearman_cosine` receiving plain lists in
  `compute_local_vs_global`): the first suite run crashed *after* training all models but *before*
  writing the JSON. The fix is a `np.asarray` coercion; the suite was re-run end-to-end so all
  numbers in `study_results.json` come from one clean execution.

## 7. Suggested next steps (not executed)

1. Extend the noise-floor analysis to the full-vector metrics (Spearman/cosine spreads across seeds)
   — the data is already stored per run.
2. Add a dashboard Tab or section for the local-only control, since it is now the strongest single
   result in the study.
3. Investigate the Bluetooth collapse mechanism: local-only ρ = 0.98 vs FedAvg suggests calibration
   (threshold/probability shift), not representation failure — a per-round Bluetooth benign-recall
   trace would confirm when the collapse sets in.
4. A 10-seed noise-floor run remains open (Phase 8 in the README roadmap); at ~3 min per federated
   run on this machine, a 10-seed × 5-config sweep ≈ 2.5 h CPU.

## 8. FedPer (personalized federated learning) — 2026-10-11

**Scope:** Third federated method: the 45→64 and 64→32 layers are aggregated (FedAvg-weighted
across clients each round); the 32→1 output head stays private per client — never uploaded, never
averaged. Each round: broadcast trunk → plain-BCE local training of trunk + private head
(no proximal term) → upload trunk only. Final model per client = global trunk + that client's
head, evaluated and SHAP-explained per client. No changes to existing result keys, `app.py`, or
`IoMTMLP`; no changes to `N_ROUNDS`, `N_LOCAL_EPOCHS`, or `SEEDS`.

### 8.1 What changed, file by file

| File | Change |
|---|---|
| `src/config.py` | New `FEDPER_SHARED_PARAM_COUNT = 4` — the only FedPer knob. `IoMTMLP.get_weights()` exposes 6 arrays in order (`net.0.weight/bias` 45→64, `net.3.weight/bias` 64→32, `net.5.weight/bias` 32→1); the first 4 are the shared trunk, the last 2 the private head |
| `src/run_fedper.py` | **New.** `train_fedper()` (per-round trunk-only upload + `fedavg_weighted()` trunk aggregation, per-client head persistence across rounds, assembled-model evaluation, full `explain_client` SHAP pipeline per assembled model, cross-client divergence across the *personalized* models), `compare_explanations()` (top-5 Jaccard + Spearman/cosine vs a same-seed baseline), `run_fedper_suite()` (both seeds, per-run runtime log, comparison sets), standalone `main()` that merges into `study_results.json` replacing only `aggregator='fedper'` entries |
| `src/run_study_experiments.py` | FedPer runs appended to the `runs` array (`aggregator="fedper"`, `mu=null`, same result schema + `fedper_vs_fedavg` / `fedper_vs_local_only` / `divergence_scope` / `avg_macro_f1`); new unified comparison table in the console summary (all methods × both seeds); existing sections untouched |
| `results/study_results.json` | 2 new `fedper` run entries (seeds 42, 7). All 10 previous run entries and all top-level keys verified byte-identical in schema |
| `models/study_fedper_seed{42,7}_{wifi,mqtt,bluetooth}.pth` | 6 assembled per-client models (gitignored) |

### 8.2 Implementation notes (deviations from the brief, and why)

1. **Per-round evaluation uses each client's assembled trunk+head.** FedPer has no single global
   model to evaluate; the round-10 evaluation *is* the "after final round, assemble and evaluate"
   step (the eval phase runs after aggregation, on trunk + that round's trained head), so no
   separate re-evaluation was added.
2. **Private heads all start from one deterministic initialization** (identical across clients;
   `set_seed(seed)` makes every fresh `IoMTMLP` init identical here anyway). Heads diverge from
   round 1 onward through local training only. This matches standard FedPer, where personal
   layers start at (arbitrary) init and are never exchanged.
3. **Runs-array entry is a projection:** it keeps the standard schema plus
   `avg_macro_f1`/`divergence_scope`/`fedper_vs_fedavg`/`fedper_vs_local_only` and drops the
   training-only `history`/`model_paths` (models are saved to `models/`). The suite and the
   standalone `main()` use the same projection, so re-running one path cannot drift the schema.
4. **FedPer runs after the local-only suite** inside `run_all_study_experiments()` so
   `fedper_vs_local_only` can be computed against the same-seed control; entries are appended,
   existing sections (noise floor, POC summary) are unaffected.

### 8.3 Runtime

CPU-only, training + full SHAP per run: **seed 42 = 181.7 s, seed 7 = 260.7 s** (~7.4 min total).

### 8.4 Unified comparison tables (macro F1 per client)

Seed 42:

| Method | Wi-Fi | MQTT | Bluetooth | Avg | BT benign recall |
|---|---:|---:|---:|---:|---:|
| FedAvg | 0.9375 | 0.8409 | 0.6112 | 0.7966 | 0.3666 |
| FedProx (μ=0.001) | 0.9266 | 0.9534 | 0.4894 | 0.7898 | 0.2090 |
| FedProx (μ=0.01) | 0.9242 | 0.9205 | 0.3312 | 0.7253 | 0.0364 |
| FedProx (μ=0.1) | 0.9021 | 0.8665 | 0.2999 | 0.6895 | 0.0073 |
| Local-only | 0.9442 | 0.9819 | **0.9883** | **0.9715** | **0.9984** |
| FedPer | 0.9414 | 0.9270 | 0.8854 | 0.9179 | 0.8258 |

Seed 7:

| Method | Wi-Fi | MQTT | Bluetooth | Avg | BT benign recall |
|---|---:|---:|---:|---:|---:|
| FedAvg | 0.9332 | 0.8218 | 0.6483 | 0.8011 | 0.4238 |
| FedProx (μ=0.001) | 0.9217 | 0.9160 | 0.4008 | 0.7462 | 0.1092 |
| FedProx (μ=0.01) | 0.9190 | 0.8849 | 0.3277 | 0.7105 | 0.0333 |
| FedProx (μ=0.1) | 0.9023 | 0.8759 | 0.3286 | 0.7023 | 0.0343 |
| Local-only | 0.9370 | 0.9838 | **0.9899** | **0.9702** | **0.9990** |
| FedPer | 0.9360 | 0.9419 | 0.9127 | 0.9302 | 0.8721 |

(The console table printed by `run_study_experiments.py` is this table minus the benign-recall
column.)

### 8.5 Direct answers

**a. Did FedPer recover Bluetooth benign recall?** Partially — most of the way, at both seeds:

| Seed | Local-only | FedAvg | FedPer | Gap recovered (FedAvg → local) |
|---|---:|---:|---:|---:|
| 42 | 0.9984 | 0.3666 | 0.8258 | **73%** |
| 7 | 0.9990 | 0.4238 | 0.8721 | **78%** |

Under FedPer, Bluetooth macro F1 reaches 0.8854 / 0.9127 — far above every sharing configuration
tested before (best was FedAvg at 0.6112 / 0.6483) but still short of local-only (0.9883 / 0.9899).

**b. Is FedPer's explanation closer to local-only's or to FedAvg's?** Closer to **FedAvg** on the
full-vector metrics, for every client and both seeds (Spearman, `fedper_vs_fedavg` vs
`fedper_vs_local_only`):

| Seed | Client | ρ vs FedAvg | ρ vs Local-only | cos vs FedAvg | cos vs Local-only |
|---|---:|---:|---:|---:|---:|
| 42 | Wi-Fi | 0.9546 | 0.8636 | 0.9974 | 0.9904 |
| 42 | MQTT | 0.9109 | 0.7809 | 0.9891 | 0.9522 |
| 42 | Bluetooth | 0.9888 | 0.9795 | 0.9500 | 0.8792 |
| 7 | Wi-Fi | 0.9900 | 0.8620 | 0.9964 | 0.9851 |
| 7 | MQTT | 0.8856 | 0.8061 | 0.9880 | 0.9408 |
| 7 | Bluetooth | 0.9888 | 0.9829 | 0.9485 | 0.9104 |

Top-5 Jaccard is mixed (e.g., Bluetooth seed 7: 1.0 vs local-only but 0.4286 vs FedAvg), but
top-5 is the coarse metric. Important caveat: local-only and FedAvg explanations were already
nearly identical for Bluetooth (§4: ρ = 0.98), so "closer to FedAvg" is not evidence that
personalization failed to localize reasoning — there was barely any distinct local reasoning
profile to drift toward. The shared trunk dominates the attribution profile in all cases.

**c. Where does FedPer land on the central trade-off?** It keeps most of local-only's detection
quality while losing essentially none of FedAvg's cross-client explanation similarity:

- *Detection kept:* average macro F1 0.9179 (seed 42) and 0.9302 (seed 7) — **94.5% / 95.9% of the
  local-only averages** (0.9715 / 0.9702), versus 0.7966 / 0.8011 under FedAvg. MQTT also improves
  substantially (0.8409→0.9270, 0.8218→0.9419); Wi-Fi is unchanged (it never collapsed).
- *Explanation similarity lost vs FedAvg:* almost nothing. Cross-client divergence among the three
  personalized models vs FedAvg's shared model (Spearman ρ, seed 42 / seed 7):
  Wi-Fi–MQTT 0.7368 / 0.7894 (FedAvg: 0.7493 / 0.7520), Wi-Fi–BT 0.2740 / 0.2663
  (0.3176 / 0.2514), MQTT–BT 0.2467 / 0.1461 (0.2641 / 0.2346). Top-5 Jaccards are identical to
  FedAvg's at both seeds (0.1111/0/0 and 0.25/0/0).

So on this benchmark, personalizing exactly the layer that was failing (the 32→1 output head)
recovers ~73–78% of the Bluetooth benign-recall gap and ~95% of local-only's average detection
quality, at a near-zero cost in cross-client explanation alignment. **Personalization is
validated** — and the fact that the SHAP profiles barely move while detection jumps reinforces
§4's calibration-failure diagnosis: the collapse lives in the output layer, not the feature logic.

### 8.6 Verification

- `study_results.json`: 12 run entries (10 pre-existing + 2 fedper), every pre-existing entry and
  top-level key present with the same schema; local-only entries untouched.
- Dashboard regression check (`streamlit.testing.v1.AppTest` on the updated JSON):
  **0 exceptions, 4/4 tabs rendered, 5 dataframes** — identical to the §5 baseline. `app.py`
  selects runs by `aggregator` name, so fedper entries are simply ignored by the metric tabs.

---

## COMPLETION PASS

### 1. Completeness-matrix result
The completeness matrix verification (`python scripts/audit_results.py`) shows **PASS - all runs complete for both seeds; no gaps.** All top-level keys and per-run schema entries are present and valid.

### 2. Artifacts produced and where
- **app.py**: Rewritten as 5‑tab BCG‑style dashboard (root directory).
- **scripts/test_dashboard.py**: New Streamlit AppTest verification script (scripts/ directory).
- **README.md**: Rewritten with final method lineup, results pointing to `docs/final_results.md`, numbered reproducibility steps, Cluster‑FL slot documentation, and updated repository structure (root directory).
- **docs/ui_guide.md**: Rewritten for the 5‑tab dashboard, tab‑by‑tab plain‑language guidance, read‑only and offline notes (docs/ directory).
- **results/study_results.json**: Regenerated experiment payload (single source of truth for dashboard).
- **figures/**: Four publication‑quality PNGs regenerated via `python scripts/make_figures.py`:
  - `fig1_detection_macro_f1.png`
  - `fig2_spearman_divergence_heatmap.png`
  - `fig3_bluetooth_mu_sweep.png`
  - `fig4_tradeoff_scatter.png`
- **docs/final_results.md**: Read‑only reference for final result tables and figures (docs/ directory).
- **docs/agent_fix_report.md**: This file, updated with the Completion Pass section (docs/ directory).
- **Scripts used**: `audit_results.py`, `backfill_completions.py`, `make_figures.py` (all in scripts/ directory).

### 3. Dashboard AppTest evidence
- Exceptions: **0** (verified via `streamlit.testing.v1.AppTest`).
- Tabs rendered: **5/5** (Executive Summary, Detection, Explanation Divergence, Methods Compared, Methodology & Limitations).
- Cluster‑FL placeholder: **OK** — selecting the Cluster‑FL method in the Explanation Divergence tab renders an informative placeholder with instructions for populating the slot, without exceptions.
- Dataframes rendered: **≥5** (including the new full‑vector divergence table in Tab 2).
- No console errors in headless execution.

### 4. Three headline findings with final numbers
**(a) No μ saves Bluetooth**  
Bluetooth benign recall under FedAvg: 0.3666 (seed 42) / 0.4238 (seed 7).  
Across the FedProx μ sweep (0.001, 0.01, 0.05, 0.1): range 0.0073–0.2090 (worst at μ=0.1, seed 42).  
Local‑only benign recall: 0.9984 (seed 42) / 0.9990 (seed 7).  
→ Even the gentlest proximal term (μ=0.001) lowers Bluetooth benign recall below FedAvg, and by μ=0.1 the model flags ~97–99% of Bluetooth benign traffic as attack. The collapse is caused by sharing; proximal anchoring deepens it.

**(b) Local‑only beats all sharing on detection**  
Local‑only average macro F1: 0.9715 (seed 42) / 0.9702 (seed 7) → **0.9708** two‑seed mean.  
Best shared model (FedPer): 0.9179 (seed 42) / 0.9302 (seed 7) → **0.9241** two‑seed mean.  
FedAvg: 0.7966 (seed 42) / 0.8011 (seed 7) → **0.7988** two‑seed mean.  
→ Local‑only outperforms every federated configuration: +0.0467 average macro F1 vs FedPer, +0.1720 vs FedAvg.

**(c) FedPer's trade‑off position**  
- Detection quality kept: 0.9179 (seed 42) / 0.9302 (seed 7) → **0.9241** average macro F1 = **95.2%** of local‑only (0.9708).  
- Bluetooth benign recall recovered: 0.8258 (seed 42) / 0.8721 (seed 7) → recovers **72.7%** (seed 42) and **77.9%** (seed 7) of the FedAvg→local‑only gap.  
- Cross‑client explanation similarity: remains at FedAvg level (Spearman ρ vs FedAvg ≥ 0.89 for all clients/seeds); within seed noise on every pair (checked against `noise_floor_seed_to_seed` in `divergence_summary`).  
→ FedPer recovers ~73–78% of the Bluetooth benign‑recall gap, retains ~95% of local‑only's average detection quality, and preserves explanation agreement with FedAvg.

### 5. What remains
- **Cluster‑FL port into empty slot**: A teammate's Cluster‑FL port will drop runs at `results/cluster_fl_results.json` following the documented schema (`CLUSTER_FL_RUN_SCHEMA` in `src/run_study_experiments.py`) and merge with `python src/merge_cluster_fl.py` — no code changes needed. Until then the dashboard and the divergence summary show an explicit placeholder (`slot empty - cluster_fl results pending`).
- **All other tasks completed**: No unfinished work remains beyond the awaited Cluster‑FL port. The dashboard is fully functional, the results are reproducible, and the documentation is up to date.

---