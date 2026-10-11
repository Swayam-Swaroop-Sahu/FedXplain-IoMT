# FedXplain-IoMT: Interactive Dashboard User Guide

This guide describes how to run and navigate the **FedXplain-IoMT Streamlit Dashboard** for research demonstration and jury evaluation.

---

## 1. Quickstart: Running the Dashboard

The dashboard is completely lightweight, self-contained, and **read-only**. It reads precomputed experimental results from `results/study_results.json` and does not require GPU compute or live re-training.

### Prerequisites & Dependencies
```bash
pip install -r requirements.txt
```

### Launch Command
From the repository root directory (`FedXplain-IoMT/`), run:
```bash
streamlit run app.py
```

The application will automatically open in your default browser at `http://localhost:8501`.

> **Note for Jury Evaluation:** No active internet connection or GPU hardware is required to run the dashboard during a presentation.

---

## 2. Dashboard Navigation Structure

The dashboard features a fixed header block with the embedded system architecture diagram, followed by **5 focused analytical tabs**:

### Fixed Header Block (Always Visible)
- **Problem Statement & Research Question**: Introduces the clinical IoMT threat landscape, healthcare privacy constraints (HIPAA/GDPR), and the explanation divergence question.
- **Architecture Diagram**: High-resolution visualization showing Phase 1 (Federated Training across heterogeneous protocol clients with FedAvg/FedProx) and Phase 2 (Post-Training Explainability & Cross-Client Divergence Analysis).

### Tab 0: Executive Summary
- **Purpose**: High-level interpretation of the study's core thesis and key performance indicators.
- **Content**:
  - One-sentence thesis: *"One shared detector cannot serve three structurally different hospital protocols: on this benchmark the shared model collapses on Bluetooth, no proximal strength rescues it, and only personalizing the final layer — or not sharing at all — does."*
  - Four KPI cards:
    1. Best shared-model average macro F1 (with method label).
    2. Local-only average macro F1 (with delta vs. best shared model).
    3. Worst FedProx Bluetooth benign recall (with μ, seed, and local-only minimum).
    4. FedAvg Wi‑Fi–Bluetooth Spearman ρ (two‑seed mean ± spread).
  - Embedded Figure 4 (detection quality vs. cross‑client explanation consistency) with caption.
  - Two‑column findings: *"What we found"* and *"What we didn't prove yet"*.

### Tab 1: Detection
- **Purpose**: Client-by-client performance metrics across Wi-Fi, MQTT, and Bluetooth.
- **Sections**:
  1. **Unified detection table**: Macro F1 for each method (FedAvg, FedProx×4 μ, FedPer, Local-only, Centralized) per client and seed, with average columns and mean ± spread.
  2. **Bluetooth benign‑recall table**: Same layout as above, highlighting any cell < 0.5 (indicating >50% false positives on benign traffic) in teal‑red.
  3. **Detailed per‑seed metrics**: Seed selector (Seed 42 vs. Seed 7) showing F1, recall, precision, and accuracy per protocol, with collapse status (Bluetooth benign recall < 0.15 AND macro F1 < 0.50).
- **Callout**: Automatic flagging of collapsed rows and a Bluetooth collapse callout summarizing the drop in benign recall under sharing.

### Tab 2: Explanation Divergence
- **Purpose**: Visualizes model-agnostic feature attributions (SHAP) and pairwise set-theoretic similarity.
- **Controls**:
  - Method selector: All methods (FedAvg, FedProx×4 μ, FedPer, Local-only, Cluster-FL).
  - Seed selector: Seed 42 vs. Seed 7.
  - Protocol selector (for SHAP chart): Wi‑Fi, MQTT, Bluetooth.
- **Visuals**:
  - Interactive horizontal bar chart of Top‑10 mean |SHAP| values (averaged across clients), with colormap from light teal to navy.
  - $3 \times 3$ Jaccard similarity heatmap (Top‑5 feature sets), annotated with values.
- **Tables**:
  - Full‑metrics table: Jaccard, Spearman ρ, and cosine similarity for each client pair (Wi‑Fi–MQTT, Wi‑Fi–Bluetooth, MQTT–Bluetooth) for both seeds, plus seed‑to‑seed |difference| (noise floor).
  - Divergence scope caption (stated per run).
  - For FedPer: side‑by‑side comparison tables (Top‑5 Jaccard, Spearman ρ, Cosine) vs. FedAvg and vs. Local‑only per client.
  - For Local‑only: local‑vs‑global comparison table (if available).
- **Note**: Selecting Cluster‑FL (or any missing run) renders an explicit placeholder with instructions for populating the slot.

### Tab 3: Methods Compared
- **Purpose**: Side‑by‑side comparison of detection performance and explanation overlap across methods.
- **Sections**:
  1. **Centerpiece chart**: Bluetooth benign recall vs. μ (categorical x‑axis: FedAvg(μ=0), μ=0.001, μ=0.01, μ=0.05, μ=0.1). Two lines (one per seed) with local‑only (teal dashed) and FedPer (amber dotted) reference lines.
  2. **Secondary chart**: Average macro F1 per method (two‑seed mean) with error bars (seed spread). Methods colored: FedAvg/FedProx (blue), FedPer (amber), Local‑only (teal), Centralized (grey).
  3. **Written findings**: Three data‑driven verdicts:
     - No μ saves Bluetooth (proximal term worsens benign recall).
     - Local‑only beats all sharing on detection.
     - FedPer’s trade‑off position (recovers ~73–78% of Bluetooth benign‑recall gap, retains ~95% of local‑only's detection, explanation similarity at FedAvg level).

### Tab 4: Methodology & Limitations
- **Purpose**: Scientific rigor checklist outlining verified items vs. planned full‑study milestones.
- **Content**:
  - Study‑scale bullets: 2 seeds (directional), 10 federated + 2 local-only + 2 FedPer runs, full 45‑dim SHAP vectors, noise floor across ALL aggregators.
  - Verified vs. planned two‑column lists.
  - Cluster‑FL status: "in progress — port pending" with instructions for teammates (drop JSON array at `results/cluster_fl_results.json` per `CLUSTER_FL_RUN_SCHEMA`, then run `python src/merge_cluster_fl.py`).
  - Seed‑to‑seed stability table (all aggregators) from preliminary noise floor.
  - System architecture diagram (`docs/system_architecture.png`).

> **Read‑only & Offline**: The dashboard reads only from `results/study_results.json` and performs no live computation. All numbers are fixed after running `python src/run_study_experiments.py`.

---

## 3. Reproducibility Note

To regenerate the dashboard data from scratch:
1. Install dependencies: `pip install -r requirements.txt`
2. Run the experiment suite: `python src/run_study_experiments.py`
3. (Optional) Verify completeness: `python scripts/audit_results.py`
4. Generate figures: `python scripts/make_figures.py`
5. Launch the dashboard: `streamlit run app.py`

The Cluster‑FL slot remains empty until a teammate ports the method and merges results via the provided script.
