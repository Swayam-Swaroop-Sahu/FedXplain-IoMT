# FedXplain-IoMT: Interactive Dashboard User Guide

This guide describes how to run and navigate the **FedXplain-IoMT Streamlit Dashboard** for research demonstration and jury evaluation.

---

## 1. Quickstart: Running the Dashboard

The dashboard is completely lightweight, self-contained, and **read-only**. It reads precomputed experimental results from `results/study_results.json` and does not require GPU compute or live re-training.

### Prerequisites & Dependencies
```bash
pip install streamlit plotly pandas
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

The dashboard features a fixed header block with the embedded system architecture diagram, followed by 4 focused analytical tabs:

### Fixed Header Block (Always Visible)
- **Problem Statement & Research Question**: Introduces the clinical IoMT threat landscape, healthcare privacy constraints (HIPAA/GDPR), and the explanation divergence question.
- **Architecture Diagram**: High-resolution visualization showing Phase 1 (Federated Training across heterogeneous protocol clients with FedAvg/FedProx) and Phase 2 (Post-Training Explainability & Cross-Client Divergence Analysis).

### Tab 1: Detection Performance
- **Purpose**: Client-by-client performance metrics across Wi-Fi, MQTT, and Bluetooth.
- **Controls**: Seed selector (Seed 42 vs. Seed 7).
- **Dynamic Interpretations**: In-place summary stating macro F1 shifts under FedProx.
- **Failure Callout & Highlighting**: Automatically flags when a client suffers a classification collapse (e.g., Bluetooth benign recall dropping to ~4%), with highlighted table rows.

### Tab 2: Explanation Divergence
- **Purpose**: Visualizes model-agnostic feature attributions (SHAP) and pairwise set-theoretic similarity.
- **Controls**: Dropdown to select model configurations (FedAvg vs. FedProx $\mu=0.01, 0.1$).
- **Visuals**:
  - Interactive horizontal bar chart of Top-10 SHAP features.
  - $3 \times 3$ Jaccard divergence similarity heatmap.
  - In-place summary text and trade-off reality checks cross-referencing detection failure against apparent feature shifts.

### Tab 3: FedAvg vs FedProx
- **Purpose**: Side-by-side comparison of detection performance and explanation overlap.
- **Findings**: Paper-style narrative findings delivering definitive data-driven verdicts for Wi-Fi (neutral), MQTT (positive), and Bluetooth (collapse).

### Tab 4: Methodology and Limitations
- **Purpose**: Scientific rigor checklist outlining verified items vs. planned full-study milestones.
- **Noise Floor Table**: Preliminary seed-to-seed stability metrics ($n=2$ seeds).
