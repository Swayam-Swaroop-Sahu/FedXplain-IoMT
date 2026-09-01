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

The dashboard contains 5 dedicated tabs designed for an interactive 5-minute presentation:

### Tab 1: 📖 Overview & Architecture
- **Purpose:** Introduces the clinical IoMT threat landscape, healthcare privacy constraints, and the core research question: *Do federated models learn divergent local decision logic across physical protocols?*
- **Key Visual:** Interactive ASCII architectural diagram illustrating client partitioning and post-hoc SHAP extraction.

### Tab 2: 📈 Detection Performance
- **Purpose:** Interactive performance metrics across Wi-Fi, MQTT, and Bluetooth clients.
- **Controls:** Toggle between initial **POC Scale** (5 rounds, 5k samples) and **Study Scale** (10 rounds, 166k samples), and select random seeds (Seed 42 vs Seed 7).
- **Class-Imbalance Transparency:** Displays Macro F1 alongside Binary F1, highlighting benign precision metrics to guard against class-imbalance false positives in Wi-Fi traffic.

### Tab 3: 🧠 Explanation Divergence
- **Purpose:** Visualizes model-agnostic feature attributions (SHAP) and pairwise set-theoretic similarity.
- **Controls:** Dropdown to switch between **FedAvg**, **FedProx ($\mu=0.01$)**, and **FedProx ($\mu=0.1$)** across seeds.
- **Visuals:** 
  - Interactive horizontal bar chart of Top-10 SHAP features with hover tooltips.
  - $3 \times 3$ Jaccard divergence similarity heatmap.
  - Automated interpretation text explaining why Bluetooth exhibits near-zero overlap with IP-based protocols.

### Tab 4: ⚖️ FedAvg vs FedProx Comparison
- **Purpose:** Direct side-by-side comparison of predictive performance and explanatory divergence between standard Federated Averaging and proximal regularization.
- **Key Takeaway:** Demonstrates that $\mu=0.01$ stabilizes MQTT client detection (+9.0pp Macro F1) while allowing protocol-specific feature specialization.

### Tab 5: 🔬 Methodology & Limitations
- **Purpose:** Scientific rigor checklist outlining what has been validated (multi-round scaling, 45-feature schema assertions) and what remains for the full multi-seed study (10-seed confidence intervals, logarithmic $\mu$ sweep).
- **Noise Floor Table:** Displays preliminary seed-to-seed stability metrics ($n=2$ seeds).
