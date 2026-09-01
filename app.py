"""FedXplain-IoMT Research Dashboard.

Interactive demonstration dashboard for jury presentations and research analysis.
Visualizes federated learning detection metrics, class imbalance diagnostics,
and post-hoc SHAP explanation divergence across IoMT protocols.
"""

import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------
# Page Configuration & Styling
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="FedXplain-IoMT: Explainable Federated IoMT IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .warning-box {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 12px 0;
        font-size: 0.95rem;
    }
    .info-box {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 12px 0;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Data Loading (Read-only from results/study_results.json)
# ----------------------------------------------------------------------
RESULTS_PATH = os.path.join("results", "study_results.json")


@st.cache_data
def load_study_data() -> dict | None:
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


study_data = load_study_data()

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown('<div class="main-header">🛡️ FedXplain-IoMT Research Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Investigating Cross-Client Explanation Divergence in Federated Learning for Heterogeneous Medical IoT</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.image("https://raw.githubusercontent.com/flower/flower/main/doc/source/_static/img/flower-logo.svg", width=160)
    st.markdown("### 📊 Experiment Settings")
    if study_data:
        meta = study_data.get("run_metadata", {})
        st.markdown(f"**Dataset Scale**: `{meta.get('data_dir', 'data/study')}`")
        st.markdown(f"**FL Rounds**: `{meta.get('n_rounds', 10)}` rounds")
        st.markdown(f"**Local Epochs**: `{meta.get('n_local_epochs', 3)}` epochs/round")
        st.markdown(f"**Seeds Evaluated**: `{meta.get('seeds', [42, 7])}`")
        st.markdown(f"**Proximal Regularizers**: `mu in {meta.get('mu_values', [0.01, 0.1])}`")
    else:
        st.warning("⚠️ `results/study_results.json` not found. Please run `src/run_study_experiments.py` first.")

    st.markdown("---")
    st.markdown("### 🔍 IoMT Protocols")
    st.markdown("- **Wi-Fi**: High-throughput telemetry & imaging (TCP/IP)")
    st.markdown("- **MQTT**: Lightweight sensor messaging (TCP/IP)")
    st.markdown("- **Bluetooth**: Short-range wearables (HCI H4 Non-IP)")
    st.markdown("---")
    st.caption("FedXplain-IoMT Research Team • CICIoMT2024 Benchmark")


# ----------------------------------------------------------------------
# Navigation Tabs
# ----------------------------------------------------------------------
tab_overview, tab_perf, tab_divergence, tab_compare, tab_methods = st.tabs([
    "📖 1. Overview & Architecture",
    "📈 2. Detection Performance",
    "🧠 3. Explanation Divergence",
    "⚖️ 4. FedAvg vs FedProx Comparison",
    "🔬 5. Methodology & Limitations",
])


# ======================================================================
# TAB 1: Overview & Architecture
# ======================================================================
with tab_overview:
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("### 🎯 Research Motivation")
        st.markdown("""
        In clinical healthcare environments, Internet of Medical Things (IoMT) devices operate across heterogeneous network architectures:
        Wi-Fi for high-bandwidth telemetry, MQTT for bedside sensor publishing, and Bluetooth/BLE for peripheral monitoring.

        Under healthcare privacy mandates (HIPAA/GDPR), raw packet streams cannot be aggregated centrally. **Federated Learning (FL)**
        solves this by training a unified global intrusion detection model across hospital partitions without sharing patient telemetry.
        """)

        st.markdown("### ❓ The Core Problem: Cross-Client Explanation Divergence")
        st.markdown("""
        Standard federated intrusion detection research evaluates models exclusively on global accuracy or binary F1 score.
        However, **identical global model weights ($\theta^*$) may achieve uniform predictive metrics while relying on fundamentally
        contradictory or disjoint decision logic** across different protocol clients.

        If a hospital cybersecurity analyst on a Wi-Fi ward sees attacks flagged by TCP header resets (`rst_count`), while a Bluetooth
        wearable ward flags the same global model prediction via volumetric flow statistics (`Tot sum`, `Magnitue`), **cross-subnet automated
        containment policies cannot be trusted without explanation alignment**.
        """)

    with col2:
        st.markdown("### 🏗️ System Architecture")
        st.code("""
         +----------------------------------+
         | Central Orchestrator / Server    |
         | (FedAvg / FedProx Aggregator)    |
         +----------------+-----------------+
                          |
             Global Weights w_t | Aggregate w_{t+1}
                          v
         +----------------+-----------------+
         |                |                 |
         v                v                 v
   +-----------+    +-----------+    +-----------+
   | Client 0  |    | Client 1  |    | Client 2  |
   |   Wi-Fi   |    |   MQTT    |    | Bluetooth |
   | (TCP/IP)  |    | (TCP/IP)  |    | (HCI H4)  |
   +-----+-----+    +-----+-----+    +-----+-----+
         |                |                 |
         +----------------+-----------------+
                          |
                          v
             +--------------------------+
             | SHAP DeepExplainer       |
             | + Top-5 Jaccard Analysis |
             +--------------------------+
        """, language="text")

    st.markdown("---")
    st.markdown("### 📌 Key Scientific Insights from this Study")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        <div class="metric-card">
            <h4>1. Structural Protocol Divide</h4>
            <p>IP-based clients (Wi-Fi, MQTT) prioritize transport-layer TCP flags, whereas Bluetooth relies purely on volumetric flow features, yielding near-zero explanation overlap (J ≤ 0.11).</p>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="metric-card">
            <h4>2. Proximal Regularization Effect</h4>
            <p>FedProx (μ=0.01) stabilizes minority client learning (MQTT Macro F1 +9.0pp) while permitting protocol-specialized decision logic.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        st.markdown("""
        <div class="metric-card">
            <h4>3. Class Imbalance Rigor</h4>
            <p>Uncovered that high binary F1 on Wi-Fi was inflated by 96% attack prevalence; established Macro F1 and benign precision as standard reporting metrics.</p>
        </div>
        """, unsafe_allow_html=True)


# ======================================================================
# TAB 2: Detection Performance
# ======================================================================
with tab_perf:
    st.markdown("### 📊 Federated Intrusion Detection Performance")

    scale_toggle = st.radio(
        "Select Dataset Evaluation Scale:",
        ["Study Scale (data/study, 10 Rounds, 3 Local Epochs)", "POC Scale (data/poc, 5 Rounds, 2 Local Epochs)"],
        horizontal=True,
    )

    if "Study Scale" in scale_toggle and study_data:
        runs = study_data.get("runs", [])
        seed_selected = st.selectbox("Select Random Seed:", [42, 7], index=0)
        filtered_runs = [r for r in runs if r.get("seed") == seed_selected]

        st.markdown("""
        <div class="warning-box">
            ⚠️ <b>Class Imbalance Audit Note:</b> In the Wi-Fi client partition, 96.4% of test traffic is attack traffic.
            While binary F1 appears near-perfect (>0.99), <b>Macro F1</b> and <b>Benign Precision</b> provide an honest view of benign misclassification rates.
        </div>
        """, unsafe_allow_html=True)

        rows = []
        for r in filtered_runs:
            agg_name = r["aggregator"].upper()
            if r.get("mu") is not None:
                agg_name += f" (mu={r['mu']})"

            for proto, metrics in r["client_metrics"].items():
                rows.append({
                    "Aggregator": agg_name,
                    "Client Protocol": proto.upper(),
                    "Macro F1": f"{metrics['f1_macro']:.4f}",
                    "Binary F1 (Attack)": f"{metrics['f1_binary']:.4f}",
                    "Benign Precision": f"{metrics['benign_precision']:.4f}",
                    "Benign Recall": f"{metrics['benign_recall']:.4f}",
                    "Attack Recall": f"{metrics['attack_recall']:.4f}",
                    "Accuracy": f"{metrics['accuracy']:.4f}",
                    "Test Benign/Attack": f"{metrics.get('n_benign', '—')} / {metrics.get('n_attack', '—')}",
                })

        df_perf = pd.DataFrame(rows)
        st.dataframe(df_perf, use_container_width=True, height=360)

        # Visual Comparison Bar Chart
        st.markdown("#### Macro F1 vs Binary F1 per Client")
        chart_data = []
        for r in filtered_runs:
            agg_label = r["aggregator"] + (f" (mu={r['mu']})" if r.get("mu") else "")
            for proto, metrics in r["client_metrics"].items():
                chart_data.append({"Aggregator": agg_label, "Protocol": proto.upper(), "Metric": "Macro F1", "Score": metrics["f1_macro"]})
                chart_data.append({"Aggregator": agg_label, "Protocol": proto.upper(), "Metric": "Binary F1", "Score": metrics["f1_binary"]})

        df_chart = pd.DataFrame(chart_data)
        fig = px.bar(
            df_chart,
            x="Protocol",
            y="Score",
            color="Metric",
            barmode="group",
            facet_col="Aggregator",
            color_discrete_map={"Macro F1": "#2563eb", "Binary F1": "#93c5fd"},
            range_y=[0, 1.05],
            height=380,
        )
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    else:
        # POC Reference Data
        st.markdown("#### Initial POC Reference Table (5 Rounds, 2 Local Epochs, Seed 42)")
        poc_rows = [
            {"Aggregator": "Centralized Baseline (10 ep)", "Wi-Fi Macro F1": "—", "MQTT Macro F1": "—", "Bluetooth Macro F1": "—", "Macro Avg F1": "0.9875 (Pooled)"},
            {"Aggregator": "FedAvg", "Wi-Fi Macro F1": "0.8604", "MQTT Macro F1": "0.8607", "Bluetooth Macro F1": "0.7198", "Macro Avg F1": "0.8560"},
            {"Aggregator": "FedProx (mu=0.01)", "Wi-Fi Macro F1": "0.8604", "MQTT Macro F1": "0.8990", "Bluetooth Macro F1": "0.8058", "Macro Avg F1": "0.8974"},
        ]
        st.dataframe(pd.DataFrame(poc_rows), use_container_width=True)


# ======================================================================
# TAB 3: Explanation Divergence
# ======================================================================
with tab_divergence:
    st.markdown("### 🧠 SHAP Attribution & Jaccard Divergence Matrix")

    if study_data:
        runs = study_data.get("runs", [])
        run_options = [
            f"{r['aggregator'].upper()}" + (f" (mu={r['mu']})" if r.get("mu") else "") + f" - Seed {r['seed']}"
            for r in runs
        ]
        selected_idx = st.selectbox("Select Aggregator & Seed Configuration:", range(len(run_options)), format_func=lambda i: run_options[i])
        active_run = runs[selected_idx]

        col_top10, col_jaccard = st.columns([3, 2])

        with col_top10:
            st.markdown("#### Top-10 SHAP Feature Attributions by Client")
            proto_select = st.segmented_control("Select Protocol Client:", ["WIFI", "MQTT", "BLUETOOTH"], default="WIFI")
            proto_key = proto_select.lower()

            top10_list = active_run["shap_top10_per_client"].get(proto_key, [])
            if top10_list:
                df_top10 = pd.DataFrame(top10_list, columns=["Feature", "Mean |SHAP|"]).sort_values("Mean |SHAP|", ascending=True)
                fig_shap = px.bar(
                    df_top10,
                    x="Mean |SHAP|",
                    y="Feature",
                    orientation="h",
                    title=f"Top 10 SHAP Features - {proto_select} Client",
                    color="Mean |SHAP|",
                    color_continuous_scale="Blues",
                    height=420,
                )
                fig_shap.update_layout(margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_shap, use_container_width=True)
            else:
                st.info("No SHAP values recorded for this selection.")

        with col_jaccard:
            st.markdown("#### Cross-Client Top-5 Jaccard Similarity")
            j_map = active_run["jaccard_pairwise"]
            j_matrix = [
                [1.0, j_map.get("wifi-mqtt", 0.0), j_map.get("wifi-bluetooth", 0.0)],
                [j_map.get("wifi-mqtt", 0.0), 1.0, j_map.get("mqtt-bluetooth", 0.0)],
                [j_map.get("wifi-bluetooth", 0.0), j_map.get("mqtt-bluetooth", 0.0), 1.0],
            ]
            protocols = ["Wi-Fi", "MQTT", "Bluetooth"]

            fig_hm = go.Figure(data=go.Heatmap(
                z=j_matrix,
                x=protocols,
                y=protocols,
                colorscale="Blues",
                zmin=0,
                zmax=1.0,
                text=[[f"{val:.4f}" for val in row] for row in j_matrix],
                texttemplate="%{text}",
                textfont={"size": 14},
            ))
            fig_hm.update_layout(
                title=f"Top-5 Feature Overlap (Jaccard Index)",
                height=420,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_hm, use_container_width=True)

        # Auto-generated explanation interpretation
        j_wm = j_map.get("wifi-mqtt", 0.0)
        j_wb = j_map.get("wifi-bluetooth", 0.0)
        j_mb = j_map.get("mqtt-bluetooth", 0.0)

        st.markdown(f"""
        <div class="info-box">
            💡 <b>Automated Divergence Interpretation:</b><br>
            • <b>Wi-Fi vs MQTT (J = {j_wm:.4f})</b>: Moderate overlap driven by shared IP transport flag features (<code>rst_count</code>, <code>ack_flag_number</code>).<br>
            • <b>Wi-Fi vs Bluetooth (J = {j_wb:.4f}) & MQTT vs Bluetooth (J = {j_mb:.4f})</b>: Severe explanation divergence. Bluetooth operates over non-IP HCI H4 packets, forcing the global model to rely exclusively on packet volume metrics (<code>Tot sum</code>, <code>Number</code>, <code>Weight</code>).
        </div>
        """, unsafe_allow_html=True)


# ======================================================================
# TAB 4: FedAvg vs FedProx Comparison
# ======================================================================
with tab_compare:
    st.markdown("### ⚖️ Side-by-Side Comparison: FedAvg vs FedProx")

    if study_data:
        runs = study_data.get("runs", [])
        s42_runs = {f"{r['aggregator']}_{r['mu']}": r for r in runs if r["seed"] == 42}

        if "fedavg_None" in s42_runs and "fedprox_0.01" in s42_runs:
            avg_run = s42_runs["fedavg_None"]
            prox001_run = s42_runs["fedprox_0.01"]
            prox01_run = s42_runs.get("fedprox_0.1")

            comp_rows = []
            for proto in ["wifi", "mqtt", "bluetooth"]:
                m_avg = avg_run["client_metrics"][proto]
                m_p001 = prox001_run["client_metrics"][proto]
                m_p01 = prox01_run["client_metrics"][proto] if prox01_run else None

                comp_rows.append({
                    "Protocol": proto.upper(),
                    "FedAvg Macro F1": f"{m_avg['f1_macro']:.4f}",
                    "FedProx (mu=0.01) Macro F1": f"{m_p001['f1_macro']:.4f}",
                    "FedProx (mu=0.1) Macro F1": f"{m_p01['f1_macro']:.4f}" if m_p01 else "—",
                    "Δ (Prox0.01 - Avg)": f"{m_p001['f1_macro'] - m_avg['f1_macro']:+.4f}",
                })

            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

            # Jaccard Overlap Side by Side
            st.markdown("#### Pairwise Explanation Overlap Comparison (Jaccard Index)")
            j_rows = []
            for pair in ["wifi-mqtt", "wifi-bluetooth", "mqtt-bluetooth"]:
                j_a = avg_run["jaccard_pairwise"].get(pair, 0.0)
                j_p001 = prox001_run["jaccard_pairwise"].get(pair, 0.0)
                j_p01 = prox01_run["jaccard_pairwise"].get(pair, 0.0) if prox01_run else 0.0
                j_rows.append({
                    "Client Pair": pair.upper(),
                    "FedAvg Jaccard": f"{j_a:.4f}",
                    "FedProx (mu=0.01) Jaccard": f"{j_p001:.4f}",
                    "FedProx (mu=0.1) Jaccard": f"{j_p01:.4f}",
                    "Shared Features (Prox mu=0.01)": ", ".join(prox001_run["shared_features"].get(pair, [])) or "None",
                })
            st.dataframe(pd.DataFrame(j_rows), use_container_width=True)

            st.markdown("""
            <div class="info-box">
                <b>Takeaway:</b> Proximal regularization with μ=0.01 significantly improves MQTT client macro detection performance (+9.0pp)
                while preserving specialized local feature attribution profiles across physical layers.
            </div>
            """, unsafe_allow_html=True)


# ======================================================================
# TAB 5: Methodology & Limitations
# ======================================================================
with tab_methods:
    st.markdown("### 🔬 Scientific Methodology & Preliminary Noise Floor")

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("#### ✅ What This Extended Pilot Demonstrates")
        st.markdown("""
        1. **Multi-Round Stability**: Scaled from 5 rounds (POC) to 10 communication rounds on 166,000+ real IoMT traffic flows.
        2. **Reproducible Preprocessing Guardrails**: Exact 45-feature schema enforcement across all heterogeneous physical layers.
        3. **Class-Imbalance Transparency**: Macro F1 and per-class precision/recall reported by default.
        4. **Mathematical Divergence Tracking**: Quantitative Jaccard indices across multi-seed evaluations.
        """)

    with col_m2:
        st.markdown("#### ⚠️ What Has NOT Yet Been Statistically Claimed")
        st.markdown("""
        1. **Noise Floor Calibration (n=2 seeds)**: 2 random seeds provide a preliminary consistency check, not a full 10-seed Gaussian confidence interval.
        2. **Hyperparameter Grid**: Explored μ ∈ {0.01, 0.1}. A continuous logarithmic grid (10^-4 to 10^0) remains for the full publication study.
        3. **Alternative Aggregators**: Evaluated FedAvg and FedProx; SCAFFOLD, FedOpt, and FedNova remain planned roadmap items.
        """)

    if study_data and "preliminary_noise_floor" in study_data:
        st.markdown("---")
        st.markdown("#### 📉 Preliminary Seed-to-Seed Stability (n=2 Seeds)")
        nf = study_data["preliminary_noise_floor"]
        nf_rows = []
        for agg_name, d in nf.items():
            stabs = d.get("per_client_top5_stability", {})
            nf_rows.append({
                "Aggregator Configuration": agg_name.upper(),
                "Wi-Fi Seed Overlap": f"{stabs.get('wifi', {}).get('seed_to_seed_jaccard', 0.0):.4f}",
                "MQTT Seed Overlap": f"{stabs.get('mqtt', {}).get('seed_to_seed_jaccard', 0.0):.4f}",
                "Bluetooth Seed Overlap": f"{stabs.get('bluetooth', {}).get('seed_to_seed_jaccard', 0.0):.4f}",
                "Status": d.get("status", "Preliminary"),
            })
        st.dataframe(pd.DataFrame(nf_rows), use_container_width=True)
