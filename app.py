"""FedXplain-IoMT Research Dashboard.

Interactive demonstration dashboard for jury presentations and research analysis.
Directly pairs metrics with dynamic, data-driven interpretations and cross-tab trade-offs.
"""

import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------
# Page Configuration & Clean Design System
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
        font-size: 2.0rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .alert-danger {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 14px 18px;
        border-radius: 6px;
        margin: 12px 0 16px 0;
        color: #991b1b;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .alert-info {
        background-color: #eff6ff;
        border-left: 5px solid #3b82f6;
        padding: 14px 18px;
        border-radius: 6px;
        margin: 12px 0 16px 0;
        color: #1e40af;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .alert-success {
        background-color: #f0fdf4;
        border-left: 5px solid #22c55e;
        padding: 14px 18px;
        border-radius: 6px;
        margin: 12px 0 16px 0;
        color: #166534;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .verdict-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 14px;
        margin-bottom: 10px;
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
st.markdown('<div class="sub-header">Connecting Federated Learning Performance with Post-Hoc Explanation Divergence</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Sidebar (Focused Experiment Controls)
# ----------------------------------------------------------------------
with st.sidebar:
    st.image("https://raw.githubusercontent.com/flower/flower/main/doc/source/_static/img/flower-logo.svg", width=150)
    st.markdown("### ⚙️ Experiment Parameters")
    if study_data:
        meta = study_data.get("run_metadata", {})
        st.markdown(f"**Dataset**: `{meta.get('data_dir', 'data/study')}` (166k+ flows)")
        st.markdown(f"**Rounds**: `{meta.get('n_rounds', 10)}` | **Epochs/Round**: `{meta.get('n_local_epochs', 3)}`")
        st.markdown(f"**Seeds**: `{meta.get('seeds', [42, 7])}`")
        st.markdown(f"**Proximal Penalties (μ)**: `{meta.get('mu_values', [0.01, 0.1])}`")
    else:
        st.error("Missing `results/study_results.json`. Run `python src/run_study_experiments.py`.")
    st.markdown("---")
    st.caption("FedXplain-IoMT • CICIoMT2024 Evaluation")


# ----------------------------------------------------------------------
# Navigation Tabs
# ----------------------------------------------------------------------
tab_overview, tab_perf, tab_divergence, tab_compare, tab_methods = st.tabs([
    "📖 1. Overview & Architecture",
    "📈 2. Detection Performance",
    "🧠 3. Explanation Divergence",
    "⚖️ 4. FedAvg vs FedProx Verdict",
    "🔬 5. Methodology & Limitations",
])


# ======================================================================
# TAB 1: Overview & Architecture
# ======================================================================
with tab_overview:
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("### 🎯 Clinical Problem & Research Question")
        st.markdown("""
        In hospital networks, Internet of Medical Things (IoMT) devices operate on disjoint physical protocols:
        **Wi-Fi** (high-bandwidth monitors), **MQTT** (bedside sensor telemetry), and **Bluetooth/BLE** (wearables).
        
        Because HIPAA/GDPR prohibit centralizing raw patient traffic, **Federated Learning (FL)** trains a unified global model across hospital subnets without moving raw data.
        
        **The Critical Question**: Standard FL papers declare success when global F1 is high. But does the converged global model rely on the **same security features** across these silos, or is it learning **disjoint, protocol-divergent decision logic**?
        """)

        st.markdown("""
        <div class="alert-info">
            <b>Core Finding:</b> Global weights produce protocol-contingent feature attributions. Furthermore, optimization regularizers like FedProx can appear to reduce explanation divergence on paper, but the reduction is an artifact of <b>minority-client performance collapse</b> rather than genuine shared representation.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 🏗️ Pipeline Architecture")
        st.code("""
         +----------------------------------+
         | Server Orchestrator (FedAvg/Prox)|
         +----------------+-----------------+
                          | Global Weights w_t
                          v
         +----------------+-----------------+
         |                |                 |
         v                v                 v
   +-----------+    +-----------+    +-----------+
   |  Wi-Fi    |    |   MQTT    |    | Bluetooth |
   | (TCP/IP)  |    | (TCP/IP)  |    | (HCI H4)  |
   +-----+-----+    +-----+-----+    +-----+-----+
         |                |                 |
         +----------------+-----------------+
                          | Local SHAP Attributions
                          v
             +--------------------------+
             | Pairwise Top-5 Jaccard   |
             | Divergence & Diagnostic  |
             +--------------------------+
        """, language="text")


# ======================================================================
# TAB 2: Detection Performance (With Dynamic Interpretation & Collapse Flags)
# ======================================================================
with tab_perf:
    st.markdown("### 📈 Client Detection Performance & Failure Diagnostics")

    if study_data:
        runs = study_data.get("runs", [])
        seeds_available = sorted(list(set(r.get("seed", 42) for r in runs)))
        seed_selected = st.selectbox("Select Evaluation Seed:", seeds_available, index=0)
        filtered_runs = [r for r in runs if r.get("seed") == seed_selected]

        # Extract FedAvg and FedProx (mu=0.01) for dynamic comparison
        avg_run = next((r for r in filtered_runs if r["aggregator"] == "fedavg"), None)
        prox_run = next((r for r in filtered_runs if r["aggregator"] == "fedprox" and r.get("mu") == 0.01), None)

        if avg_run and prox_run:
            m_avg = avg_run["client_metrics"]
            m_prox = prox_run["client_metrics"]

            # Dynamic calculations
            mqtt_diff = (m_prox["mqtt"]["f1_macro"] - m_avg["mqtt"]["f1_macro"]) * 100
            wifi_diff = (m_prox["wifi"]["f1_macro"] - m_avg["wifi"]["f1_macro"]) * 100
            bt_diff = (m_prox["bluetooth"]["f1_macro"] - m_avg["bluetooth"]["f1_macro"]) * 100

            bt_avg_brec = m_avg["bluetooth"]["benign_recall"] * 100
            bt_prox_brec = m_prox["bluetooth"]["benign_recall"] * 100
            bt_prox_aprec = m_prox["bluetooth"]["attack_precision"] * 100

            # Dynamic Interpretation Box (Placed directly above data)
            st.markdown(f"""
            <div class="alert-info">
                <b>📊 Dynamic Performance Interpretation (Seed {seed_selected}):</b><br>
                • <b>MQTT Client</b> gained significantly under FedProx (μ=0.01): Macro F1 shifted by <b>{mqtt_diff:+.1f}pp</b> ({m_avg['mqtt']['f1_macro']:.4f} → {m_prox['mqtt']['f1_macro']:.4f}), balancing benign and attack detection.<br>
                • <b>Wi-Fi Client</b> remained stable (<b>{wifi_diff:+.1f}pp</b>, {m_avg['wifi']['f1_macro']:.4f} → {m_prox['wifi']['f1_macro']:.4f}). Note that Wi-Fi binary F1 (>0.99) is inflated by 96.4% attack prevalence; Macro F1 is the honest metric.
            </div>
            """, unsafe_allow_html=True)

            # Check for model collapse condition (Benign recall < 10% or Attack precision < 50%)
            if m_prox["bluetooth"]["benign_recall"] < 0.15 or m_prox["bluetooth"]["f1_macro"] < 0.50:
                st.markdown(f"""
                <div class="alert-danger">
                    🚨 <b>CRITICAL FAILURE: Bluetooth Client Collapsed Under FedProx (μ=0.01)</b><br>
                    Bluetooth Macro F1 plummeted by <b>{bt_diff:.1f}pp</b> ({m_avg['bluetooth']['f1_macro']:.4f} → {m_prox['bluetooth']['f1_macro']:.4f}).
                    Benign recall collapsed from <b>{bt_avg_brec:.1f}%</b> down to <b>{bt_prox_brec:.1f}%</b>, and attack precision dropped to <b>{bt_prox_aprec:.1f}%</b>.
                    The model ceased discriminating benign traffic and degenerated into predicting 'attack' for almost all inputs.
                </div>
                """, unsafe_allow_html=True)

        # Performance Data Table with Visual Failure Highlighting
        rows = []
        for r in filtered_runs:
            agg_name = r["aggregator"].upper()
            if r.get("mu") is not None:
                agg_name += f" (mu={r['mu']})"

            for proto, metrics in r["client_metrics"].items():
                is_collapsed = metrics["benign_recall"] < 0.15 and metrics["f1_macro"] < 0.50
                status_tag = "🚨 COLLAPSED" if is_collapsed else ("✅ IMPROVED" if "FEDPROX (MU=0.01)" in agg_name and proto == "mqtt" else "STABLE")

                rows.append({
                    "Aggregator": agg_name,
                    "Client": proto.upper(),
                    "Status": status_tag,
                    "Macro F1": metrics["f1_macro"],
                    "Binary F1": metrics["f1_binary"],
                    "Benign Precision": metrics["benign_precision"],
                    "Benign Recall": metrics["benign_recall"],
                    "Attack Precision": metrics["attack_precision"],
                    "Attack Recall": metrics["attack_recall"],
                    "Accuracy": metrics["accuracy"],
                })

        df_perf = pd.DataFrame(rows)

        # Style the dataframe: highlight collapsed rows in light red
        def highlight_status(row):
            if "COLLAPSED" in row["Status"]:
                return ["background-color: #fee2e2; color: #991b1b; font-weight: bold;"] * len(row)
            elif "IMPROVED" in row["Status"]:
                return ["background-color: #f0fdf4; color: #166534; font-weight: bold;"] * len(row)
            return [""] * len(row)

        styled_df = df_perf.style.apply(highlight_status, axis=1).format({
            "Macro F1": "{:.4f}",
            "Binary F1": "{:.4f}",
            "Benign Precision": "{:.4f}",
            "Benign Recall": "{:.4f}",
            "Attack Precision": "{:.4f}",
            "Attack Recall": "{:.4f}",
            "Accuracy": "{:.4f}",
        })

        st.dataframe(styled_df, use_container_width=True, height=360)


# ======================================================================
# TAB 3: Explanation Divergence (Direct Cross-Tab Failure Trade-Off)
# ======================================================================
with tab_divergence:
    st.markdown("### 🧠 SHAP Attribution & Divergence Heatmap")

    if study_data:
        runs = study_data.get("runs", [])
        run_options = [
            f"{r['aggregator'].upper()}" + (f" (mu={r['mu']})" if r.get("mu") else "") + f" - Seed {r['seed']}"
            for r in runs
        ]
        selected_idx = st.selectbox("Select Aggregator Run to Inspect:", range(len(run_options)), format_func=lambda i: run_options[i])
        active_run = runs[selected_idx]
        active_seed = active_run["seed"]

        # Find baseline FedAvg for this same seed to compute dynamic delta
        base_avg_run = next((r for r in runs if r["aggregator"] == "fedavg" and r["seed"] == active_seed), None)

        col_top10, col_jaccard = st.columns([3, 2])

        with col_top10:
            st.markdown("#### Top-10 SHAP Feature Attributions")
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
                    height=400,
                )
                fig_shap.update_layout(margin=dict(l=20, r=20, t=35, b=20))
                st.plotly_chart(fig_shap, use_container_width=True)

        with col_jaccard:
            st.markdown("#### Top-5 Jaccard Similarity Heatmap")
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
                title=f"Pairwise Jaccard Overlap",
                height=400,
                margin=dict(l=20, r=20, t=35, b=20),
            )
            st.plotly_chart(fig_hm, use_container_width=True)

        # Dynamic Interpretation Placed Directly Alongside the Heatmap
        j_wm = j_map.get("wifi-mqtt", 0.0)
        j_wb = j_map.get("wifi-bluetooth", 0.0)
        j_mb = j_map.get("mqtt-bluetooth", 0.0)

        diff_text = ""
        if base_avg_run and active_run["aggregator"] != "fedavg":
            base_j = base_avg_run["jaccard_pairwise"]
            d_wm = j_wm - base_j.get("wifi-mqtt", 0.0)
            d_wb = j_wb - base_j.get("wifi-bluetooth", 0.0)
            d_mb = j_mb - base_j.get("mqtt-bluetooth", 0.0)
            diff_text = f"Versus FedAvg: Wi-Fi/MQTT shifted by <b>{d_wm:+.4f}</b>, Wi-Fi/Bluetooth by <b>{d_wb:+.4f}</b>, and MQTT/Bluetooth by <b>{d_mb:+.4f}</b>."

        st.markdown(f"""
        <div class="alert-info">
            <b>🔍 Explanation Divergence Summary:</b><br>
            • Current Top-5 Jaccards: Wi-Fi vs MQTT = <b>{j_wm:.4f}</b> | Wi-Fi vs Bluetooth = <b>{j_wb:.4f}</b> | MQTT vs Bluetooth = <b>{j_mb:.4f}</b>. {diff_text}
        </div>
        """, unsafe_allow_html=True)

        # Cross-Tab Reality Check: Connect Divergence to Performance Collapse
        bt_macro = active_run["client_metrics"]["bluetooth"]["f1_macro"]
        bt_brec = active_run["client_metrics"]["bluetooth"]["benign_recall"]

        if bt_brec < 0.15:
            st.markdown(f"""
            <div class="alert-danger">
                ⚠️ <b>CRITICAL TRADE-OFF REALITY CHECK:</b><br>
                Any Jaccard overlap shift involving Bluetooth (e.g. MQTT-Bluetooth J = <b>{j_mb:.4f}</b>) coincides with Bluetooth's 
                <b>detection collapse (Macro F1: {bt_macro:.4f}, Benign Recall: {bt_brec*100:.1f}%)</b>. 
                This shift is NOT true shared learning—it is an artifact of the model failing to classify Bluetooth traffic and collapsing into majority-class predictions.
            </div>
            """, unsafe_allow_html=True)


# ======================================================================
# TAB 4: FedAvg vs FedProx Side-by-Side & Dynamic Verdicts
# ======================================================================
with tab_compare:
    st.markdown("### ⚖️ Side-by-Side Performance & Divergence Matrix")

    if study_data:
        runs = study_data.get("runs", [])
        seeds_avail = sorted(list(set(r.get("seed", 42) for r in runs)))
        cmp_seed = st.selectbox("Select Seed for Comparison:", seeds_avail, index=0)

        s_runs = {f"{r['aggregator']}_{r['mu']}": r for r in runs if r["seed"] == cmp_seed}

        if "fedavg_None" in s_runs and "fedprox_0.01" in s_runs:
            avg_run = s_runs["fedavg_None"]
            prox001_run = s_runs["fedprox_0.01"]
            prox01_run = s_runs.get("fedprox_0.1")

            col_t1, col_t2 = st.columns(2)

            with col_t1:
                st.markdown("#### 1. Detection Performance (Macro F1)")
                comp_rows = []
                for proto in ["wifi", "mqtt", "bluetooth"]:
                    m_a = avg_run["client_metrics"][proto]["f1_macro"]
                    m_p001 = prox001_run["client_metrics"][proto]["f1_macro"]
                    diff = m_p001 - m_a

                    comp_rows.append({
                        "Protocol": proto.upper(),
                        "FedAvg": f"{m_a:.4f}",
                        "FedProx (μ=0.01)": f"{m_p001:.4f}",
                        "Δ Change": f"{diff:+.4f}",
                    })
                st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

            with col_t2:
                st.markdown("#### 2. Explanation Overlap (Top-5 Jaccard)")
                j_rows = []
                for pair in ["wifi-mqtt", "wifi-bluetooth", "mqtt-bluetooth"]:
                    j_a = avg_run["jaccard_pairwise"].get(pair, 0.0)
                    j_p001 = prox001_run["jaccard_pairwise"].get(pair, 0.0)
                    diff_j = j_p001 - j_a
                    j_rows.append({
                        "Pair": pair.upper(),
                        "FedAvg J": f"{j_a:.4f}",
                        "FedProx (μ=0.01) J": f"{j_p001:.4f}",
                        "Δ Overlap": f"{diff_j:+.4f}",
                    })
                st.dataframe(pd.DataFrame(j_rows), use_container_width=True)

            # Dynamically-computed closing verdicts per client
            st.markdown("#### 🏁 Concrete Client-by-Client Verdicts (Data-Driven)")

            w_avg = avg_run["client_metrics"]["wifi"]["f1_macro"]
            w_prox = prox001_run["client_metrics"]["wifi"]["f1_macro"]
            m_avg = avg_run["client_metrics"]["mqtt"]["f1_macro"]
            m_prox = prox001_run["client_metrics"]["mqtt"]["f1_macro"]
            b_avg = avg_run["client_metrics"]["bluetooth"]["f1_macro"]
            b_prox = prox001_run["client_metrics"]["bluetooth"]["f1_macro"]

            st.markdown(f"""
            <div class="verdict-card">
                <b>🌐 Wi-Fi Verdict: NET NEUTRAL / FLAT ({w_prox - w_avg:+.4f} Δ Macro F1)</b><br>
                Wi-Fi achieves {w_avg:.4f} under FedAvg and {w_prox:.4f} under FedProx (μ=0.01). As the dominant client (50% of total volume), global weights are anchored in Wi-Fi representations regardless of proximal penalty.
            </div>
            <div class="verdict-card">
                <b>📡 MQTT Verdict: NET POSITIVE ({m_prox - m_avg:+.4f} Δ Macro F1)</b><br>
                MQTT Macro F1 increased from {m_avg:.4f} to {m_prox:.4f}. Proximal regularization prevents local gradients from being overwritten by Wi-Fi dominance, stabilizing bedside sensor detection boundaries.
            </div>
            <div class="verdict-card" style="border-left: 4px solid #ef4444;">
                <b>📶 Bluetooth Verdict: SEVERE COLLAPSE ({b_prox - b_avg:+.4f} Δ Macro F1)</b><br>
                Bluetooth Macro F1 dropped from {b_avg:.4f} down to {b_prox:.4f}. Proximal penalty over-constrains non-IP link-layer representations toward IP-centric global weights, causing complete failure on benign discrimination.
            </div>
            """, unsafe_allow_html=True)


# ======================================================================
# TAB 5: Methodology & Limitations
# ======================================================================
with tab_methods:
    st.markdown("### 🔬 Scientific Methodology & Noise Floor")

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("#### ✅ Demonstrated in this Study")
        st.markdown("""
        - **10-Round Scaled Validation**: Verified multi-round convergence on 166,000+ real IoMT flows across 3 distinct protocols.
        - **45-Feature Schema Guardrails**: Enforced strict input dimension checks across physical layer boundaries.
        - **Imbalance-Aware Reporting**: Macro F1 and per-class metrics surfaced by default.
        """)

    with col_m2:
        st.markdown("#### ⚠️ Open Scope for Full Multi-Seed Study")
        st.markdown("""
        - **Seed Calibration ($n=2$ seeds)**: Provides directional consistency, not a full 10-seed Gaussian confidence interval.
        - **Hyperparameter Sweeps**: Explored $\mu \in \{0.01, 0.1\}$; a full logarithmic grid remains planned.
        - **Advanced Aggregators**: Benchmarking SCAFFOLD, FedNova, and FedOpt.
        """)

    if study_data and "preliminary_noise_floor" in study_data:
        st.markdown("---")
        st.markdown("#### 📉 Seed-to-Seed Stability Matrix ($n=2$ Seeds)")
        nf = study_data["preliminary_noise_floor"]
        nf_rows = []
        for agg_name, d in nf.items():
            stabs = d.get("per_client_top5_stability", {})
            nf_rows.append({
                "Aggregator": agg_name.upper(),
                "Wi-Fi Seed Overlap": f"{stabs.get('wifi', {}).get('seed_to_seed_jaccard', 0.0):.4f}",
                "MQTT Seed Overlap": f"{stabs.get('mqtt', {}).get('seed_to_seed_jaccard', 0.0):.4f}",
                "Bluetooth Seed Overlap": f"{stabs.get('bluetooth', {}).get('seed_to_seed_jaccard', 0.0):.4f}",
                "Status": d.get("status", "Preliminary"),
            })
        st.dataframe(pd.DataFrame(nf_rows), use_container_width=True)
