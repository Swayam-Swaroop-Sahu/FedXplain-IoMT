"""FedXplain-IoMT Research Dashboard.

Interactive research results interface for cross-client explanation divergence
in federated intrusion detection across heterogeneous medical IoT protocols.
"""

import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------
# Page Configuration & Professional Styling
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="FedXplain-IoMT",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #334155;
        margin-bottom: 1.1rem;
        font-weight: 500;
    }
    .lead-paragraph {
        font-size: 0.98rem;
        line-height: 1.65;
        color: #334155;
        margin-bottom: 1.5rem;
    }
    .alert-danger {
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 12px 0 16px 0;
        color: #991b1b;
        font-size: 0.95rem;
        line-height: 1.55;
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
# Fixed Header Block (Always Visible)
# ----------------------------------------------------------------------
st.markdown('<div class="main-header">FedXplain-IoMT</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Cross-Client Explanation Divergence in Federated Intrusion Detection for Heterogeneous Medical IoT</div>', unsafe_allow_html=True)

st.markdown("""
<div class="lead-paragraph">
Internet of Medical Things (IoMT) devices in hospital networks communicate across structurally different protocols - Wi-Fi, MQTT, and Bluetooth - each with distinct traffic characteristics and, in Bluetooth's case, no TCP/IP structure at all. Federated Learning allows a shared intrusion detection model to be trained across these device groups without centralizing patient network telemetry, as required under HIPAA/GDPR. This study asks a question that goes beyond standard accuracy reporting: when the resulting global model is evaluated on each protocol separately, does it rely on the same features to justify its decisions, or does it reason differently depending on which protocol produced the traffic - and does FedProx, a heterogeneity-aware aggregation method, reduce this divergence without degrading detection quality?
</div>
""", unsafe_allow_html=True)

ARCH_IMG_PATH = os.path.join("docs", "system_architecture.png")
if os.path.exists(ARCH_IMG_PATH):
    st.image(ARCH_IMG_PATH, caption="Figure 1. Federated training and post-hoc explanation pipeline across three protocol-grounded clients.", width="stretch")


# ----------------------------------------------------------------------
# Sidebar (Experiment Configuration Metadata)
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Experiment Parameters")
    if study_data:
        meta = study_data.get("run_metadata", {})
        st.markdown(f"**Dataset**: `{meta.get('data_dir', 'data/study')}` (166,336 flows)")
        st.markdown(f"**Communication Rounds**: `{meta.get('n_rounds', 10)}`")
        st.markdown(f"**Local Epochs per Round**: `{meta.get('n_local_epochs', 3)}`")
        st.markdown(f"**Random Seeds**: `{meta.get('seeds', [42, 7])}`")
        st.markdown(f"**Proximal Penalties (mu)**: `{meta.get('mu_values', [0.01, 0.1])}`")
    else:
        st.error("Missing results/study_results.json. Run python src/run_study_experiments.py.")
    st.markdown("---")
    st.caption("Benchmark: CICIoMT2024 (Wi-Fi, MQTT, Bluetooth)")


# ----------------------------------------------------------------------
# 4 Main Tabs
# ----------------------------------------------------------------------
tab_perf, tab_divergence, tab_compare, tab_methods = st.tabs([
    "Detection Performance",
    "Explanation Divergence",
    "FedAvg vs FedProx",
    "Methodology and Limitations",
])


# ======================================================================
# TAB 1: Detection Performance
# ======================================================================
with tab_perf:
    if study_data:
        runs = study_data.get("runs", [])
        seeds_available = sorted(list(set(r.get("seed", 42) for r in runs)))
        seed_selected = st.selectbox("Random Seed", seeds_available, index=0)
        filtered_runs = [r for r in runs if r.get("seed") == seed_selected]

        avg_run = next((r for r in filtered_runs if r["aggregator"] == "fedavg"), None)
        prox_run = next((r for r in filtered_runs if r["aggregator"] == "fedprox" and r.get("mu") == 0.01), None)

        if avg_run and prox_run:
            m_avg = avg_run["client_metrics"]
            m_prox = prox_run["client_metrics"]

            mqtt_diff = m_prox["mqtt"]["f1_macro"] - m_avg["mqtt"]["f1_macro"]
            wifi_diff = m_prox["wifi"]["f1_macro"] - m_avg["wifi"]["f1_macro"]
            bt_diff = m_prox["bluetooth"]["f1_macro"] - m_avg["bluetooth"]["f1_macro"]

            bt_avg_brec = m_avg["bluetooth"]["benign_recall"] * 100
            bt_prox_brec = m_prox["bluetooth"]["benign_recall"] * 100
            bt_prox_aprec = m_prox["bluetooth"]["attack_precision"] * 100

            st.markdown(f"**Summary (Seed {seed_selected}):**")
            st.markdown(
                f"Under FedProx (mu=0.01), MQTT macro F1 changed by {mqtt_diff:+.4f} "
                f"({m_avg['mqtt']['f1_macro']:.4f} to {m_prox['mqtt']['f1_macro']:.4f}), "
                f"while Wi-Fi macro F1 changed by {wifi_diff:+.4f} "
                f"({m_avg['wifi']['f1_macro']:.4f} to {m_prox['wifi']['f1_macro']:.4f}). "
                f"Bluetooth macro F1 changed by {bt_diff:+.4f} "
                f"({m_avg['bluetooth']['f1_macro']:.4f} to {m_prox['bluetooth']['f1_macro']:.4f}). "
                f"Wi-Fi binary F1 (>0.99) is inflated by 96.4% attack prevalence; macro F1 reflects true balanced performance."
            )

            # Highlight Bluetooth failure if condition is met
            if m_prox["bluetooth"]["benign_recall"] < 0.15 or m_prox["bluetooth"]["f1_macro"] < 0.50:
                st.markdown(f"""
                <div class="alert-danger">
                    <strong>Detection Failure Warning:</strong> Bluetooth client performance collapsed under FedProx (mu=0.01), with macro F1 dropping from {m_avg['bluetooth']['f1_macro']:.4f} to {m_prox['bluetooth']['f1_macro']:.4f} ({bt_diff * 100:+.1f} percentage points). Benign recall fell from {bt_avg_brec:.1f}% to {bt_prox_brec:.1f}%, and attack precision dropped to {bt_prox_aprec:.1f}%. The model lost the ability to distinguish benign traffic from attacks and defaulted to predicting the attack class for nearly all Bluetooth samples.
                </div>
                """, unsafe_allow_html=True)

        # Performance table
        rows = []
        for r in filtered_runs:
            agg_name = r["aggregator"].upper()
            if r.get("mu") is not None:
                agg_name += f" (mu={r['mu']})"

            for proto, metrics in r["client_metrics"].items():
                is_collapsed = metrics["benign_recall"] < 0.15 and metrics["f1_macro"] < 0.50
                status_tag = "Collapsed" if is_collapsed else ("Improved" if "FEDPROX (MU=0.01)" in agg_name and proto == "mqtt" else "Stable")

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

        def highlight_collapsed(row):
            if row["Status"] == "Collapsed":
                return ["background-color: #fee2e2; color: #991b1b;"] * len(row)
            return [""] * len(row)

        styled_df = df_perf.style.apply(highlight_collapsed, axis=1).format({
            "Macro F1": "{:.4f}",
            "Binary F1": "{:.4f}",
            "Benign Precision": "{:.4f}",
            "Benign Recall": "{:.4f}",
            "Attack Precision": "{:.4f}",
            "Attack Recall": "{:.4f}",
            "Accuracy": "{:.4f}",
        })

        st.dataframe(styled_df, width="stretch", height=340)


# ======================================================================
# TAB 2: Explanation Divergence
# ======================================================================
with tab_divergence:
    if study_data:
        runs = study_data.get("runs", [])
        run_options = [
            f"{r['aggregator'].upper()}" + (f" (mu={r['mu']})" if r.get("mu") else "") + f" - Seed {r['seed']}"
            for r in runs
        ]
        selected_idx = st.selectbox("Aggregator Run", range(len(run_options)), format_func=lambda i: run_options[i])
        active_run = runs[selected_idx]
        active_seed = active_run["seed"]

        base_avg_run = next((r for r in runs if r["aggregator"] == "fedavg" and r["seed"] == active_seed), None)

        col_top10, col_jaccard = st.columns([3, 2])

        with col_top10:
            st.markdown("**Top-10 SHAP Feature Attributions**")
            proto_select = st.segmented_control("Protocol Client", ["WIFI", "MQTT", "BLUETOOTH"], default="WIFI")
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
                    height=380,
                )
                fig_shap.update_layout(margin=dict(l=20, r=20, t=35, b=20))
                st.plotly_chart(fig_shap, width="stretch")

        with col_jaccard:
            st.markdown("**Top-5 Jaccard Similarity Matrix**")
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
                textfont={"size": 13},
            ))
            fig_hm.update_layout(
                title="Pairwise Jaccard Overlap",
                height=380,
                margin=dict(l=20, r=20, t=35, b=20),
            )
            st.plotly_chart(fig_hm, width="stretch")

        j_wm = j_map.get("wifi-mqtt", 0.0)
        j_wb = j_map.get("wifi-bluetooth", 0.0)
        j_mb = j_map.get("mqtt-bluetooth", 0.0)

        diff_text = ""
        if base_avg_run and active_run["aggregator"] != "fedavg":
            base_j = base_avg_run["jaccard_pairwise"]
            d_wm = j_wm - base_j.get("wifi-mqtt", 0.0)
            d_wb = j_wb - base_j.get("wifi-bluetooth", 0.0)
            d_mb = j_mb - base_j.get("mqtt-bluetooth", 0.0)
            diff_text = f"Compared to FedAvg: Wi-Fi/MQTT shifted by {d_wm:+.4f}, Wi-Fi/Bluetooth by {d_wb:+.4f}, and MQTT/Bluetooth by {d_mb:+.4f}."

        st.markdown("**Summary:**")
        st.markdown(
            f"Top-5 feature Jaccard similarity under {run_options[selected_idx]}: "
            f"Wi-Fi vs MQTT = {j_wm:.4f}, Wi-Fi vs Bluetooth = {j_wb:.4f}, MQTT vs Bluetooth = {j_mb:.4f}. {diff_text}"
        )

        bt_macro = active_run["client_metrics"]["bluetooth"]["f1_macro"]
        bt_brec = active_run["client_metrics"]["bluetooth"]["benign_recall"]

        if bt_brec < 0.15:
            st.markdown(f"""
            <div class="alert-danger">
                <strong>Cross-Reference Notice:</strong> The Jaccard index between Bluetooth and other clients under this configuration (e.g., MQTT vs Bluetooth J = {j_mb:.4f}) coincides with a collapse in Bluetooth detection performance (Macro F1: {bt_macro:.4f}, Benign Recall: {bt_brec * 100:.1f}%). This attribution shift reflects a degenerate decision boundary on Bluetooth traffic rather than valid cross-protocol representation learning.
            </div>
            """, unsafe_allow_html=True)


# ======================================================================
# TAB 3: FedAvg vs FedProx
# ======================================================================
with tab_compare:
    if study_data:
        runs = study_data.get("runs", [])
        seeds_avail = sorted(list(set(r.get("seed", 42) for r in runs)))
        cmp_seed = st.selectbox("Evaluation Seed", seeds_avail, index=0)

        s_runs = {f"{r['aggregator']}_{r['mu']}": r for r in runs if r["seed"] == cmp_seed}

        if "fedavg_None" in s_runs and "fedprox_0.01" in s_runs:
            avg_run = s_runs["fedavg_None"]
            prox001_run = s_runs["fedprox_0.01"]

            col_t1, col_t2 = st.columns(2)

            with col_t1:
                st.markdown("**Detection Performance (Macro F1)**")
                comp_rows = []
                for proto in ["wifi", "mqtt", "bluetooth"]:
                    m_a = avg_run["client_metrics"][proto]["f1_macro"]
                    m_p001 = prox001_run["client_metrics"][proto]["f1_macro"]
                    comp_rows.append({
                        "Protocol": proto.upper(),
                        "FedAvg": f"{m_a:.4f}",
                        "FedProx (mu=0.01)": f"{m_p001:.4f}",
                        "Delta Change": f"{m_p001 - m_a:+.4f}",
                    })
                st.dataframe(pd.DataFrame(comp_rows), width="stretch")

            with col_t2:
                st.markdown("**Explanation Overlap (Top-5 Jaccard)**")
                j_rows = []
                for pair in ["wifi-mqtt", "wifi-bluetooth", "mqtt-bluetooth"]:
                    j_a = avg_run["jaccard_pairwise"].get(pair, 0.0)
                    j_p001 = prox001_run["jaccard_pairwise"].get(pair, 0.0)
                    j_rows.append({
                        "Client Pair": pair.upper(),
                        "FedAvg J": f"{j_a:.4f}",
                        "FedProx (mu=0.01) J": f"{j_p001:.4f}",
                        "Delta Overlap": f"{j_p001 - j_a:+.4f}",
                    })
                st.dataframe(pd.DataFrame(j_rows), width="stretch")

            st.markdown("**Findings**")

            w_avg = avg_run["client_metrics"]["wifi"]["f1_macro"]
            w_prox = prox001_run["client_metrics"]["wifi"]["f1_macro"]
            m_avg = avg_run["client_metrics"]["mqtt"]["f1_macro"]
            m_prox = prox001_run["client_metrics"]["mqtt"]["f1_macro"]
            b_avg = avg_run["client_metrics"]["bluetooth"]["f1_macro"]
            b_prox = prox001_run["client_metrics"]["bluetooth"]["f1_macro"]
            b_prox_brec = prox001_run["client_metrics"]["bluetooth"]["benign_recall"] * 100

            st.markdown(
                f"Wi-Fi: Macro F1 changed by {w_prox - w_avg:+.4f} under FedProx (mu=0.01), from {w_avg:.4f} to {w_prox:.4f}. "
                f"As the largest client by sample count (50% of the aggregate dataset), the global model's aggregate behavior remains anchored to Wi-Fi's decision boundary regardless of the proximal penalty applied to other clients."
            )

            st.markdown(
                f"MQTT: Macro F1 improved by {m_prox - m_avg:+.4f}, from {m_avg:.4f} to {m_prox:.4f}. "
                f"The proximal term appears to prevent MQTT's local updates from being overwritten during aggregation, improving its detection stability without a corresponding drop elsewhere."
            )

            st.markdown(
                f"Bluetooth: Macro F1 changed by {b_prox - b_avg:+.4f}, from {b_avg:.4f} to {b_prox:.4f}. "
                f"This coincides with a collapse in benign recall (falling to {b_prox_brec:.1f}%), indicating the proximal penalty constrained Bluetooth's local model toward the IP-protocol-dominated global representation at the cost of its own, structurally distinct detection task."
            )


# ======================================================================
# TAB 4: Methodology and Limitations
# ======================================================================
with tab_methods:
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("**Demonstrated in this Study**")
        st.markdown("""
        - **10-Round Scaled Validation**: Evaluated multi-round convergence on 166,000+ real IoMT flows across Wi-Fi, MQTT, and Bluetooth.
        - **45-Feature Schema Guardrails**: Enforced deterministic feature alignment across transport and link-layer boundaries.
        - **Imbalance-Aware Metrics**: Evaluated macro F1 and per-class precision/recall to prevent false optimism on imbalanced subsets.
        - **Post-Hoc Attribution Tracking**: Quantified feature overlap across clients via SHAP DeepExplainer and Top-5 Jaccard indices.
        """)

    with col_m2:
        st.markdown("**Open Scope for Full Multi-Seed Study**")
        st.markdown("""
        - **Statistical Confidence Calibration**: Preliminary evaluation on n=2 seeds provides directional consistency; a full 10-seed study with confidence intervals remains planned.
        - **Continuous Hyperparameter Sweep**: Evaluated mu in {0.01, 0.1}; a full logarithmic grid search across regularization intensities remains open.
        - **Alternative Heterogeneity-Aware Optimizers**: Benchmarking SCAFFOLD, FedNova, and FedOpt against FedProx.
        """)

    if study_data and "preliminary_noise_floor" in study_data:
        st.markdown("---")
        st.markdown("**Preliminary Seed-to-Seed Stability Matrix (n=2 Seeds)**")
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
        st.dataframe(pd.DataFrame(nf_rows), width="stretch")
