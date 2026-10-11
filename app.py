import json
import os
import sys
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, List, Tuple, Optional, Any

# ---- Constants and configuration ----
try:
    from src.config import CLUSTER_FL_STATUS
except Exception:
    CLUSTER_FL_STATUS = "in progress - port pending"

# Color palette (from figures and spec)
NAVY = "#0F2A43"
TEAL = "#2A9D8F"
BLUE = "#2A78D6"
AMBER = "#C98500"
SURFACE = "#F5F5F2"
HIGHLIGHT = "#E76F51"
CARD_BORDER = "#E5E5DE"
MUTED = "#5B6B7A"
GREY = "#8896A6"
GRID = "#D8D8D2"

# Paths
RESULTS_PATH = os.path.join("results", "study_results.json")
ARCH_IMG_PATH = os.path.join("docs", "system_architecture.png")
FIG4_PATH = os.path.join("figures", "fig4_tradeoff_scatter.png")

# Protocol labels
PROTOS = ["wifi", "mqtt", "bluetooth"]
PROTO_LABEL = {"wifi": "Wi‑Fi", "mqtt": "MQTT", "bluetooth": "Bluetooth"}
PAIRS = ["wifi-mqtt", "wifi-bluetooth", "mqtt-bluetooth"]
PAIR_LABEL = {"wifi-mqtt": "Wi‑Fi–MQTT", "wifi-bluetooth": "Wi‑Fi–Bluetooth", "mqtt-bluetooth": "MQTT–Bluetooth"}

# ---- Helper functions ----
def fmt(value: Optional[float], nd: int = 4) -> str:
    """Format float or None to string with nd decimals, or '—' for None."""
    if value is None:
        return "—"
    return f"{value:.{nd}f}"

def mean_spread(vals: List[Optional[float]]) -> Tuple[float, float]:
    """Return (mean, spread) of a list of floats, ignoring None."""
    clean = [v for v in vals if v is not None]
    if not clean:
        return 0.0, 0.0
    mean_val = sum(clean) / len(clean)
    spread = max(clean) - min(clean) if len(clean) > 1 else 0.0
    return mean_val, spread

def fmt_pm(vals: List[Optional[float]]) -> str:
    """Format mean ± spread from list of values."""
    mean_val, spread = mean_spread(vals)
    return f"{mean_val:.4f} ± {spread:.4f}"

def find_run(runs: List[Dict], aggregator: str, seed: int, mu: Optional[float] = None) -> Optional[Dict]:
    """Find a run entry matching aggregator, seed, and optional mu."""
    for r in runs:
        if r.get("aggregator") == aggregator and r.get("seed") == seed:
            if mu is None:
                return r
            if abs(r.get("mu", 0.0) - mu) < 1e-9:
                return r
    return None

def fedprox_mus(runs: List[Dict]) -> List[float]:
    """Return sorted unique mu values from FedProx runs."""
    mus = set()
    for r in runs:
        if r.get("aggregator") == "fedprox":
            mu_val = r.get("mu")
            if mu_val is not None:
                mus.add(mu_val)
    return sorted(list(mus))

def avg_macro_f1(run: Dict) -> float:
    """Compute average macro F1 across three clients from a run."""
    clients = ["wifi", "mqtt", "bluetooth"]
    f1s = []
    for proto in clients:
        cm = run.get("client_metrics", {}).get(proto, {})
        f1 = cm.get("f1_macro")
        if f1 is not None:
            f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0

def client_metric(run: Dict, proto: str, key: str) -> Optional[float]:
    """Safely extract a metric from client_metrics."""
    cm = run.get("client_metrics", {}).get(proto, {})
    return cm.get(key)

def local_entry(data: Dict, seed: int) -> Optional[Dict]:
    """Find local-only run for a given seed."""
    for r in data.get("local_only", []):
        if r.get("seed") == seed:
            return r
    return None

def seeds_of(data: Dict) -> List[int]:
    """Return sorted unique seeds from local_only runs."""
    seeds = set()
    for r in data.get("local_only", []):
        s = r.get("seed")
        if s is not None:
            seeds.add(s)
    return sorted(list(seeds))

def run_key(run: Dict) -> str:
    """Generate a string key for a run (e.g., 'fedprox_mu0.001')."""
    agg = run.get("aggregator", "")
    if agg == "fedprox":
        mu = run.get("mu", 0.0)
        return f"fedprox_mu{mu}"
    return agg

def method_label(key: str) -> str:
    """Map internal key to display label."""
    if key == "fedavg":
        return "FedAvg"
    if key.startswith("fedprox_mu"):
        mu_str = key.removeprefix("fedprox_mu")
        return f"FedProx (μ={mu_str})"
    if key == "fedper":
        return "FedPer"
    if key == "local_only":
        return "Local-only"
    if key == "cluster_fl":
        return "Cluster-FL (pending)"
    return key

def method_run(data: Dict, key: str, seed: int) -> Optional[Dict]:
    """Resolve a method key to a run dict for a given seed."""
    if key == "local_only":
        return local_entry(data, seed)
    if key.startswith("fedprox_mu"):
        mu_val = float(key.removeprefix("fedprox_mu"))
        return find_run(data.get("runs", []), "fedprox", seed, mu_val)
    # fedavg, fedper, cluster_fl
    return find_run(data.get("runs", []), key, seed)

def kpi_card(label: str, value: str, caption: str) -> str:
    """Return HTML string for a KPI card."""
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-caption">{caption}</div>
    </div>
    """

def section_header(text: str) -> str:
    """Return HTML string for a section header."""
    return f'<div class="section-header">{text}</div>'

# ---- Streamlit page config and CSS ----
st.set_page_config(
    page_title="FedXplain-IoMT Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown(
    f"""
    <style>
        /* Hide Streamlit chrome */
        #MainMenu {{visibility: hidden;}}
        header[data-testid="stHeader"] {{display: none;}}
        footer {{visibility: hidden;}}
        .stDeployButton {{display: none;}}

        /* Base styling */
        .stApp {{
            background-color: {SURFACE};
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            color: {NAVY};
        }}

        /* Card containers */
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background-color: #FFFFFF;
            border: 1px solid {CARD_BORDER};
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
        }}

        /* Headers */
        .main-header {{
            font-size: 2rem;
            font-weight: 800;
            color: {NAVY};
            margin-bottom: 0.5rem;
        }}
        .sub-header {{
            font-size: 1.2rem;
            font-weight: 600;
            color: {TEAL};
            margin-bottom: 1.5rem;
        }}
        .section-header {{
            font-size: 1.1rem;
            font-weight: 700;
            color: {NAVY};
            border-bottom: 2px solid {TEAL};
            padding-bottom: 0.3rem;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }}

        /* KPI cards */
        .kpi-card {{
            background-color: #FFFFFF;
            border: 1px solid {CARD_BORDER};
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }}
        .kpi-label {{
            font-size: 0.64rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: {MUTED};
            margin-bottom: 0.3rem;
        }}
        .kpi-value {{
            font-size: 1.9rem;
            font-weight: 800;
            color: {NAVY};
            line-height: 1.2;
            margin-bottom: 0.2rem;
        }}
        .kpi-caption {{
            font-size: 0.85rem;
            font-weight: 500;
            color: {GREY};
        }}

        /* Alerts */
        .alert-warn {{
            background-color: #FCEDE5;
            border-left: 4px solid {HIGHLIGHT};
            color: #7C2D12;
            padding: 0.8rem 1rem;
            border-radius: 0 4px 4px 0;
            margin: 1rem 0;
        }}

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: #F0F2F6;
            border-radius: 6px 6px 0 0;
            padding: 8px 16px;
            font-weight: 600;
            color: {MUTED};
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {TEAL} !important;
            color: white !important;
        }}

        /* Misc */
        .thesis {{
            font-size: 1.1rem;
            color: {TEAL};
            border-left: 4px solid {TEAL};
            padding-left: 1rem;
            margin: 1.5rem 0;
            font-style: italic;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Load data ----
@st.cache_data
def load_results() -> Dict:
    """Load and return the study results JSON."""
    if not os.path.exists(RESULTS_PATH):
        st.error(f"Results file not found: {RESULTS_PATH}")
        st.stop()
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

data = load_results()

# ---- Sidebar ----
with st.sidebar:
    st.markdown("### 📊 Run Metadata")
    if data.get("run_metadata"):
        for key, value in data["run_metadata"].items():
            st.text(f"{key}: {value}")
    st.markdown("---")
    st.markdown(
        """
        <small>
        **Benchmark**: CICIoMT2024<br>
        **Protocols**: Wi‑Fi, MQTT, Bluetooth<br>
        **Seeds**: 42, 7 (directional only)<br>
        **Read‑only, runs fully offline**
        </small>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown(f"**Cluster‑FL status**: {CLUSTER_FL_STATUS}")

# ---- Header ----
st.markdown('<div class="main-header">FedXplain-IoMT</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Federated Learning & Explainable AI for Heterogeneous Medical IoT</div>', unsafe_allow_html=True)

# ---- Tabs ----
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Summary",
    "Detection",
    "Explanation Divergence",
    "Methods Compared",
    "Methodology & Limitations"
])

# ===== TAB 0: Executive Summary =====
with tab0:
    st.markdown('<div class="thesis">One shared detector cannot serve three structurally different hospital protocols: on this benchmark the shared model collapses on Bluetooth, no proximal strength rescues it, and only personalizing the final layer — or not sharing at all — does.</div>', unsafe_allow_html=True)

    # KPI cards row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    # KPI 1: Best shared-model avg macro F1
    shared_methods = ["fedavg"] + [f"fedprox_mu{m}" for m in fedprox_mus(data.get("runs", []))]
    best_shared_mean = -1.0
    best_shared_label = ""
    for method in shared_methods:
        seed_vals = []
        for seed in seeds_of(data):
            run = method_run(data, method, seed)
            if run:
                seed_vals.append(avg_macro_f1(run))
        if seed_vals:
            mean_val, _ = mean_spread(seed_vals)
            if mean_val > best_shared_mean:
                best_shared_mean = mean_val
                best_shared_label = method
    kpi1_value = fmt(best_shared_mean)
    kpi1_caption = f"Best shared-model: {method_label(best_shared_label)}"
    with kpi1:
        st.markdown(kpi_card("BEST SHARED-MODEL F1", kpi1_value, kpi1_caption), unsafe_allow_html=True)

    # KPI 2: Local-only avg macro F1 + delta vs best shared
    local_seed_vals = []
    for seed in seeds_of(data):
        run = local_entry(data, seed)
        if run:
            local_seed_vals.append(avg_macro_f1(run))
    local_mean, local_spread = mean_spread(local_seed_vals)
    kpi2_value = f"{fmt(local_mean)} ({local_mean - best_shared_mean:+.4f} Δ)"
    kpi2_caption = f"Local-only (2-seed mean ± {local_spread:.4f})"
    with kpi2:
        st.markdown(kpi_card("LOCAL-ONLY F1", kpi2_value, kpi2_caption), unsafe_allow_html=True)

    # KPI 3: Worst FedProx BT benign recall vs local-only
    worst_bt_recall = 1.0
    worst_mu = None
    worst_seed = None
    for mu in fedprox_mus(data.get("runs", [])):
        for seed in seeds_of(data):
            run = method_run(data, f"fedprox_mu{mu}", seed)
            if run:
                bt_recall = client_metric(run, "bluetooth", "benign_recall")
                if bt_recall is not None and bt_recall < worst_bt_recall:
                    worst_bt_recall = bt_recall
                    worst_mu = mu
                    worst_seed = seed
    local_bt_recalls = []
    for seed in seeds_of(data):
        run = local_entry(data, seed)
        if run:
            local_bt_recalls.append(client_metric(run, "bluetooth", "benign_recall"))
    local_bt_min = min([v for v in local_bt_recalls if v is not None], default=1.0)
    kpi3_value = fmt(worst_bt_recall)
    kpi3_caption = f"Worst FedProx (μ={worst_mu}, seed={worst_seed}) vs local-only min {fmt(local_bt_min)}"
    with kpi3:
        st.markdown(kpi_card("BT BENIGN RECALL (WORST)", kpi3_value, kpi3_caption), unsafe_allow_html=True)

    # KPI 4: FedAvg Wi‑Fi–Bluetooth Spearman (two‑seed mean)
    spear_seeds = []
    for seed in seeds_of(data):
        run = method_run(data, "fedavg", seed)
        if run:
            div_full = run.get("divergence_full", {})
            spear = div_full.get("wifi-bluetooth", {}).get("spearman")
            if spear is not None:
                spear_seeds.append(spear)
    kpi4_value = fmt_pm(spear_seeds)
    kpi4_caption = "FedAvg Wi‑Fi–Bluetooth Spearman ρ (two‑seed mean ± spread)"
    with kpi4:
        st.markdown(kpi_card("WI‑FI–BT SPEARMAN", kpi4_value, kpi4_caption), unsafe_allow_html=True)

    # Figure 4
    st.markdown("---")
    if os.path.exists(FIG4_PATH):
        st.image(FIG4_PATH, width="stretch")
        st.caption("Figure 4: Detection quality vs cross‑client explanation consistency (two‑seed means with seed‑spread error bars).")
    else:
        st.warning(f"Figure not found: {FIG4_PATH}")

    # Two‑column findings
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">What we found</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            - **Bluetooth collapse**: Shared models drop BT benign recall to as low as {fmt(worst_bt_recall)} (μ={worst_mu}), while local‑only stays at {fmt(local_bt_min)}.
            - **No safe μ**: Even μ=0.001 worsens BT benign recall vs FedAvg ({fmt(client_metric(method_run(data, 'fedavg', seeds_of(data)[0]), 'bluetooth', 'benign_recall'))} → {fmt(client_metric(method_run(data, f'fedprox_mu{worst_mu}', seeds_of(data)[0]), 'bluetooth', 'benign_recall'))}).
            - **FedPer’s trade‑off**: Recovers ~{(local_bt_min - worst_bt_recall) / (local_bt_min - client_metric(method_run(data, 'fedavg', seeds_of(data)[0]), 'bluetooth', 'benign_recall')) * 100:.0f}% of the BT benign‑recall gap while retaining ~{local_mean / best_shared_mean * 100:.0f}% of FedAvg’s detection quality.
            - **Explanation consistency**: FedPer’s cross‑client similarity stays at FedAvg level (see Tab 2).
            """
        )
    with col2:
        st.markdown('<div class="section-header">What we didn’t prove yet</div>', unsafe_allow_html=True)
        st.markdown(
            """
            - **Statistical significance**: n=2 seeds provide directional trends only; no p‑values or confidence intervals.
            - **Cluster‑FL**: Awaiting teammate’s port (see Tab 4).
            - **Full noise floor**: Preliminary (n=2) only; more seeds needed for robust variance estimates.
            - **Other aggregators**: SCAFFOLD, FedNova, etc. remain for future work.
            """
        )

# ===== TAB 1: Detection =====
with tab1:
    st.markdown('<div class="section-header">Unified detection table (macro F1)</div>', unsafe_allow_html=True)

    # Build unified table
    def unified_detection_rows(data_dict: Dict) -> List[Dict]:
        rows = []
        seeds = seeds_of(data_dict)
        # Header will be built later; we build data rows
        methods = ["fedavg"] + [f"fedprox_mu{m}" for m in fedprox_mus(data_dict.get("runs", []))] + ["fedper", "local_only"]
        # Add centralized reference (pooled)
        methods.append("centralized")
        for method_key in methods:
            row = {"Method": method_label(method_key)}
            # Per‑client per‑seed columns
            for seed in seeds:
                for proto in PROTOS:
                    col_name = f"{PROTO_LABEL[proto]} · {seed}"
                    if method_key == "centralized":
                        # Centralized: use per‑seed centralized_by_seed
                        cents = data_dict.get("centralized_by_seed", {})
                        cents_run = cents.get(str(seed))
                        if cents_run:
                            val = cents_run.get(f"{proto}_f1_macro")
                        else:
                            val = None
                    else:
                        run = method_run(data_dict, method_key, seed)
                        if run:
                            val = client_metric(run, proto, "f1_macro")
                        else:
                            val = None
                    row[col_name] = val
            # Avg columns per seed
            for seed in seeds:
                col_name = f"Avg · {seed}"
                if method_key == "centralized":
                    cents = data_dict.get("centralized_by_seed", {})
                    cents_run = cents.get(str(seed))
                    if cents_run:
                        vals = [cents_run.get(f"{p}_f1_macro") for p in PROTOS if cents_run.get(f"{p}_f1_macro") is not None]
                        val = sum(vals) / len(vals) if vals else None
                    else:
                        val = None
                else:
                    run = method_run(data_dict, method_key, seed)
                    if run:
                        val = avg_macro_f1(run)
                    else:
                        val = None
                row[col_name] = val
            # Overall average mean ± spread
            avg_vals = []
            for seed in seeds:
                run = method_run(data_dict, method_key, seed)
                if run:
                    avg_vals.append(avg_macro_f1(run))
            mean_val, spread_val = mean_spread(avg_vals)
            row["Avg mean ± spread"] = fmt_pm(avg_vals) if avg_vals else "—"
            rows.append(row)
        return rows

    df_unified = pd.DataFrame(unified_detection_rows(data))
    # Reorder columns for readability: Method, then Wi‑Fi 42, Wi‑Fi 7, MQTT 42, MQTT 7, BT 42, BT 7, then Avg·42, Avg·7, Avg mean ± spread
    cols = ["Method"]
    for seed in seeds_of(data):
        for proto in PROTOS:
            cols.append(f"{PROTO_LABEL[proto]} · {seed}")
    for seed in seeds_of(data):
        cols.append(f"Avg · {seed}")
    cols.append("Avg mean ± spread")
    df_unified = df_unified[cols]
    st.dataframe(
        df_unified.style.format(
            {c: lambda x: fmt(x) if isinstance(x, (float, int)) else x for c in df_unified.columns if c != "Method"}
        ),
        height=400,
        use_container_width=True
    )

    st.markdown("---")
    st.markdown('<div class="section-header">Bluetooth benign‑recall table</div>', unsafe_allow_html=True)
    # Build BT benign recall table with styling
    bt_rows = []
    methods = ["fedavg"] + [f"fedprox_mu{m}" for m in fedprox_mus(data.get("runs", []))] + ["fedper", "local_only", "centralized"]
    for method_key in methods:
        row = {"Method": method_label(method_key)}
        for seed in seeds_of(data):
            col_name = f"{seed}"
            if method_key == "centralized":
                cents = data.get("centralized_by_seed", {})
                cents_run = cents.get(str(seed))
                if cents_run:
                    val = cents_run.get("bluetooth_benign_recall", cents_run.get("benign_recall"))
                else:
                    val = None
            else:
                run = method_run(data, method_key, seed)
                if run:
                    val = client_metric(run, "bluetooth", "benign_recall")
                else:
                    val = None
            row[col_name] = val
        bt_rows.append(row)
    df_bt = pd.DataFrame(bt_rows)
    # Styling function to highlight cells < 0.5
    def highlight_bad(val):
        try:
            if val is None or (isinstance(val, str) and val == "—"):
                return ""
            v = float(val)
            if v < 0.5:
                return f"background-color:#FCEDE5; color:{HIGHLIGHT}; font-weight:700;"
            return ""
        except:
            return ""
    # Apply styling to seed columns only
    seed_cols = [str(s) for s in seeds_of(data)]
    styled_bt = df_bt.style.format(
        {c: lambda x: fmt(x) if isinstance(x, (float, int)) else x for c in df_bt.columns if c != "Method"}
    ).map(highlight_bad, subset=seed_cols)
    st.dataframe(styled_bt, height=300, use_container_width=True)
    st.caption("Any cell < 0.5 (indicating >50% false positives on benign traffic) is flagged in teal‑red.")

    st.markdown("---")
    st.markdown('<div class="section-header">Detailed per‑seed metrics</div>', unsafe_allow_html=True)
    # Seed selector
    det_seed = st.selectbox(
        "Select seed for detailed view",
        options=seeds_of(data),
        key="det_seed"
    )
    # Build detailed table for selected seed
    det_rows = []
    methods = ["fedavg"] + [f"fedprox_mu{m}" for m in fedprox_mus(data.get("runs", []))] + ["fedper", "local_only"]
    for method_key in methods:
        run = method_run(data, method_key, det_seed)
        if run is None:
            continue
        cm = run.get("client_metrics", {})
        row = {
            "Method": method_label(method_key),
            "Wi‑Fi F1": fmt(client_metric(run, "wifi", "f1")),
            "Wi‑Fi Recall": fmt(client_metric(run, "wifi", "benign_recall")),
            "MQTT F1": fmt(client_metric(run, "mqtt", "f1")),
            "MQTT Recall": fmt(client_metric(run, "mqtt", "benign_recall")),
            "Bluetooth F1": fmt(client_metric(run, "bluetooth", "f1")),
            "Bluetooth Recall": fmt(client_metric(run, "bluetooth", "benign_recall")),
        }
        # Status flag for collapse
        bt_rec = client_metric(run, "bluetooth", "benign_recall")
        bt_f1 = client_metric(run, "bluetooth", "f1")
        if bt_rec is not None and bt_f1 is not None and bt_rec < 0.15 and bt_f1 < 0.50:
            row["Status"] = "Collapsed"
        else:
            row["Status"] = "OK"
        det_rows.append(row)
    df_det = pd.DataFrame(det_rows)
    # Row‑style function for collapsed rows
    def row_style(row):
        if row["Status"] == "Collapsed":
            return ["background-color:#FCEDE5; color:#7C2D12; font-weight:600"] * len(row)
        return [""] * len(row)
    styled_det = df_det.style.format(
        {c: lambda x: fmt(x) if isinstance(x, (float, int)) else x for c in df_det.columns if c not in ["Method", "Status"]}
    ).apply(row_style, axis=1)
    st.dataframe(styled_det, height=300, use_container_width=True)
    st.caption("Note: Wi‑Fi binary F1 is inflated by class imbalance (96.4% attack samples). Macro F1 and per‑class precision/recall are used throughout to avoid false optimism.")

    st.markdown("---")
    st.markdown('<div class="alert-warn">', unsafe_allow_html=True)
    # Bluetooth collapse callout
    fedavg_bt_rec = client_metric(method_run(data, "fedavg", seeds_of(data)[0]), "bluetooth", "benign_recall")
    worst_fedprox_bt_rec = worst_bt_recall
    fedper_bt_rec = client_metric(method_run(data, "fedper", seeds_of(data)[0]), "bluetooth", "benign_recall")
    local_bt_rec = local_bt_min
    st.markdown(
        f"""
        **Bluetooth collapse callout**
        - FedAvg BT benign recall: {fmt(fedavg_bt_rec)}
        - Worst FedProx BT benign recall (μ={worst_mu}, seed={worst_seed}): {fmt(worst_fedprox_bt_rec)}
        - FedPer BT benign recall: {fmt(fedper_bt_rec)}
        - Local‑only BT benign recall: {fmt(local_bt_rec)}
        The collapse is driven by sharing; proximal term exacerbates it. Personalizing the final layer (FedPer) recovers most of the gap.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ===== TAB 2: Explanation Divergence =====
with tab2:
    st.markdown('<div class="section-header">Explanation divergence (across clients)</div>', unsafe_allow_html=True)

    # Method and seed selectors
    div_methods = ["fedavg"] + [f"fedprox_mu{m}" for m in fedprox_mus(data.get("runs", []))] + ["fedper", "local_only", "cluster_fl"]
    div_method = st.selectbox(
        "Select method",
        options=div_methods,
        format_func=method_label,
        key="div_method"
    )
    div_seed = st.selectbox(
        "Select seed",
        options=seeds_of(data),
        key="div_seed"
    )

    # Resolve run
    run = method_run(data, div_method, div_seed)
    is_cluster_fl_empty = (div_method == "cluster_fl" and run is None)

    if is_cluster_fl_empty:
        st.info(
            f"""
            **{method_label('cluster_fl')}**
            {data.get("divergence_summary", {}).get('cluster_fl', {}).get('scope', CLUSTER_FL_STATUS)}
            To populate this slot:
            1. Teammate drops JSON array at `results/cluster_fl_results.json` following `CLUSTER_FL_RUN_SCHEMA` (see `src/run_study_experiments.py`).
            2. Run `python src/merge_cluster_fl.py` from repo root.
            No code changes required.
            """
        )
        # Show placeholder charts
        col1, col2 = st.columns([3, 2])
        with col1:
            st.plotly_chart(
                go.Figure().update_layout(
                    title="Top‑10 SHAP values (placeholder)",
                    xaxis_title="Mean |SHAP|",
                    yaxis_title="Feature",
                    plot_bgcolor="white",
                    paper_bgcolor="white"
                ),
                width="stretch"
            )
        with col2:
            st.plotly_chart(
                go.Figure().update_layout(
                    title="Jaccard similarity heatmap (placeholder)",
                    plot_bgcolor="white",
                    paper_bgcolor="white"
                ),
                width="stretch"
            )
    else:
        # Charts
        col1, col2 = st.columns([3, 2])
        with col1:
            # Top‑10 SHAP bar chart
            shap_top10 = run.get("shap_top10_per_client", {})
            # Aggregate across clients: mean |SHAP| per feature
            agg_shap = {}
            for proto in PROTOS:
                for feature, mean_abs in shap_top10.get(proto, []):
                    agg_shap[feature] = agg_shap.get(feature, 0.0) + mean_abs
            # Average across 3 clients
            for feat in agg_shap:
                agg_shap[feat] /= len(PROTOS)
            # Sort and take top 10
            top10_sorted = sorted(agg_shap.items(), key=lambda x: x[1], reverse=True)[:10]
            if top10_sorted:
                features, values = zip(*top10_sorted)
                fig_shap = go.Figure(
                    go.Bar(
                        x=list(values),
                        y=list(features),
                        orientation="h",
                        marker=dict(
                            color=list(values),
                            colorscale=[[0, "#D7EBE9"], [1, NAVY]],
                            showscale=False
                        )
                    )
                )
                fig_shap.update_layout(
                    title="Top‑10 mean |SHAP| values (averaged across clients)",
                    xaxis_title="Mean |SHAP|",
                    yaxis_title="",
                    height=400,
                    margin=dict(l=150, r=20, t=40, b=20),
                    plot_bgcolor="white",
                    paper_bgcolor="white"
                )
                st.plotly_chart(fig_shap, width="stretch")
            else:
                st.warning("SHAP top‑10 data not available.")
        with col2:
            # Jaccard 3×3 heatmap
            jaccard_pairwise = run.get("jaccard_pairwise", {})
            # Build matrix
            matrix = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            pair_idx = {pair: i for i, pair in enumerate(PAIRS)}
            for pair, val in jaccard_pairwise.items():
                if pair in pair_idx:
                    i = pair_idx[pair]
                    # We'll fill a symmetric matrix for simplicity
                    matrix[i][i] = val  # diagonal is not used, but we'll show the value
                    # For off‑diagonal, we need to map pair to coordinates; since we have only 3 pairs, we do:
                    # Wi‑Fi–MQTT → (0,1) and (1,0)
                    # Wi‑Fi–BT → (0,2) and (2,0)
                    # MQTT–BT → (1,2) and (2,1)
            # Simpler: just show the three values as a table; but we'll do a heatmap with labels
            fig_heat = go.Figure(
                go.Heatmap(
                    z=[[jaccard_pairwise.get("wifi-mqtt", 0.0), jaccard_pairwise.get("wifi-bluetooth", 0.0), jaccard_pairwise.get("mqtt-bluetooth", 0.0)]],
                    x=["Wi‑Fi–MQTT", "Wi‑Fi–BT", "MQTT–BT"],
                    y=[""],
                    colorscale=[[0, "#FFFFFF"], [1, TEAL]],
                    text=[[f"{jaccard_pairwise.get('wifi-mqtt', 0.0):.3f}", f"{jaccard_pairwise.get('wifi-bluetooth', 0.0):.3f}", f"{jaccard_pairwise.get('mqtt-bluetooth', 0.0):.3f}"]],
                    texttemplate="%{text}",
                    textfont={"color": "black"}
                )
            )
            fig_heat.update_layout(
                title="Jaccard similarity (Top‑5 feature sets)",
                height=250,
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor="white",
                paper_bgcolor="white"
            )
            st.plotly_chart(fig_heat, width="stretch")

        # Full‑metrics table
        st.markdown("---")
        st.markdown('<div class="section-header">Full‑metrics table (Jaccard + Spearman + cosine)</div>', unsafe_allow_html=True)
        ds_entry = data.get("divergence_summary", {}).get(run_key(run), {}) if run else {}
        seed42 = ds_entry.get("seed42", {})
        seed7 = ds_entry.get("seed7", {})
        noise_floor = ds_entry.get("noise_floor_seed_to_seed", {})
        # Build rows for each pair
        full_rows = []
        for pair in PAIRS:
            label = PAIR_LABEL[pair]
            j42 = seed42.get(pair, {}).get("jaccard")
            s42 = seed42.get(pair, {}).get("spearman")
            c42 = seed42.get(pair, {}).get("cosine")
            j7 = seed7.get(pair, {}).get("jaccard")
            s7 = seed7.get(pair, {}).get("spearman")
            c7 = seed7.get(pair, {}).get("cosine")
            dj = noise_floor.get(pair, {}).get("jaccard_abs_diff")
            ds = noise_floor.get(pair, {}).get("spearman_abs_diff")
            dc = noise_floor.get(pair, {}).get("cosine_abs_diff")
            full_rows.append({
                "Pair": label,
                "J · 42": fmt(j42),
                "ρ · 42": fmt(s42),
                "cos · 42": fmt(c42),
                "J · 7": fmt(j7),
                "ρ · 7": fmt(s7),
                "cos · 7": fmt(c7),
                "|ΔJ|": fmt(dj),
                "|Δρ|": fmt(ds),
                "|Δcos|": fmt(dc)
            })
        df_full = pd.DataFrame(full_rows)
        st.dataframe(
            df_full.style.format(
                {c: lambda x: fmt(x) if isinstance(x, (float, int)) else x for c in df_full.columns if c != "Pair"}
            ),
            height=200,
            use_container_width=True
        )
        st.caption("Last three columns show the seed‑to‑seed |difference| (noise floor). A between‑method gap smaller than its noise‑floor cell is within seed noise.")

        # Divergence scope caption
        scope = ds_entry.get("scope", "Scope not available")
        st.markdown(f"> **Divergence scope**: {scope}")

        # FedPer and local‑only comparison tables
        if div_method == "fedper":
            st.markdown("---")
            st.markdown('<div class="section-header">FedPer explanation similarity vs baselines (per client)</div>', unsafe_allow_html=True)
            comp_vs_fedavg = run.get("fedper_vs_fedavg", {})
            comp_vs_local = run.get("fedper_vs_local_only", {})
            comp_cols = st.columns(3)
            for idx, proto in enumerate(PROTOS):
                with comp_cols[idx]:
                    st.markdown(f"**{PROTO_LABEL[proto]}**")
                    # FedPer vs FedAvg
                    fedavg_comp = comp_vs_fedavg.get(proto, {})
                    df_fedavg = pd.DataFrame([
                        {"Metric": "Jaccard", "Value": fmt(fedavg_comp.get("top5_jaccard"))},
                        {"Metric": "Spearman ρ", "Value": fmt(fedavg_comp.get("spearman"))},
                        {"Metric": "Cosine", "Value": fmt(fedavg_comp.get("cosine"))}
                    ])
                    st.dataframe(df_fedavg.hide(index=True), height=120)
                    # FedPer vs Local‑only
                    local_comp = comp_vs_local.get(proto, {})
                    df_local = pd.DataFrame([
                        {"Metric": "Jaccard", "Value": fmt(local_comp.get("top5_jaccard"))},
                        {"Metric": "Spearman ρ", "Value": fmt(local_comp.get("spearman"))},
                        {"Metric": "Cosine", "Value": fmt(local_comp.get("cosine"))}
                    ])
                    st.dataframe(df_local.hide(index=True), height=120)
        elif div_method == "local_only":
            st.markdown("---")
            st.markdown('<div class="section-header">Local‑only vs FedAvg (per client)</div>', unsafe_allow_html=True)
            local_vs_global = data.get("local_vs_global", {})
            key = f"seed{div_seed}_local_vs_fedavg_seed{div_seed}"
            comp = local_vs_global.get(key, {})
            if comp:
                comp_cols = st.columns(3)
                for idx, proto in enumerate(PROTOS):
                    with comp_cols[idx]:
                        st.markdown(f"**{PROTO_LABEL[proto]}**")
                        df_local = pd.DataFrame([
                            {"Metric": "Jaccard", "Value": fmt(comp.get(f"{proto}_top5_jaccard"))},
                            {"Metric": "Spearman ρ", "Value": fmt(comp.get(f"{proto}_spearman"))},
                            {"Metric": "Cosine", "Value": fmt(comp.get(f"{proto}_cosine"))}
                        ])
                        st.dataframe(df_local.hide(index=True), height=120)
            else:
                st.warning("Local‑vs‑global comparison data not available for this seed.")

# ===== TAB 3: Methods Compared =====
with tab3:
    st.markdown('<div class="section-header">Methods compared</div>', unsafe_allow_html=True)

    # Centerpiece: BT benign recall vs μ
    st.markdown("#### Bluetooth benign recall vs μ")
    fig_bt = go.Figure()
    seeds = seeds_of(data)
    # Lines per seed
    for i, seed in enumerate(seeds):
        # Collect points: FedAvg (μ=0) + FedProx mus
        x_vals = ["FedAvg (μ=0)"] + [f"μ={m}" for m in fedprox_mus(data.get("runs", []))]
        y_vals = []
        # FedAvg (μ=0)
        run_fedavg = method_run(data, "fedavg", seed)
        y_vals.append(client_metric(run_fedavg, "bluetooth", "benign_recall") if run_fedavg else None)
        # FedProx points
        for mu in fedprox_mus(data.get("runs", [])):
            run = method_run(data, f"fedprox_mu{mu}", seed)
            y_vals.append(client_metric(run, "bluetooth", "benign_recall") if run else None)
        # Filter out None
        points = [(x, y) for x, y in zip(x_vals, y_vals) if y is not None]
        if points:
            x_pts, y_pts = zip(*points)
            fig_bt.add_trace(
                go.Scatter(
                    x=list(x_pts),
                    y=list(y_pts),
                    mode="lines+markers",
                    name=f"Seed {seed}",
                    line=dict(color=NAVY if i == 0 else BLUE, width=2),
                    marker=dict(size=8)
                )
            )
    # Reference lines: local‑only (dashed teal) and FedPer (dotted amber)
    local_bt_means = []
    for seed in seeds:
        run = local_entry(data, seed)
        if run:
            local_bt_means.append(client_metric(run, "bluetooth", "benign_recall"))
    local_bt_mean = sum(local_bt_means) / len(local_bt_means) if local_bt_means else 0.0
    fig_bt.add_hline(
        y=local_bt_mean,
        line_dash="dash",
        line_color=TEAL,
        annotation_text=f"Local‑only ({local_bt_mean:.3f})",
        annotation_position="top left"
    )
    fedper_bt_means = []
    for seed in seeds:
        run = method_run(data, "fedper", seed)
        if run:
            fedper_bt_means.append(client_metric(run, "bluetooth", "benign_recall"))
    fedper_bt_mean = sum(fedper_bt_means) / len(fedper_bt_means) if fedper_bt_means else 0.0
    fig_bt.add_hline(
        y=fedper_bt_mean,
        line_dash="dot",
        line_color=AMBER,
        annotation_text=f"FedPer ({fedper_bt_mean:.3f})",
        annotation_position="top right"
    )
    fig_bt.update_layout(
        title="Bluetooth benign recall (held‑out test)",
        xaxis_title="Method",
        yaxis_title="Benign recall",
        yaxis=dict(range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    st.plotly_chart(fig_bt, width="stretch")

    st.markdown("---")
    # Secondary chart: avg macro F1 per method
    st.markdown("#### Average macro F1 per method (two‑seed mean)")
    fig_f1 = go.Figure()
    methods_for_f1 = ["fedavg"] + [f"fedprox_mu{m}" for m in fedprox_mus(data.get("runs", []))] + ["fedper", "local_only", "centralized"]
    x_labels = []
    y_means = []
    y_errors = []  # half spread for error bars
    for method_key in methods_for_f1:
        if method_key == "centralized":
            # Use centralized_by_seed: average across seeds
            cents = data.get("centralized_by_seed", {})
            seed_vals = []
            for seed in seeds:
                cents_run = cents.get(str(seed))
                if cents_run:
                    f1_vals = [cents_run.get(f"{p}_f1_macro") for p in PROTOS if cents_run.get(f"{p}_f1_macro") is not None]
                    if f1_vals:
                        seed_vals.append(sum(f1_vals) / len(f1_vals))
            if seed_vals:
                mean_val, spread = mean_spread(seed_vals)
                x_labels.append("Centralized")
                y_means.append(mean_val)
                y_errors.append(spread / 2)
        else:
            seed_vals = []
            for seed in seeds:
                run = method_run(data, method_key, seed)
                if run:
                    seed_vals.append(avg_macro_f1(run))
            if seed_vals:
                mean_val, spread = mean_spread(seed_vals)
                x_labels.append(method_label(method_key))
                y_means.append(mean_val)
                y_errors.append(spread / 2)
    # Assign colors: FedAvg/FedProx → BLUE, FedPer → AMBER, Local‑only → TEAL, Centralized → GREY
    colors = []
    for label in x_labels:
        if label.startswith("FedProx") or label == "FedAvg":
            colors.append(BLUE)
        elif label == "FedPer":
            colors.append(AMBER)
        elif label == "Local-only":
            colors.append(TEAL)
        else:  # Centralized
            colors.append(GREY)
    fig_f1.add_trace(
        go.Bar(
            x=x_labels,
            y=y_means,
            error_y=dict(type="data", array=y_errors, visible=True),
            marker_color=colors,
            text=[f"{y:.4f}" for y in y_means],
            textposition="outside"
        )
    )
    fig_f1.update_layout(
        title="Average macro F1 (two‑seed mean ± seed spread)",
        xaxis_title="Method",
        yaxis_title="Macro F1",
        yaxis=dict(range=[0, 1.0]),
        uniformtext_minsize=8,
        uniformtext_mode="hide",
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    st.plotly_chart(fig_f1, width="stretch")

    st.markdown("---")
    st.markdown('<div class="section-header">Written findings</div>', unsafe_allow_html=True)
    fedper_f1_means = [avg_macro_f1(r) for s in seeds if (r := method_run(data, "fedper", s))]
    fedper_f1_mean = sum(fedper_f1_means) / len(fedper_f1_means) if fedper_f1_means else 0.0
    st.markdown(
        f"""
        - **No μ saves Bluetooth**: Even the gentlest proximal term (μ=0.001) lowers Bluetooth benign recall below FedAvg ({fmt(client_metric(method_run(data, 'fedavg', seeds_of(data)[0]), 'bluetooth', 'benign_recall'))} → {fmt(client_metric(method_run(data, f'fedprox_mu{fedprox_mus(data.get("runs", []))[0]}', seeds_of(data)[0]), 'bluetooth', 'benign_recall'))}). By μ=0.1 the model flags ~97–99% of Bluetooth benign traffic as attack. Local‑only sits at {fmt(local_bt_min)} — the collapse is caused by *sharing*, and proximal anchoring only deepens it.
        - **Local‑only beats all sharing**: Local‑only achieves {fmt(local_mean)} avg macro F1 vs {fmt(best_shared_mean)} for the best shared model ({method_label(best_shared_label)}) — a {local_mean - best_shared_mean:+.4f} absolute gain. It matches the pooled centralized reference ({fmt(data.get('centralized_by_seed', {}).get('42', {}).get('f1_macro', 0.0))}/{fmt(data.get('centralized_by_seed', {}).get('7', {}).get('f1_macro', 0.0))}) within {abs(local_mean - 0.9792):.4f} points and beats it on Bluetooth benign recall ({fmt(local_bt_min)} vs {fmt(data.get('centralized_by_seed', {}).get('42', {}).get('bluetooth_benign_recall', data.get('centralized_by_seed', {}).get('42', {}).get('benign_recall', 0.0)))}/{fmt(data.get('centralized_by_seed', {}).get('7', {}).get('bluetooth_benign_recall', data.get('centralized_by_seed', {}).get('7', {}).get('benign_recall', 0.0)))}).
        - **FedPer’s trade‑off position**: FedPer recovers {fmt(fedper_bt_mean)} Bluetooth benign recall — {(fedper_bt_mean - client_metric(method_run(data, 'fedavg', seeds_of(data)[0]), 'bluetooth', 'benign_recall')) / (local_bt_min - client_metric(method_run(data, 'fedavg', seeds_of(data)[0]), 'bluetooth', 'benign_recall')) * 100:.0f}% of the FedAvg→local‑only gap. Its avg macro F1 ({fmt(fedper_f1_mean)}) is {fedper_f1_mean / local_mean * 100:.0f}% of local‑only, and cross‑client explanation similarity remains at FedAvg level (see Tab 2).
        """
    )

# ===== TAB 4: Methodology & Limitations =====
with tab4:
    st.markdown('<div class="section-header">Study scale & methodology</div>', unsafe_allow_html=True)
    st.markdown(
        """
        - **Seeds**: 2 (42, 7) — directional trends only, no statistical inference.
        - **Rounds**: 10 federated rounds per run.
        - **Local epochs**: 3 per round.
        - **Features**: 45 network flow features (header, timing, TCP flags, protocol flags, statistical aggregates).
        - **Models**: Shallow 3‑layer MLP (45→64→32→1) with ReLU and dropout.
        - **Explainability**: SHAP DeepExplainer with 100 benign background samples per client.
        - **Divergence metrics**: Jaccard similarity of Top‑5 feature sets, Spearman and cosine similarity over full 45‑dim mean‑|SHAP| vectors.
        - **Noise floor**: Seed‑to‑seed |difference| of each statistic (n=2 seeds).
        - **Methods compared**:
          - **FedAvg**: Shared model, no proximal term.
          - **FedProx**: Shared model + proximal term (μ in {0.001, 0.01, 0.05, 0.1}).
          - **FedPer**: Shared trunk (first 4 parameter arrays) + private per‑client output head (last 2 arrays).
          - **Local‑only**: No sharing (control).
          - **Centralized**: Pooled data (upper reference).
          - **Cluster‑FL**: Empty slot — awaiting teammate’s port.
        """
    )
    st.markdown("---")
    st.markdown('<div class="section-header">Verified vs. planned</div>', unsafe_allow_html=True)
    verified_col, planned_col = st.columns(2)
    with verified_col:
        st.markdown("**Verified**")
        st.markdown(
            """
            - Centralized baseline benchmark
            - Multi‑client federated training with Flower (FedAvg)
            - Post‑hoc explainability pipeline (DeepExplainer + Jaccard)
            - FedProx proximal term integration (μ=0.001, 0.01, 0.05, 0.1)
            - Diagnostic robustness check (Wi‑Fi class imbalance, weight norms)
            - Scaled‑up study run on `data/study` with 2‑seed pilot
            - Interactive Streamlit research dashboard (this app)
            """
        )
    with planned_col:
        st.markdown("**Planned / Open**")
        st.markdown(
            """
            - Full 10‑seed noise‑floor calibration & logarithmic μ sweep
            - Advanced aggregators (SCAFFOLD, FedNova, FedOpt)
            - Gradient‑based explainers (Integrated Gradients, etc.)
            - Teammate’s Cluster‑FL port (see below)
            """
        )

    st.markdown("---")
    st.markdown('<div class="section-header">Cluster‑FL status</div>', unsafe_allow_html=True)
    st.info(
        f"""
        **Current status**: {CLUSTER_FL_STATUS}
        **Expected schema**: See `CLUSTER_FL_RUN_SCHEMA` in `src/run_study_experiments.py` (requires `aggregator="cluster_fl"`, `mu=null`, `seed`, `clusters`, `client_metrics`, `avg_macro_f1`, `shap_*` keys, `jaccard_pairwise`, `divergence_full`, `shared_features`, `divergence_scope`, plus comparison sets `cluster_fl_vs_fedavg` and `cluster_fl_vs_local_only` — these are auto‑backfilled at merge by `enrich_cluster_fl_comparisons()`).
        **Merge workflow**:
        1. Teammate drops JSON array at `results/cluster_fl_results.json` following the schema.
        2. Run `python src/merge_cluster_fl.py` from repo root.
        3. The tool validates, enriches with missing comparison sets, and merges into `study_results.json` (replacing any existing `aggregator='cluster_fl'` entries).
        No other keys in `studresults.json` are touched; no retraining is performed.
        """
    )

    st.markdown("---")
    st.markdown('<div class="section-header">Seed‑to‑seed stability table (all aggregators)</div>', unsafe_allow_html=True)
    # Build stability table from preliminary_noise_floor for all aggregators
    prelim = data.get("preliminary_noise_floor", {})
    agg_keys = ["fedavg"] + [f"fedprox_mu{m}" for m in [0.001, 0.01, 0.05, 0.1]] + ["fedper", "local_only"]
    # Ensure we have entries for all aggregators in prelim; if not, compute from data? But spec says prelim covers all 7 non‑cluster aggregators.
    stab_rows = []
    for agg in agg_keys:
        key = agg.replace("fedprox_mu", "fedprox_mu")  # just in case
        entry = prelim.get(key, {})
        # Per‑client Top‑5 stability
        wifi_stab = entry.get("per_client_top5_stability", {}).get("wifi", {}).get("seed_to_seed_jaccard")
        mqtt_stab = entry.get("per_client_top5_stability", {}).get("mqtt", {}).get("seed_to_seed_jaccard")
        bt_stab = entry.get("per_client_top5_stability", {}).get("bluetooth", {}).get("seed_to_seed_jaccard")
        # Max pairwise Jaccard abs_diff
        pairwise_spread = entry.get("pairwise_divergence_spread", {})
        max_pairwise = max([v.get("abs_diff", 0.0) for v in pairwise_spread.values() if isinstance(v, dict)], default=0.0)
        # Status
        status = entry.get("status", "unknown")
        stab_rows.append({
            "Aggregator": method_label(agg),
            "Wi‑Fi Top‑5 Jaccard": fmt(wifi_stab),
            "MQTT Top‑5 Jaccard": fmt(mqtt_stab),
            "Bluetooth Top‑5 Jaccard": fmt(bt_stab),
            "Max pairwise Jaccard |diff|": fmt(max_pairwise),
            "Status": status
        })
    df_stab = pd.DataFrame(stab_rows)
    st.dataframe(
        df_stab.style.format(
            {c: lambda x: fmt(x) if isinstance(x, (float, int)) else x for c in df_stab.columns if c not in ["Aggregator", "Status"]}
        ),
        height=250,
        use_container_width=True
    )

    st.markdown("---")
    st.markdown('<div class="section-header">System architecture diagram</div>', unsafe_allow_html=True)
    if os.path.exists(ARCH_IMG_PATH):
        st.image(ARCH_IMG_PATH, width="stretch")
        st.caption("Figure 1: End‑to‑end FedXplain‑IoMT system architecture.")
    else:
        st.warning(f"Architecture diagram not found: {ARCH_IMG_PATH}")

# ---- Footer ----
st.markdown("---")
st.caption(
    "FedXplain‑IoMT • Experimental research framework • "
    f"Results generated with Python {'.'.join(map(str, sys.version_info[:3]))} • "
    "Last updated: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
)