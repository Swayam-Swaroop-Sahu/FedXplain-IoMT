import json
import os
import sys
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from typing import Dict, List, Tuple, Optional

# ---- Constants and configuration ----
try:
    from src.config import CLUSTER_FL_STATUS
except ImportError:
    CLUSTER_FL_STATUS = "in progress - port pending"

# Color palette (from figures and spec)
NAVY = "#0F2A43"
TEAL = "#2A9D8F"
BLUE = "#2A78D6"
AMBER = "#C98500"
HIGHLIGHT = "#E76F51"
# Darker text variants of the accent colours (>= 4.5:1 on white) for chart labels
TEAL_TEXT = "#1D6F65"
AMBER_TEXT = "#8A5B00"
ALERT_TEXT = "#A63A1F"
CARD_BORDER = "#E5E5DE"
MUTED = "#5B6B7A"
GREY = "#8896A6"
GRID = "#D8D8D2"
FONT_STACK = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'

# Charts are rendered with theme=None, so this template fully controls their look
# (otherwise Streamlit's dark theme paints light text onto the white plot backgrounds).
_AXIS = dict(
    gridcolor="#ECECE6", linecolor=GRID, zerolinecolor=GRID, automargin=True, ticklabelstandoff=6,
    tickfont=dict(color=MUTED, size=12), title=dict(font=dict(color=MUTED, size=13), standoff=10),
)
pio.templates["fedxplain"] = go.layout.Template(pio.templates["plotly_white"])
pio.templates["fedxplain"].layout.update(
    font=dict(family=FONT_STACK, color=NAVY, size=13),
    title=dict(font=dict(color=NAVY, size=14), x=0, xanchor="left"),
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=8, r=8, t=36, b=16),
    annotationdefaults=dict(font=dict(color=MUTED, size=12)),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                font=dict(color=MUTED), bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(bgcolor="white", bordercolor=CARD_BORDER, font=dict(family=FONT_STACK, color=NAVY)),
    xaxis=_AXIS,
    yaxis=_AXIS,
)
pio.templates.default = "fedxplain"


def show_chart(fig: go.Figure, alt: str) -> None:
    # Set explicitly: the template's backgrounds are not always honoured once Streamlit serializes the figure
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False}, alt=alt)

# One colour per method family, used by every chart.
METHOD_COLOR = {"fedavg": BLUE, "fedprox": BLUE, "fedper": AMBER, "cluster_fl": NAVY,
                "local_only": TEAL, "centralized": GREY}

def method_color(key: str) -> str:
    return METHOD_COLOR.get("fedprox" if key.startswith("fedprox") else key, GREY)

# Paths
RESULTS_PATH = os.path.join("results", "study_results.json")
GENERALIZATION_PATH = os.path.join("results", "generalization_results.json")
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

def client_metric(run: Optional[Dict], proto: str, key: str) -> Optional[float]:
    """Safely extract a metric from client_metrics."""
    if run is None:
        return None
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
        return "Cluster-FL"
    if key == "centralized":
        return "Centralized"
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

def bt_recall(data: Dict, key: str, seed: int) -> Optional[float]:
    """Bluetooth benign recall for a method key and seed, or None."""
    return client_metric(method_run(data, key, seed), "bluetooth", "benign_recall")

def kpi_card(label: str, value: str, caption: str, delta: Optional[str] = None,
             delta_description: Optional[str] = None, tip: Optional[str] = None,
             chart: Optional[List[float]] = None, delta_color: str = "normal",
             delta_arrow: str = "auto") -> None:
    """Bordered, equal-height KPI card: native metric (optional sparkline) plus a caption."""
    with st.container(border=True, height="stretch", gap="xsmall"):
        st.metric(label, value, delta, delta_color=delta_color, delta_arrow=delta_arrow, help=tip,
                  delta_description=delta_description, chart_data=chart or None, chart_type="line")
        st.caption(caption)

def takeaway(text: str) -> None:
    """One-sentence lede that opens every tab."""
    st.markdown(f":material/lightbulb: **{text}**")

def card(key: str, title: str, subtitle: Optional[str] = None, height: str = "content"):
    """Bordered card container with its header inside; use with `with`."""
    box = st.container(border=True, key=f"card_{key}", height=height)
    box.subheader(title, anchor=False)
    if subtitle:
        box.caption(subtitle)
    return box

# ---- Streamlit page config (look comes from .streamlit/config.toml) ----
st.set_page_config(
    page_title="FedXplain-IoMT Dashboard",
    page_icon=":material/local_hospital:",
    layout="wide",
    initial_sidebar_state="expanded"
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

@st.cache_data
def load_generalization() -> Optional[Dict]:
    """Load the zero-day / data-scarcity results, or None if not generated yet."""
    if not os.path.exists(GENERALIZATION_PATH):
        return None
    with open(GENERALIZATION_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

gen_data = load_generalization()

seeds = seeds_of(data)
mus = fedprox_mus(data.get("runs", []))
run_meta = data.get("run_metadata", {})
# Centralized is trained on pooled data; only pooled (all-protocol) metrics are stored.
_cents = data.get("centralized_by_seed", {})
cent_f1 = {s: _cents.get(str(s), {}).get("f1_macro") for s in seeds}
cent_benign = {s: _cents.get(str(s), {}).get("benign_recall") for s in seeds}

# ---- Sidebar ----
def _kv_table(items, alt: str) -> None:
    st.table(
        {str(k): ", ".join(map(str, v)) if isinstance(v, list) else str(v) for k, v in items},
        border="horizontal",
        alt=alt,
    )

with st.sidebar:
    st.subheader("About this run", anchor=False)
    if run_meta:
        _kv_table(((k.replace("_", " "), v) for k, v in run_meta.items()), alt="Run metadata")
    st.subheader("Study", anchor=False)
    _kv_table([
        ("Benchmark", "CICIoMT2024"),
        ("Protocols", "Wi‑Fi, MQTT, Bluetooth"),
        ("Seeds", f"{', '.join(map(str, seeds))} (directional only)"),
        ("Mode", "Read‑only, runs fully offline"),
    ], alt="Study summary")
    _cluster_fl_merged = bool(seeds) and all(
        find_run(data.get("runs", []), "cluster_fl", s) is not None for s in seeds
    )
    st.subheader("Cluster‑FL status", anchor=False)
    if _cluster_fl_merged:
        st.badge("Merged", icon=":material/check_circle:", color="green")
        st.caption("Results in Detection, Explanation Divergence and Methods Compared")
    else:
        st.badge(CLUSTER_FL_STATUS, icon=":material/pending:", color="orange")

# ---- Headline numbers (computed once, reused by every tab) ----
shared_methods = ["fedavg"] + [f"fedprox_mu{m}" for m in mus]

# KPI 1: Best shared-model avg macro F1
best_shared_mean = -1.0
best_shared_label = ""
shared_f1_sweep = []  # seed-mean avg macro F1 along FedAvg → FedProx μ sweep (sparkline)
for method in shared_methods:
    seed_vals = []
    for seed in seeds:
        run = method_run(data, method, seed)
        if run:
            seed_vals.append(avg_macro_f1(run))
    if seed_vals:
        mean_val, _ = mean_spread(seed_vals)
        shared_f1_sweep.append(mean_val)
        if mean_val > best_shared_mean:
            best_shared_mean = mean_val
            best_shared_label = method

# KPI 2: Local-only avg macro F1 + delta vs best shared
local_seed_vals = []
for seed in seeds:
    run = local_entry(data, seed)
    if run:
        local_seed_vals.append(avg_macro_f1(run))
local_mean, local_spread = mean_spread(local_seed_vals)
kpi2_delta = f"{local_mean - best_shared_mean:+.4f}"

# KPI 3: Worst FedProx BT benign recall vs local-only
worst_bt_recall = None
worst_mu = None
worst_seed = None
for mu in mus:
    for seed in seeds:
        bt_rec = bt_recall(data, f"fedprox_mu{mu}", seed)
        if bt_rec is not None and (worst_bt_recall is None or bt_rec < worst_bt_recall):
            worst_bt_recall = bt_rec
            worst_mu = mu
            worst_seed = seed
local_bt_recalls = [bt_recall(data, "local_only", s) for s in seeds]
local_bt_min = min([v for v in local_bt_recalls if v is not None], default=None)
worst_seed_sweep = [v for k in shared_methods if (v := bt_recall(data, k, worst_seed)) is not None]

# KPI 4: FedAvg Wi‑Fi–Bluetooth Spearman (two‑seed mean)
def wb_spearman(key: str, seed: int) -> Optional[float]:
    run = method_run(data, key, seed)
    return run.get("divergence_full", {}).get("wifi-bluetooth", {}).get("spearman") if run else None

spear_seeds = [v for s in seeds if (v := wb_spearman("fedavg", s)) is not None]
spear_mean, spear_spread = mean_spread(spear_seeds)
spear_sweep = [
    mean_spread(vals)[0] for k in shared_methods
    if (vals := [v for s in seeds if (v := wb_spearman(k, s)) is not None])
]

# Per-seed derived quantities shared by Tabs 0, 1 and 3
mu_min = mus[0] if mus else None
smallest_mu_text = "; ".join(
    f"{fmt(bt_recall(data, 'fedavg', s))} → {fmt(bt_recall(data, f'fedprox_mu{mu_min}', s))} (seed {s})"
    for s in seeds
) if mu_min is not None else "—"
fedper_gap = {}
for s in seeds:
    fa, fp, lo = (bt_recall(data, k, s) for k in ("fedavg", "fedper", "local_only"))
    if None not in (fa, fp, lo) and lo != fa:
        fedper_gap[s] = (fp - fa) / (lo - fa) * 100
fedper_gap_text = " / ".join(f"{v:.1f}% (seed {s})" for s, v in fedper_gap.items()) or "—"
fedper_f1_means = [avg_macro_f1(r) for s in seeds if (r := method_run(data, "fedper", s))]
fedper_f1_mean = sum(fedper_f1_means) / len(fedper_f1_means) if fedper_f1_means else None
fedper_vs_local_text = (
    f"{fedper_f1_mean / local_mean * 100:.1f}%" if fedper_f1_mean is not None and local_mean else "—"
)
_sweep_tip = "Sparkline: FedAvg → FedProx μ=" + " → ".join(map(str, mus))

# ---- Header ----
st.title("FedXplain-IoMT", anchor=False)
st.markdown(":gray[Federated learning & explainable AI for heterogeneous medical IoT]")
with st.container(horizontal=True, gap="small"):
    st.badge("CICIoMT2024", color="gray")
    st.badge(f"{len(PROTOS)} protocols · " + " / ".join(PROTO_LABEL[p] for p in PROTOS), color="gray")
    st.badge(f"Seeds {', '.join(map(str, seeds))}", color="gray")
    if run_meta.get("n_rounds") is not None:
        st.badge(f"{run_meta['n_rounds']} rounds × {run_meta.get('n_local_epochs', '—')} local epochs", color="gray")
    if run_meta.get("timestamp"):
        st.badge(f"Results {run_meta['timestamp']}", color="gray")
    st.badge(f"Directional only (n={len(seeds)})", icon=":material/info:", color="orange",
             help="Two seeds show trends, not statistical significance")
    if _cluster_fl_merged:
        st.badge("Cluster‑FL merged", icon=":material/check_circle:", color="green")
    st.badge("Offline · read‑only", icon=":material/wifi_off:", color="green")

# Headline numbers row; columns keep the four cards equal height
kpi1, kpi2, kpi3, kpi4 = st.columns(4, gap="medium")
with kpi1:
    kpi_card("Best shared-model F1", fmt(best_shared_mean),
             f"{method_label(best_shared_label)} · seed mean",
             tip=f"Best shared-model: {method_label(best_shared_label)} (avg macro F1, seed mean). {_sweep_tip}.",
             chart=shared_f1_sweep)
with kpi2:
    kpi_card("Local-only F1", fmt(local_mean),
             f"± {local_spread:.4f} seed spread · {len(seeds)}-seed mean",
             kpi2_delta, delta_description="vs best shared",
             tip=f"{kpi2_delta} vs best shared model ({method_label(best_shared_label)})")
with kpi3:
    kpi_card("BT benign recall (worst)", fmt(worst_bt_recall),
             f"FedProx μ={worst_mu}, seed {worst_seed}",
             f"{worst_bt_recall - local_bt_min:+.4f}" if None not in (worst_bt_recall, local_bt_min) else None,
             delta_description="vs local‑only min",
             tip=f"Worst FedProx (μ={worst_mu}, seed={worst_seed}) vs local-only min {fmt(local_bt_min)}. "
                 f"{_sweep_tip} (seed {worst_seed}).",
             chart=worst_seed_sweep)
with kpi4:
    kpi_card("Wi‑Fi–BT Spearman", fmt(spear_mean), f"FedAvg · {len(seeds)}‑seed mean",
             f"± {spear_spread:.4f}", delta_description="seed spread", delta_color="off", delta_arrow="off",
             tip=f"FedAvg Wi‑Fi–Bluetooth Spearman ρ ({len(seeds)}‑seed mean ± spread). {_sweep_tip} (seed mean).",
             chart=spear_sweep)

# ---- Tabs ----
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Executive Summary",
    "Detection",
    "Explanation Divergence",
    "Methods Compared",
    "Methodology & Limitations",
    "Zero-Day & Data Scarcity"
])

# ===== TAB 0: Executive Summary =====
with tab0:
    takeaway("One shared detector cannot serve three structurally different hospital protocols: on this benchmark "
             "the shared model collapses on Bluetooth, no proximal strength rescues it, and only personalizing "
             "the final layer — or not sharing at all — does.")

    fig_col, text_col = st.columns([3, 2], gap="medium")
    # Figure 4 (the image carries its own title and error-bar note)
    with fig_col, card("fig4", "Personalization trade‑off", "Figure 4 · detection quality vs explanation consistency",
                       height="stretch"):
        if os.path.exists(FIG4_PATH):
            st.image(FIG4_PATH, width="stretch",
                     alt="Scatter plot of average macro F1 versus cross-client explanation consistency per method")
        else:
            st.warning(f"Figure not found: {FIG4_PATH}")

    with text_col:
        with card("found", "What we found"):
            st.markdown(
                f"""
                - **Bluetooth collapse**: Shared models drop BT benign recall to as low as {fmt(worst_bt_recall)} (μ={worst_mu}, seed {worst_seed}), while local‑only stays at {fmt(local_bt_min)}.
                - **No safe μ**: Even μ={mu_min} worsens BT benign recall vs FedAvg ({smallest_mu_text}).
                - **FedPer’s trade‑off**: Recovers {fedper_gap_text} of the FedAvg→local‑only BT benign‑recall gap, with avg macro F1 {fmt(fedper_f1_mean)} ({fedper_vs_local_text} of local‑only {fmt(local_mean)}).
                - **Explanation consistency**: FedPer’s cross‑client similarity stays at FedAvg level (see the Explanation Divergence tab).
                """
            )
        with card("not_proven", "What we didn’t prove yet"):
            st.badge(f"Directional only (n={len(seeds)})", icon=":material/info:", color="orange")
            st.markdown(
                """
                - **Statistical significance**: n=2 seeds provide directional trends only; no p‑values or confidence intervals.
                - **Full noise floor**: Preliminary (n=2) only; more seeds needed for robust variance estimates.
                - **Other aggregators**: SCAFFOLD, FedNova, etc. remain for future work.
                """
            )

# ===== TAB 1: Detection =====
with tab1:
    takeaway(f"Sharing is what breaks Bluetooth: benign recall falls to {fmt(worst_bt_recall)} in the worst FedProx run "
             f"while local‑only keeps at least {fmt(local_bt_min)}, and personalizing the final layer (FedPer) "
             "recovers most of the gap.")
    # Build unified table
    def unified_detection_rows(data_dict: Dict) -> List[Dict]:
        rows = []
        methods = ["fedavg"] + [f"fedprox_mu{m}" for m in mus] + ["fedper", "cluster_fl", "local_only", "centralized"]
        for method_key in methods:
            row = {"Method": "Centralized (pooled)" if method_key == "centralized" else method_label(method_key)}
            # Per‑client per‑seed columns (centralized has no per-client metrics → None)
            for seed in seeds:
                run = method_run(data_dict, method_key, seed)
                for proto in PROTOS:
                    row[f"{PROTO_LABEL[proto]} · {seed}"] = client_metric(run, proto, "f1_macro")
            # Avg columns per seed
            avg_vals = []
            for seed in seeds:
                if method_key == "centralized":
                    val = cent_f1.get(seed)
                else:
                    run = method_run(data_dict, method_key, seed)
                    val = avg_macro_f1(run) if run else None
                row[f"Avg · {seed}"] = val
                if val is not None:
                    avg_vals.append(val)
            row["Avg mean ± spread"] = fmt_pm(avg_vals) if avg_vals else "—"
            rows.append(row)
        return rows

    df_unified = pd.DataFrame(unified_detection_rows(data))
    # Reorder columns for readability: Method, then Wi‑Fi 42, Wi‑Fi 7, MQTT 42, MQTT 7, BT 42, BT 7, then Avg·42, Avg·7, Avg mean ± spread
    cols = ["Method"]
    for seed in seeds:
        for proto in PROTOS:
            cols.append(f"{PROTO_LABEL[proto]} · {seed}")
    for seed in seeds:
        cols.append(f"Avg · {seed}")
    cols.append("Avg mean ± spread")
    df_unified = df_unified[cols]
    # Streamlit renders nulls as a grey "None" regardless of Styler na_rep, so pre-format.
    for c in cols[1:-1]:
        df_unified[c] = df_unified[c].map(lambda v: "—" if v is None or pd.isna(v) else f"{v:.4f}")
    # Build BT benign recall rows (chart + table)
    bt_rows = []
    bt_methods = ["fedavg"] + [f"fedprox_mu{m}" for m in mus] + ["fedper", "cluster_fl", "local_only", "centralized"]
    for method_key in bt_methods:
        if method_key == "centralized":
            row = {"Method": "Centralized (pooled, all protocols)"}
            row.update({str(s): cent_benign.get(s) for s in seeds})
        else:
            row = {"Method": method_label(method_key)}
            row.update({str(s): bt_recall(data, method_key, s) for s in seeds})
        bt_rows.append(row)
    df_bt = pd.DataFrame(bt_rows)
    seed_cols = [str(s) for s in seeds]

    # Dot plot: one row per method, one marker per seed, a faint bar spanning the seeds
    SEED_SYMBOLS = [("circle", "●"), ("circle-open", "○"), ("diamond", "◆"), ("square", "■")]
    fig_btm = go.Figure()
    for method_key, row in zip(bt_methods, bt_rows):
        color = method_color(method_key)
        y_lab = "Centralized (pooled)" if method_key == "centralized" else row["Method"]
        vals = [row[c] for c in seed_cols if row[c] is not None]
        if len(vals) > 1:
            fig_btm.add_trace(go.Scatter(
                x=[min(vals), max(vals)], y=[y_lab] * 2, mode="lines",
                line=dict(color=color, width=4), opacity=0.3, hoverinfo="skip",
            ))
        for i, s in enumerate(seeds):
            if row[str(s)] is None:
                continue
            fig_btm.add_trace(go.Scatter(
                x=[row[str(s)]], y=[y_lab], mode="markers", name=f"Seed {s}",
                marker=dict(color=color, size=11, symbol=SEED_SYMBOLS[i % 4][0], line=dict(width=2, color=color)),
                hovertemplate=f"Seed {s}: %{{x:.4f}}<extra></extra>",
            ))
        fig_btm.add_annotation(xref="paper", x=1, xanchor="left", xshift=12, yref="y", y=y_lab,
                               text=" / ".join(fmt(row[c]) for c in seed_cols), showarrow=False,
                               font=dict(color=NAVY, size=12))
    fig_btm.add_annotation(xref="paper", x=1, xanchor="left", xshift=12, yref="paper", y=1, yanchor="bottom",
                           text="Seeds " + " / ".join(seed_cols), showarrow=False)
    fig_btm.add_vline(x=0.5, line_dash="dot", line_color=HIGHLIGHT)
    fig_btm.add_annotation(x=0.5, xanchor="center", yref="paper", y=1, yanchor="bottom", showarrow=False,
                           text="0.5 = half of benign traffic flagged", font=dict(color=ALERT_TEXT))
    fig_btm.update_layout(
        showlegend=False,
        hovermode="y unified",
        height=60 + 40 * len(bt_rows),
        margin=dict(l=8, r=110, t=32, b=16),
        xaxis=dict(title="Bluetooth benign recall", range=[-0.02, 1.02]),
        yaxis=dict(autorange="reversed", title=None),
    )

    # Styling function to highlight cells < 0.5
    def highlight_bad(val):
        try:
            if val is None or (isinstance(val, str) and val == "—"):
                return ""
            v = float(val)
            if v < 0.5:
                return f"background-color:#FCEDE5; color:{ALERT_TEXT}; font-weight:700;"
            return ""
        except (TypeError, ValueError):
            return ""
    styled_bt = df_bt.style.map(highlight_bad, subset=seed_cols)

    seed_legend = ", ".join(f"{SEED_SYMBOLS[i % 4][1]} seed {s}" for i, s in enumerate(seeds))
    bt_col, callout_col = st.columns([7, 3], gap="medium")
    with bt_col, card("bt_recall", "Bluetooth benign recall by method", f"{seed_legend}; values below 0.5 highlighted",
                      height="stretch"):
        show_chart(fig_btm, alt="Dot plot of Bluetooth benign recall per method and seed; shared models fall far below 0.5 while local-only stays near 1")
        with st.expander("Table view", icon=":material/table_chart:"):
            st.dataframe(
                styled_bt,
                hide_index=True,
                column_config={c: st.column_config.NumberColumn(f"Seed {c}", format="%.4f") for c in seed_cols},
                height=(len(df_bt) + 1) * 35 + 3,
                width="stretch",
                alt="Bluetooth benign recall per method and seed",
            )

    # Bluetooth collapse callout
    bt_by_seed = {k: " / ".join(fmt(bt_recall(data, k, s)) for s in seeds) for k in ("fedavg", "fedper", "local_only")}
    with callout_col, card("bt_callout", "Bluetooth collapse", f"BT benign recall, seeds {' / '.join(map(str, seeds))}",
                           height="stretch"):
        st.badge("Shared models collapse", icon=":material/warning:", color="red")
        st.markdown(
            f"- **FedAvg**: {bt_by_seed['fedavg']}\n"
            f"- **Worst FedProx** (μ={worst_mu}, seed {worst_seed}): {fmt(worst_bt_recall)}\n"
            f"- **FedPer**: {bt_by_seed['fedper']}\n"
            f"- **Local‑only**: {bt_by_seed['local_only']}\n\n"
            "The collapse is driven by sharing; the proximal term exacerbates it. "
            "Personalizing the final layer (FedPer) recovers most of the gap."
        )
        st.caption("< 0.5 means >50% false positives on benign traffic. Centralized stores only pooled benign recall, not a Bluetooth‑specific value.")

    with card("unified", "Unified detection table (macro F1)", "Per‑client macro F1 by seed, then per‑seed averages"):
        st.dataframe(
            df_unified,
            hide_index=True,
            column_config={"Method": st.column_config.TextColumn(pinned=True)},
            height=(len(df_unified) + 1) * 35 + 3,
            width="stretch",
            alt="Per-client and average macro F1 for every method and seed",
        )
        st.caption("Centralized stores only pooled metrics: per‑client cells are empty and its Avg columns show pooled macro F1.")

    det_card = card("det_detail", "Detailed per‑seed metrics", "Binary F1 and benign recall per client; collapsed rows highlighted")
    # Seed selector
    det_seed = det_card.segmented_control(
        "Seed",
        options=seeds,
        default=seeds[0],
        required=True,
        key="det_seed"
    )
    # Build detailed table for selected seed
    det_rows = []
    methods = ["fedavg"] + [f"fedprox_mu{m}" for m in mus] + ["fedper", "cluster_fl", "local_only"]
    for method_key in methods:
        run = method_run(data, method_key, det_seed)
        if run is None:
            continue
        row = {
            "Method": method_label(method_key),
            "Wi‑Fi binary F1": client_metric(run, "wifi", "f1"),
            "Wi‑Fi benign recall": client_metric(run, "wifi", "benign_recall"),
            "MQTT binary F1": client_metric(run, "mqtt", "f1"),
            "MQTT benign recall": client_metric(run, "mqtt", "benign_recall"),
            "Bluetooth binary F1": client_metric(run, "bluetooth", "f1"),
            "Bluetooth benign recall": client_metric(run, "bluetooth", "benign_recall"),
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
    styled_det = df_det.style.apply(row_style, axis=1)
    _recall_col = st.column_config.ProgressColumn(format="%.4f", min_value=0.0, max_value=1.0)
    det_card.dataframe(
        styled_det,
        hide_index=True,
        column_config={
            c: _recall_col if "recall" in c else st.column_config.NumberColumn(format="%.4f")
            for c in df_det.columns if c not in ("Method", "Status")
        } | {"Method": st.column_config.TextColumn(pinned=True)},
        height=(len(df_det) + 1) * 35 + 3,
        width="stretch",
        alt=f"Binary F1 and benign recall per client for seed {det_seed}",
    )
    _wifi_att = client_metric(method_run(data, "fedavg", det_seed), "wifi", "n_attack")
    _wifi_ben = client_metric(method_run(data, "fedavg", det_seed), "wifi", "n_benign")
    wifi_attack_share = f"{_wifi_att / (_wifi_att + _wifi_ben) * 100:.1f}%" if _wifi_att and _wifi_ben is not None else "—"
    det_card.caption(f"Wi‑Fi binary F1 is inflated by class imbalance ({wifi_attack_share} attack samples in the Wi‑Fi test split); macro F1 and per‑class precision/recall are used throughout.")

# ===== TAB 2: Explanation Divergence =====
with tab2:
    takeaway("FedPer’s cross‑client explanation similarity stays at FedAvg level; read any between‑method gap "
             "against the seed‑to‑seed noise floor before treating it as real.")
    # Method and seed selectors, placed directly above the charts they drive
    div_methods = ["fedavg"] + [f"fedprox_mu{m}" for m in mus] + ["fedper", "local_only", "cluster_fl"]
    with st.container(horizontal=True, vertical_alignment="top", gap="medium"):
        div_method = st.selectbox(
            "Method",
            options=div_methods,
            format_func=method_label,
            key="div_method",
            width=260,
        )
        div_seed = st.segmented_control(
            "Seed",
            options=seeds,
            default=seeds[0],
            required=True,
            key="div_seed"
        )

    # Resolve run
    run = method_run(data, div_method, div_seed)
    is_cluster_fl_empty = (div_method == "cluster_fl" and run is None)

    if is_cluster_fl_empty:
        st.info(
            f"**{method_label('cluster_fl')}**: "
            f"{data.get('divergence_summary', {}).get('cluster_fl', {}).get('scope', CLUSTER_FL_STATUS)}"
        )
        with st.expander("How to populate this slot"):
            st.markdown(
                """
                1. Teammate drops JSON array at `results/cluster_fl_results.json` following `CLUSTER_FL_RUN_SCHEMA` (see `src/run_study_experiments.py`).
                2. Run `python src/merge_cluster_fl.py` from repo root.

                No code changes required.
                """
            )
        # Show placeholder charts
        col1, col2 = st.columns([3, 2])
        with col1:
            show_chart(
                go.Figure().update_layout(
                    title="Top‑10 SHAP values (placeholder)",
                    xaxis_title="Mean |SHAP|",
                    yaxis_title="Feature",
                    plot_bgcolor="white",
                    paper_bgcolor="white"
                ),
                alt="Empty placeholder for the top-10 SHAP chart",
            )
        with col2:
            show_chart(
                go.Figure().update_layout(
                    title="Jaccard similarity heatmap (placeholder)",
                    plot_bgcolor="white",
                    paper_bgcolor="white"
                ),
                alt="Empty placeholder for the Jaccard heatmap",
            )
    elif run is None:
        st.warning(f"No {method_label(div_method)} run found for seed {div_seed} in `{RESULTS_PATH}`.")
    else:
        ds_entry = data.get("divergence_summary", {}).get(run_key(run), {})
        # Divergence scope caption
        scope = ds_entry.get("scope", "Scope not available")
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.caption(f":material/info: **Divergence scope**: {scope}", width="content")
            if div_method == "cluster_fl":
                st.badge("Merged", icon=":material/check_circle:", color="green")
            st.badge(f"Directional only (n={len(seeds)})", icon=":material/info:", color="orange")
        div_color = method_color(div_method)

        # Charts
        col1, col2 = st.columns([1, 1], gap="medium")
        with col1, card("div_shap", "Top‑10 mean |SHAP| values", "Averaged across clients", height="stretch"):
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
                        marker_color=div_color,
                        text=[fmt(v) for v in values],
                        textposition="outside",
                        cliponaxis=False,
                        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
                    )
                )
                fig_shap.update_layout(
                    xaxis=dict(title="Mean |SHAP|", range=[0, max(values) * 1.18]),
                    yaxis=dict(title=None, autorange="reversed"),
                    height=380,
                    margin=dict(l=8, r=8, t=8, b=16),
                )
                show_chart(fig_shap, alt=f"Bar chart of top-10 mean absolute SHAP features for {method_label(div_method)}, seed {div_seed}")
            else:
                st.warning("SHAP top‑10 data not available.")
        with col2, card("div_jaccard", "Jaccard similarity", "Top‑5 feature sets, per client pair", height="stretch"):
            # Jaccard similarity per client pair; bars on a fixed 0–1 scale
            jaccard_pairwise = run.get("jaccard_pairwise", {})
            jac_vals = [jaccard_pairwise.get(p) for p in PAIRS]
            fig_jac = go.Figure(
                go.Bar(
                    x=jac_vals,
                    y=[PAIR_LABEL[p] for p in PAIRS],
                    orientation="h",
                    marker_color=div_color,
                    text=[fmt(v) for v in jac_vals],
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate="%{y}: %{x:.4f}<extra></extra>",
                )
            )
            fig_jac.update_layout(
                xaxis=dict(title="Top‑5 Jaccard", range=[0, 1.15], tickvals=[0, 0.25, 0.5, 0.75, 1]),
                yaxis=dict(title=None, autorange="reversed"),
                bargap=0.55,
                height=380,
                margin=dict(l=8, r=8, t=8, b=16),
            )
            show_chart(fig_jac, alt=f"Bar chart of top-5 Jaccard similarity per client pair for {method_label(div_method)}, seed {div_seed}")

        # Full‑metrics table
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
                "J · 42": j42,
                "ρ · 42": s42,
                "cos · 42": c42,
                "J · 7": j7,
                "ρ · 7": s7,
                "cos · 7": c7,
                "|ΔJ|": dj,
                "|Δρ|": ds,
                "|Δcos|": dc
            })
        df_full = pd.DataFrame(full_rows)
        _unit = st.column_config.ProgressColumn(format="%.4f", min_value=0.0, max_value=1.0)
        with card("div_full", "Full divergence metrics", "Top‑5 Jaccard (J), Spearman (ρ) and cosine per client pair and seed, with seed‑to‑seed noise floor"):
            st.dataframe(
                df_full,
                hide_index=True,
                column_config={
                    c: _unit if c.startswith(("J ·", "cos ·")) else st.column_config.NumberColumn(format="%.4f")
                    for c in df_full.columns if c != "Pair"
                } | {"Pair": st.column_config.TextColumn(pinned=True)},
                height=(len(df_full) + 1) * 35 + 3,
                width="stretch",
                alt="Jaccard, Spearman and cosine similarity per client pair and seed, with noise floor",
            )
            st.caption("|Δ| columns = seed‑to‑seed noise floor. A between‑method gap smaller than its noise‑floor cell is within seed noise.")

        # FedPer and local‑only comparison tables
        _sim_cfg = {"Jaccard": _unit, "Spearman ρ": st.column_config.NumberColumn(format="%.4f"), "Cosine": _unit}
        if div_method == "fedper":
            with card("div_fedper", "FedPer explanation similarity vs baselines", "Per client, against FedAvg and Local‑only"):
                comp_vs = {"FedAvg": run.get("fedper_vs_fedavg", {}), "Local‑only": run.get("fedper_vs_local_only", {})}
                df_comp = pd.DataFrame([
                    {
                        "Client": PROTO_LABEL[proto],
                        "Compared with": base,
                        "Jaccard": comp.get(proto, {}).get("top5_jaccard"),
                        "Spearman ρ": comp.get(proto, {}).get("spearman"),
                        "Cosine": comp.get(proto, {}).get("cosine"),
                    }
                    for proto in PROTOS for base, comp in comp_vs.items()
                ])
                st.dataframe(df_comp, hide_index=True, column_config=_sim_cfg,
                             height=(len(df_comp) + 1) * 35 + 3, width="stretch",
                             alt="FedPer explanation similarity per client against FedAvg and local-only")
        elif div_method == "local_only":
            with card("div_local", "Local‑only vs FedAvg", "Per‑client explanation similarity"):
                local_vs_global = data.get("local_vs_global", {})
                key = f"seed{div_seed}_local_vs_fedavg_seed{div_seed}"
                comp = local_vs_global.get(key, {})
                if comp:
                    df_local = pd.DataFrame([
                        {
                            "Client": PROTO_LABEL[proto],
                            "Jaccard": comp.get(f"{proto}_top5_jaccard"),
                            "Spearman ρ": comp.get(f"{proto}_spearman"),
                            "Cosine": comp.get(f"{proto}_cosine"),
                        }
                        for proto in PROTOS
                    ])
                    st.dataframe(df_local, hide_index=True, column_config=_sim_cfg,
                                 height=(len(df_local) + 1) * 35 + 3, width="stretch",
                                 alt="Local-only versus FedAvg explanation similarity per client")
                else:
                    st.warning("Local‑vs‑global comparison data not available for this seed.")

# ===== TAB 3: Methods Compared =====
with tab3:
    takeaway(f"Local‑only beats every shared model ({local_mean - best_shared_mean:+.4f} avg macro F1) and no μ rescues "
             "Bluetooth; FedPer recovers most of the gap and Cluster‑FL resolves the collapse structurally.")

    # Centerpiece: BT benign recall vs μ
    fig_bt = go.Figure()
    # Lines per seed (FedAvg/FedProx family → BLUE; seeds distinguished by dash), labelled at the line end
    seed_ends = []
    for i, seed in enumerate(seeds):
        x_vals = ["FedAvg (μ=0)"] + [f"μ={m}" for m in mus]
        y_vals = [bt_recall(data, "fedavg", seed)] + [bt_recall(data, f"fedprox_mu{mu}", seed) for mu in mus]
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
                    line=dict(color=BLUE, width=2, dash="solid" if i == 0 else "dashdot"),
                    marker=dict(size=8),
                    hovertemplate="%{y:.4f}",
                )
            )
            seed_ends.append((y_pts[-1], x_pts[-1], seed))
    # End labels stacked above the points (highest value on top) so they never touch the x tick labels
    for k, (y_end, x_end, seed) in enumerate(sorted(seed_ends, reverse=True)):
        fig_bt.add_annotation(x=x_end, y=y_end, text=f"Seed {seed}", showarrow=False, xanchor="left",
                              xshift=6, yanchor="bottom", yshift=6 + 16 * (len(seed_ends) - 1 - k),
                              font=dict(color=BLUE))
    # Reference lines: local‑only (dashed teal) and FedPer (dotted amber)
    local_bt_means = [v for s in seeds if (v := bt_recall(data, "local_only", s)) is not None]
    local_bt_mean = sum(local_bt_means) / len(local_bt_means) if local_bt_means else None
    if local_bt_mean is not None:
        fig_bt.add_hline(
            y=local_bt_mean,
            line_dash="dash",
            line_color=TEAL,
            annotation_text=f"Local‑only ({fmt(local_bt_mean)})",
            annotation_position="top left",
            annotation_font_color=TEAL_TEXT,
        )
    fedper_bt_means = [v for s in seeds if (v := bt_recall(data, "fedper", s)) is not None]
    fedper_bt_mean = sum(fedper_bt_means) / len(fedper_bt_means) if fedper_bt_means else None
    if fedper_bt_mean is not None:
        fig_bt.add_hline(
            y=fedper_bt_mean,
            line_dash="dot",
            line_color=AMBER,
            annotation_text=f"FedPer ({fmt(fedper_bt_mean)})",
            annotation_position="top right",
            annotation_font_color=AMBER_TEXT,
        )
    fig_bt.update_layout(
        xaxis_title="Aggregator (μ = FedProx proximal strength)",
        yaxis_title="Bluetooth benign recall",
        yaxis=dict(range=[0, 1.05]),
        hovermode="x unified",
        showlegend=False,
        height=400,
        margin=dict(l=8, r=72, t=24, b=16),
    )

    # Secondary chart: avg macro F1 per method (horizontal: long method names stay readable)
    fig_f1 = go.Figure()
    methods_for_f1 = ["fedavg"] + [f"fedprox_mu{m}" for m in mus] + ["fedper", "cluster_fl", "local_only", "centralized"]
    y_labels = []
    x_means = []
    x_errors = []  # half spread for error bars
    colors = []  # FedAvg/FedProx → BLUE, FedPer → AMBER, Cluster-FL → NAVY, Local‑only → TEAL, Centralized → GREY
    for method_key in methods_for_f1:
        if method_key == "centralized":
            # Only pooled macro F1 is stored for centralized
            seed_vals = [v for v in cent_f1.values() if v is not None]
            label = "Centralized (pooled)"
        else:
            seed_vals = [avg_macro_f1(r) for seed in seeds if (r := method_run(data, method_key, seed))]
            label = method_label(method_key)
        if seed_vals:
            mean_val, spread = mean_spread(seed_vals)
            y_labels.append(label)
            x_means.append(mean_val)
            x_errors.append(spread / 2)
            colors.append(method_color(method_key))
    fig_f1.add_trace(
        go.Bar(
            x=x_means,
            y=y_labels,
            orientation="h",
            error_x=dict(type="data", array=x_errors, visible=True, color=NAVY, thickness=1.5, width=4),
            marker_color=colors,
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    # Values sit in the right margin so they never collide with the error bars
    for label, mean_val in zip(y_labels, x_means):
        fig_f1.add_annotation(xref="paper", x=1, xanchor="left", xshift=12, yref="y", y=label,
                              text=fmt(mean_val), showarrow=False, font=dict(color=NAVY, size=12))
    fig_f1.update_layout(
        xaxis=dict(title="Avg macro F1 (seed mean)", range=[0, 1.0]),
        yaxis=dict(title=None, autorange="reversed"),
        bargap=0.35,
        height=460,
        margin=dict(l=8, r=72, t=24, b=16),
    )

    cluster_f1_means = [avg_macro_f1(r) for s in seeds if (r := method_run(data, "cluster_fl", s))]
    cluster_f1_mean = sum(cluster_f1_means) / len(cluster_f1_means) if cluster_f1_means else 0.0
    cluster_bt_means = [client_metric(r, "bluetooth", "benign_recall") for s in seeds if (r := method_run(data, "cluster_fl", s))]
    cluster_bt_mean = sum(cluster_bt_means) / len(cluster_bt_means) if cluster_bt_means else 0.0
    mu_max = mus[-1] if mus else None
    _flag_pct = sorted(
        (1 - v) * 100 for s in seeds if mu_max is not None and (v := bt_recall(data, f"fedprox_mu{mu_max}", s)) is not None
    )
    flag_range_text = f"{_flag_pct[0]:.0f}–{_flag_pct[-1]:.0f}%" if _flag_pct else "—"
    _cent_vals = [v for v in cent_f1.values() if v is not None]
    cent_f1_mean = sum(_cent_vals) / len(_cent_vals) if _cent_vals else None
    cent_gap_text = fmt(abs(local_mean - cent_f1_mean)) if cent_f1_mean is not None else "—"
    cent_f1_text = "/".join(fmt(cent_f1.get(s)) for s in seeds)
    cent_benign_text = "/".join(fmt(cent_benign.get(s)) for s in seeds)

    chart_col, text_col = st.columns([3, 2], gap="medium")
    with chart_col, card("bt_vs_mu", "Bluetooth benign recall vs μ",
                         "FedAvg and FedProx per seed (held‑out test), with local‑only and FedPer reference lines",
                         height="stretch"):
        show_chart(fig_bt, alt="Line chart of Bluetooth benign recall across FedAvg and FedProx mu values per seed")
    with text_col, card("findings_mu", "No μ saves Bluetooth", height="stretch"):
        st.markdown(
            f"Even the gentlest proximal term (μ={mu_min}) lowers Bluetooth benign recall below FedAvg "
            f"({smallest_mu_text}).\n\n"
            f"By μ={mu_max} the model flags **{flag_range_text}** of Bluetooth benign traffic as attack.\n\n"
            f"Local‑only sits at {fmt(local_bt_min)} — the collapse is caused by *sharing*, and proximal anchoring "
            "only deepens it."
        )

    chart_col, text_col = st.columns([3, 2], gap="medium")
    with chart_col, card("f1_methods", "Average macro F1 per method", "Seed mean; error bars span seed min–max",
                         height="stretch"):
        show_chart(fig_f1, alt="Horizontal bar chart of average macro F1 per method with seed range error bars")
    with text_col, card("findings", "What the comparison shows", height="stretch"):
        st.markdown(
            f"""
            - **Local‑only beats all sharing**: Local‑only achieves {fmt(local_mean)} avg macro F1 vs {fmt(best_shared_mean)} for the best shared model ({method_label(best_shared_label)}) — a {local_mean - best_shared_mean:+.4f} absolute gain. It matches the pooled centralized reference ({cent_f1_text}) within {cent_gap_text} points and beats it on Bluetooth benign recall ({fmt(local_bt_min)} vs pooled {cent_benign_text}).
            - **FedPer’s trade‑off position**: FedPer recovers {fmt(fedper_bt_mean)} Bluetooth benign recall (seed mean) — {fedper_gap_text} of the FedAvg→local‑only gap. Its avg macro F1 ({fmt(fedper_f1_mean)}) is {fedper_vs_local_text} of local‑only, and cross‑client explanation similarity remains at FedAvg level (see the Explanation Divergence tab).
            - **Cluster‑FL resolves the collapse structurally**: Protocol‑aligned clusters ({'{Wi‑Fi}'}, {'{MQTT, Bluetooth}'}) avoid inter‑protocol gradient conflict entirely — {fmt(cluster_f1_mean)} avg macro F1 with {fmt(cluster_bt_mean)} Bluetooth benign recall, without a private per‑client head.
            """
        )

# ===== TAB 4: Methodology & Limitations =====
with tab4:
    takeaway(f"{len(seeds)} seeds × {run_meta.get('n_rounds', '—')} rounds × {run_meta.get('n_local_epochs', '—')} local "
             "epochs: every result in this dashboard is a directional trend, with no p‑values or confidence intervals.")
    _mu_list = ", ".join(map(str, mus))
    meth_card = card("methodology", "Study scale & methodology")
    scale_col, methods_col = meth_card.columns([3, 2], gap="large")
    scale_col.markdown("**Setup**")
    scale_col.markdown(
        f"""
        - **Seeds**: {len(seeds)} ({', '.join(map(str, seeds))}) — directional trends only, no statistical inference.
        - **Rounds**: {run_meta.get('n_rounds', '—')} federated rounds per run.
        - **Local epochs**: {run_meta.get('n_local_epochs', '—')} per round.
        - **Features**: 45 network flow features (header, timing, TCP flags, protocol flags, statistical aggregates).
        - **Models**: Shallow 3‑layer MLP (45→64→32→1) with ReLU and dropout.
        - **Explainability**: SHAP DeepExplainer with {run_meta.get('shap_bg_size', '—')} benign background samples per client.
        - **Divergence metrics**: Jaccard similarity of Top‑5 feature sets, Spearman and cosine similarity over full 45‑dim mean‑|SHAP| vectors.
        - **Noise floor**: Seed‑to‑seed |difference| of each statistic (n={len(seeds)} seeds).
        """
    )
    methods_col.markdown("**Methods compared**")
    methods_col.markdown(
        f"""
        - **FedAvg**: Shared model, no proximal term.
        - **FedProx**: Shared model + proximal term (μ in {{{_mu_list}}}).
        - **FedPer**: Shared trunk (first 4 parameter arrays) + private per‑client output head (last 2 arrays).
        - **Local‑only**: No sharing (control).
        - **Centralized**: Pooled data (upper reference).
        - **Cluster‑FL**: Protocol‑aligned clusters ({{Wi‑Fi}}, {{MQTT, Bluetooth}}) — one shared model per cluster, no inter‑cluster sharing.
        """
    )
    verified_col, planned_col = card("verified_planned", "Verified vs. planned").columns(2, gap="large")
    with verified_col:
        st.badge("Verified", icon=":material/check_circle:", color="green")
        st.markdown(
            f"""
            - Centralized baseline benchmark
            - Multi‑client federated training with Flower (FedAvg)
            - Post‑hoc explainability pipeline (DeepExplainer + Jaccard)
            - FedProx proximal term integration (μ={_mu_list})
            - Diagnostic robustness check (Wi‑Fi class imbalance, weight norms)
            - Scaled‑up study run on `data/study` with {len(seeds)}‑seed pilot
            - Cluster‑FL port (protocol‑aligned clusters, merged from `results/cluster_fl_results.json`)
            - Interactive Streamlit research dashboard (this app)
            """
        )
    with planned_col:
        st.badge("Planned / open", icon=":material/schedule:", color="orange")
        st.markdown(
            """
            - Full 10‑seed noise‑floor calibration & logarithmic μ sweep
            - Advanced aggregators (SCAFFOLD, FedNova, FedOpt)
            - Gradient‑based explainers (Integrated Gradients, etc.)
            - Zero‑day and data‑scarcity generalization experiments (in progress; see the Zero‑Day & Data Scarcity tab)
            """
        )

    cf_card = card("cluster_fl", "Cluster‑FL results")
    _cf_runs = [(s, method_run(data, "cluster_fl", s)) for s in seeds]
    if seeds and all(r is not None for _, r in _cf_runs):
        _cf_f1 = [avg_macro_f1(r) for _, r in _cf_runs]
        _cf_bt = [client_metric(r, "bluetooth", "benign_recall") for _, r in _cf_runs]
        _cf_mu = sum(_cf_f1) / len(_cf_f1)
        _cf_bt_m = sum(_cf_bt) / len(_cf_bt)
        _cf_seed_txt = " / ".join(map(str, seeds))
        cf_card.badge(f"Merged ({len(seeds)} seeds)", icon=":material/check_circle:", color="green")
        cf_card.markdown(
            f"""
            Protocol‑aligned clusters {{Wi‑Fi}} and {{MQTT, Bluetooth}}, one shared model per cluster.

            - **Avg macro F1** (seeds {_cf_seed_txt}): {' / '.join(fmt(v) for v in _cf_f1)} — seed mean **{fmt(_cf_mu)}**
            - **Bluetooth benign recall** (seeds {_cf_seed_txt}): {' / '.join(fmt(v) for v in _cf_bt)} — seed mean **{fmt(_cf_bt_m)}**
            """
        )
        cf_card.expander("Details", icon=":material/info:").markdown(
            f"""
            - **Divergence scope**: {data.get('divergence_summary', {}).get('cluster_fl', {}).get('scope', 'across cluster‑personalized models')}
            - **Provenance**: merged from `results/cluster_fl_results.json` via `src/merge_cluster_fl.py` (no retraining; no other keys touched).
            """
        )
    else:
        cf_card.info(f"**Current status**: {CLUSTER_FL_STATUS}")
        cf_card.expander("Expected schema & merge workflow").markdown(
            """
            **Expected schema**: See `CLUSTER_FL_RUN_SCHEMA` in `src/run_study_experiments.py` (requires `aggregator="cluster_fl"`, `mu=null`, `seed`, `clusters`, `client_metrics`, `avg_macro_f1`, `shap_*` keys, `jaccard_pairwise`, `divergence_full`, `shared_features`, `divergence_scope`, plus comparison sets `cluster_fl_vs_fedavg` and `cluster_fl_vs_local_only` — these are auto‑backfilled at merge by `enrich_cluster_fl_comparisons()`).

            **Merge workflow**:
            1. Teammate drops JSON array at `results/cluster_fl_results.json` following the schema.
            2. Run `python src/merge_cluster_fl.py` from repo root.
            3. The tool validates, enriches with missing comparison sets, and merges into `study_results.json` (replacing any existing `aggregator='cluster_fl'` entries).

            No other keys in `study_results.json` are touched; no retraining is performed.
            """
        )

    # Build stability table from preliminary_noise_floor for all aggregators
    prelim = data.get("preliminary_noise_floor", {})
    agg_keys = ["fedavg"] + [f"fedprox_mu{m}" for m in mus] + ["fedper", "local_only", "cluster_fl"]
    stab_rows = []
    for agg in agg_keys:
        entry = prelim.get(agg, {})
        # Per‑client Top‑5 stability
        wifi_stab = entry.get("per_client_top5_stability", {}).get("wifi", {}).get("seed_to_seed_jaccard")
        mqtt_stab = entry.get("per_client_top5_stability", {}).get("mqtt", {}).get("seed_to_seed_jaccard")
        bt_stab = entry.get("per_client_top5_stability", {}).get("bluetooth", {}).get("seed_to_seed_jaccard")
        # Max pairwise Jaccard abs_diff
        pairwise_spread = entry.get("pairwise_divergence_spread", {})
        max_pairwise = max([v["abs_diff"] for v in pairwise_spread.values() if isinstance(v, dict) and v.get("abs_diff") is not None], default=None)
        # Status
        status = entry.get("status", "unknown")
        stab_rows.append({
            "Aggregator": method_label(agg),
            "Wi‑Fi Top‑5 Jaccard": wifi_stab,
            "MQTT Top‑5 Jaccard": mqtt_stab,
            "Bluetooth Top‑5 Jaccard": bt_stab,
            "Max pairwise Jaccard |diff|": max_pairwise,
            "Status": status
        })
    df_stab = pd.DataFrame(stab_rows)
    # A status shared by every row is shown once as a caption instead of a repeated column.
    _statuses = df_stab["Status"].unique()
    shared_status = _statuses[0] if len(_statuses) == 1 else None
    if shared_status is not None:
        df_stab = df_stab.drop(columns="Status")
    _jac = st.column_config.ProgressColumn(format="%.4f", min_value=0.0, max_value=1.0)
    stab_card = card("stability", "Seed‑to‑seed stability table (all aggregators)", "Top‑5 Jaccard between seeds per client; bars span 0–1")
    stab_card.dataframe(
        df_stab,
        hide_index=True,
        column_config={
            "Aggregator": st.column_config.TextColumn(pinned=True),
            "Wi‑Fi Top‑5 Jaccard": _jac,
            "MQTT Top‑5 Jaccard": _jac,
            "Bluetooth Top‑5 Jaccard": _jac,
            "Max pairwise Jaccard |diff|": st.column_config.NumberColumn(format="%.4f"),
        },
        height=(len(df_stab) + 1) * 35 + 3,
        width="stretch",
        alt="Seed-to-seed top-5 Jaccard stability per client for every aggregator",
    )
    if shared_status is not None:
        stab_card.caption(f"Status (all aggregators): {shared_status}")

    arch_card = card("architecture", "System architecture diagram")
    if os.path.exists(ARCH_IMG_PATH):
        arch_card.image(ARCH_IMG_PATH, width="stretch", alt="FedXplain-IoMT system architecture diagram")
        arch_card.caption("Figure 1: End‑to‑end FedXplain‑IoMT system architecture.")
    else:
        arch_card.warning(f"Architecture diagram not found: {ARCH_IMG_PATH}")

# ===== TAB 5: Zero-Day & Data Scarcity =====
with tab5:
    if gen_data is None:
        st.info(
            f"Generalization results not generated yet (`{GENERALIZATION_PATH}` missing). "
            "Run `python -m src.run_generalization_study` from the repo root."
        )
    else:
        takeaway("FL is justified here only where its zero‑day recall exceeds local‑only with a positive detection "
                 "margin, and where the head‑only curve (especially the trunk trained without the target client) "
                 "stays above local‑only at small N.")
        GEN_COLORS = {"local_only": TEAL, "fedavg": BLUE, "fedper": AMBER, "cluster_fl": NAVY,
                      "fedper_head": AMBER, "fedper_head_unseen": AMBER}
        # Both FedPer-head arms stay in the FedPer amber; the trunk-without-target arm is dashed.
        GEN_DASH = {"fedper_head_unseen": "dash"}
        GEN_LABELS = {"fedper_head": "FedPer head (trunk saw target data)",
                      "fedper_head_unseen": "FedPer head (trunk without target)"}
        _gen_seeds = gen_data.get("zero_day", {}).get("config", {}).get("seeds", seeds)

        # ---- Experiment 1: Zero-Day ----
        zd_card = card("zero_day", "Experiment 1 — Zero‑day (novel attack) detection", "Attack recall on attack types never seen in the target client's local training set")
        zd = gen_data.get("zero_day", {})
        zd_targets = list(zd.get("targets", {}).keys())
        zd_target = zd_card.segmented_control(
            "Target client",
            options=zd_targets,
            format_func=lambda p: PROTO_LABEL.get(p, p),
            default=zd_targets[0] if zd_targets else None,
            required=bool(zd_targets),
            key="zd_target",
            help="Attacks never seen in its local training set",
        )
        zd_entry = zd.get("targets", {}).get(zd_target, {})
        zd_summary = zd_entry.get("summary", {})
        zd_methods = [m for m in zd.get("config", {}).get("methods", []) if m in zd_summary]
        attacks = sorted(
            zd_entry.get("unseen_attacks", {}).keys(),
            key=lambda a: (zd_entry["unseen_attacks"][a].get("source", ""), a),
        )

        # Horizontal bars keep the (long) attack names readable instead of rotated x-ticks.
        fig_zd = go.Figure()
        for m in zd_methods:
            per_attack = zd_summary[m].get("per_attack", {})
            fig_zd.add_trace(go.Bar(
                name=GEN_LABELS.get(m, method_label(m)),
                y=attacks,
                x=[per_attack.get(a, {}).get("mean") for a in attacks],
                orientation="h",
                marker_color=GEN_COLORS.get(m, GREY),
                hovertemplate="%{x:.4f}",
            ))
        fig_zd.update_layout(
            barmode="group",
            yaxis=dict(title="Unseen attack type (from other protocols' held‑out test splits)", autorange="reversed"),
            xaxis=dict(title="Attack recall (seed mean)", range=[0, 1.05]),
            bargap=0.2,
            hovermode="y unified",
            height=max(360, 120 + len(attacks) * (7 * max(len(zd_methods), 1) + 10)),
        )
        with zd_card:
            show_chart(fig_zd, alt=f"Grouped horizontal bar chart of per-attack recall on unseen attacks for the {PROTO_LABEL.get(zd_target, zd_target)} client")

        zd_rows = []
        for m in zd_methods:
            s = zd_summary[m]
            ind = s.get("in_distribution", {})
            row = {
                "Method": GEN_LABELS.get(m, method_label(m)),
                "Mean unseen‑attack recall": s.get("mean_unseen_attack_recall"),
                "Detection margin": s.get("mean_detection_margin"),
            }
            for src, rate in s.get("foreign_benign_flag_rate", {}).items():
                row[f"{PROTO_LABEL.get(src, src)} benign flagged"] = rate
            row["In‑dist macro F1"] = ind.get("f1_macro")
            row["In‑dist benign recall"] = ind.get("benign_recall")
            zd_rows.append(row)
        df_zd = pd.DataFrame(zd_rows)
        zd_card.dataframe(
            df_zd,
            hide_index=True,
            column_config={
                c: st.column_config.NumberColumn(format="%.4f", help="Attack recall − foreign benign flag rate")
                if c == "Detection margin"
                else st.column_config.ProgressColumn(format="%.4f", min_value=0.0, max_value=1.0)
                for c in df_zd.columns if c != "Method"
            } | {"Method": st.column_config.TextColumn(pinned=True)},
            width="stretch",
            alt=f"Zero-day recall, detection margin and in-distribution metrics per method for the {PROTO_LABEL.get(zd_target, zd_target)} client",
        )
        zd_card.expander("How to read this table", icon=":material/help:").markdown(
            "Foreign inputs are standardized with the target client's own scaler. "
            "'Benign flagged' is the control: if foreign benign traffic is flagged as often as foreign attacks, "
            "the model is reacting to distribution shift, not recognising the attack. "
            "Detection margin (attack − foreign benign) > 0 means attacks are flagged more often than benign traffic from the same protocol."
        )

        # ---- Experiment 2: Data Scarcity ----
        ds_card = card("scarcity", "Experiment 2 — Small‑clinic sample efficiency", "Macro F1 as the target clinic's training set shrinks (mean ± std over seeds × repeats)")
        ds_all = gen_data.get("data_scarcity", {})
        # Older single-target files store one experiment at the top level.
        ds_targets = ds_all.get("targets") or ({ds_all.get("config", {}).get("target", "bluetooth"): ds_all} if ds_all else {})
        ds_target = ds_card.segmented_control(
            "Target client",
            options=list(ds_targets.keys()),
            format_func=lambda p: PROTO_LABEL.get(p, p),
            default=next(iter(ds_targets), None),
            required=bool(ds_targets),
            key="ds_target",
            help="The small clinic whose training set is subsampled",
        )
        ds = ds_targets.get(ds_target, {})
        ds_summary = ds.get("summary", {})
        fig_ds = go.Figure()
        for arm, points in ds_summary.items():
            fig_ds.add_trace(go.Scatter(
                name=GEN_LABELS.get(arm, method_label(arm)),
                x=[p["n_train"] for p in points],
                y=[p["f1_macro_mean"] for p in points],
                error_y=dict(type="data", array=[p["f1_macro_std"] for p in points], visible=True),
                mode="lines+markers",
                line=dict(color=GEN_COLORS.get(arm, GREY), dash=GEN_DASH.get(arm, "solid")),
                marker=dict(symbol="circle-open" if arm in GEN_DASH else "circle", size=7),
                hovertemplate="%{y:.4f}",
            ))
        _n_ticks = sorted({p["n_train"] for pts in ds_summary.values() for p in pts})
        fig_ds.update_layout(
            # Label only the sampled N values; Plotly's default log ticks crowd the axis with minor labels
            xaxis=dict(title="Training samples available at the clinic (N, log scale)", type="log",
                       tickvals=_n_ticks, ticktext=[f"{n:,}" for n in _n_ticks]),
            yaxis=dict(title="Macro F1 on full held‑out test set", range=[0, 1.05]),
            hovermode="x unified",
            height=420,
        )
        # Smallest-N value per arm (equal-width cards), then the full curve
        _ds_arms = [(arm, pts) for arm, pts in ds_summary.items() if pts]
        for col, (arm, points) in zip(ds_card.columns(max(len(_ds_arms), 1), gap="medium"), _ds_arms):
            col.metric(
                GEN_LABELS.get(arm, method_label(arm)),
                fmt(points[0]["f1_macro_mean"]),
                f"± {points[0]['f1_macro_std']:.4f}", delta_color="off", delta_arrow="off",
                delta_description=f"std at N={points[0]['n_train']}",
                border=True,
                help=f"Macro F1 at the smallest training set (N={points[0]['n_train']}), mean ± std over seeds × repeats",
            )
        with ds_card:
            show_chart(fig_ds, alt=f"Line chart of macro F1 versus training samples for the {PROTO_LABEL.get(ds_target, ds_target)} client")
        ds_cfg = ds.get("config", {})
        ds_card.expander("Method details", icon=":material/science:").markdown(
            f"Seeds {ds_cfg.get('seeds')} × {ds_cfg.get('repeats_per_seed')} stratified draws per N; "
            f"{ds_cfg.get('epochs')} epochs with a floor of {ds_cfg.get('min_optimizer_steps')} optimizer steps for every arm; "
            "scaler refit on the N‑sample subset.\n\n"
            + "\n".join(f"- **{GEN_LABELS.get(k, method_label(k))}**: {v}." for k, v in ds.get("methods", {}).items())
        )

        # ---- Takeaway (computed from the JSON, not hard-coded) ----
        tk_card = card("takeaway", "Results summary", "Computed from the generalization results file for the selected targets")
        tk_card.badge(f"Directional only (n={len(_gen_seeds)} seeds)", icon=":material/info:", color="orange")
        lines = []
        if zd_summary:
            parts = ", ".join(
                f"{GEN_LABELS.get(m, method_label(m))} {fmt(zd_summary[m].get('mean_unseen_attack_recall'))} "
                f"(margin {fmt(zd_summary[m].get('mean_detection_margin'))})"
                for m in zd_methods
            )
            lines.append(f"- **Zero‑day ({PROTO_LABEL.get(zd_target, zd_target)})** — mean recall on unseen attacks: {parts}.")
        if ds_summary:
            def _at(arm: str, idx: int) -> Optional[Dict]:
                pts = ds_summary.get(arm, [])
                return pts[idx] if len(pts) > abs(idx) else None
            smallest = _at("local_only", 0)
            if smallest:
                parts = ", ".join(
                    f"{GEN_LABELS.get(arm, method_label(arm))} {fmt(_at(arm, 0)['f1_macro_mean'])}"
                    for arm in ds_summary if _at(arm, 0)
                )
                lines.append(f"- **Data scarcity ({PROTO_LABEL.get(ds_target, ds_target)})** — macro F1 at N={smallest['n_train']}: {parts}.")
        tk_card.markdown("\n".join(lines) if lines else "No results available.")

# ---- Footer ----
st.space("medium")
st.caption(
    "FedXplain‑IoMT • Experimental research framework • "
    f"Dashboard rendered with Python {'.'.join(map(str, sys.version_info[:3]))} • "
    f"Study results timestamp: {run_meta.get('timestamp', '—')}"
)