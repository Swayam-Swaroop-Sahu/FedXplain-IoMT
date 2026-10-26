import html
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
SURFACE = "#F5F5F2"
HIGHLIGHT = "#E76F51"
CARD_BORDER = "#E5E5DE"
MUTED = "#5B6B7A"
GREY = "#8896A6"
GRID = "#D8D8D2"
FONT_STACK = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
MONO_STACK = 'ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace'

# Charts are rendered with theme=None, so this template fully controls their look
# (otherwise Streamlit's dark theme paints light text onto the white plot backgrounds).
_AXIS = dict(
    gridcolor="#ECECE6", linecolor=GRID, zerolinecolor=GRID, automargin=True,
    tickfont=dict(color=MUTED), title=dict(font=dict(color=MUTED, size=12)),
)
pio.templates["fedxplain"] = go.layout.Template(pio.templates["plotly_white"])
pio.templates["fedxplain"].layout.update(
    font=dict(family=FONT_STACK, color=NAVY, size=12),
    title=dict(font=dict(color=NAVY, size=14), x=0, xanchor="left"),
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=8, r=8, t=44, b=16),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                font=dict(color=MUTED), bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(bgcolor="white", bordercolor=CARD_BORDER, font=dict(family=FONT_STACK, color=NAVY)),
    xaxis=_AXIS,
    yaxis=_AXIS,
)
pio.templates.default = "fedxplain"


def show_chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})

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

def kpi_card(label: str, value: str, caption: str, delta: Optional[str] = None, good: bool = True,
             tip: Optional[str] = None) -> str:
    """Return HTML for a KPI card; `delta` renders as a green (good) or red (bad) chip, `tip` as a hover tooltip."""
    main, _, spread = value.partition(" ± ")
    spread_html = f'<span class="kpi-spread">± {spread}</span>' if spread else ""
    chip = f'<span class="chip chip-{"good" if good else "bad"}">{delta}</span>' if delta else ""
    title = f' title="{html.escape(tip, quote=True)}"' if tip else ""
    return (
        f'<div class="kpi-card"{title}><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{main}{spread_html}</div>'
        f'<div class="kpi-foot">{chip}<span class="kpi-caption">{caption}</span></div></div>'
    )

def pill(text: str, tone: str = "neutral") -> str:
    """Return HTML for a small rounded badge (tone: neutral, good, warn)."""
    return f'<span class="pill pill-{tone}">{html.escape(str(text))}</span>'

def callout(body: str, kind: str = "info", title: Optional[str] = None) -> None:
    """Render a callout (kind: info, warning, success); `body` is trusted HTML."""
    head = f'<div class="callout-title">{title}</div>' if title else ""
    st.markdown(f'<div class="callout callout-{kind}">{head}{body}</div>', unsafe_allow_html=True)

def card(key: str, title: str, subtitle: Optional[str] = None):
    """Bordered card container with its header inside; use with `with`."""
    box = st.container(border=True, key=f"card_{key}")
    sub = f'<div class="card-sub">{subtitle}</div>' if subtitle else ""
    box.markdown(f'<div class="card-head"><div class="card-title">{title}</div>{sub}</div>', unsafe_allow_html=True)
    return box

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
        /* Hide Streamlit chrome, but keep the header: it hosts the expand-sidebar button */
        header[data-testid="stHeader"] {{background: transparent;}}
        [data-testid="stAppDeployButton"], [data-testid="stMainMenu"] {{display: none;}}
        footer {{visibility: hidden;}}

        /* Base & typography */
        .stApp {{ background-color: {SURFACE}; color: {NAVY}; font-family: {FONT_STACK}; }}
        .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3 {{ font-family: {FONT_STACK}; }}
        [data-testid="stMainBlockContainer"] {{ max-width: 1360px; padding: 3.5rem 2.5rem 3rem; }}
        [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{ line-height: 1.6; }}
        [data-testid="stMarkdownContainer"] li {{ margin-bottom: 0.55rem; }}
        [data-testid="stMarkdownContainer"] li:last-child {{ margin-bottom: 0; }}
        [data-testid="stMarkdownContainer"] ul ul {{ margin-top: 0.5rem; }}
        /* Readable measure for prose; charts and tables keep full width */
        [data-testid="stMarkdownContainer"] > p, [data-testid="stMarkdownContainer"] > ul {{ max-width: 78ch; }}
        [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{
            color: {MUTED}; opacity: 1; font-size: 0.82rem; line-height: 1.55;
        }}
        [data-testid="stWidgetLabel"] {{ margin-bottom: 0.25rem; }}
        [data-testid="stWidgetLabel"] p {{ font-size: 0.78rem; font-weight: 600; color: {MUTED}; }}
        [data-testid="stSelectbox"] div:has(> input) {{ background: #FAFAF7; border-color: {CARD_BORDER}; border-radius: 8px; }}
        hr {{ border-color: {CARD_BORDER} !important; }}
        code {{ font-family: {MONO_STACK}; }}
        .js-plotly-plot .main-svg text {{ font-variant-numeric: tabular-nums; }}

        /* Sidebar */
        [data-testid="stSidebar"] {{ background-color: #FFFFFF; border-right: 1px solid {CARD_BORDER}; }}
        .side-title {{
            font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em;
            color: {MUTED}; margin: 0.25rem 0 0.4rem;
        }}
        .kv {{
            display: flex; justify-content: space-between; align-items: baseline; gap: 0.75rem;
            font-size: 0.78rem; padding: 0.32rem 0; border-bottom: 1px solid #EFEFEA;
        }}
        .kv:last-child {{ border-bottom: none; }}
        .kv-k {{ color: {MUTED}; white-space: nowrap; }}
        .kv-v {{ color: {NAVY}; font-family: {MONO_STACK}; font-size: 0.74rem; text-align: right; overflow-wrap: anywhere; }}
        .side-note {{ font-size: 0.78rem; color: {MUTED}; line-height: 1.45; margin-top: 0.45rem; }}

        /* Hero header */
        .hero {{ padding: 0 0 1.5rem; }}
        .hero-eyebrow {{ font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #1D6F65; }}
        .hero-title {{ font-size: 2rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.2; color: {NAVY}; margin: 0.15rem 0 0.2rem; }}
        .hero-sub {{ font-size: 1rem; color: {MUTED}; margin-bottom: 1.1rem; }}
        .pill-row {{ display: flex; flex-wrap: wrap; gap: 0.5rem; }}
        .pill {{
            display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.22rem 0.7rem;
            border: 1px solid {CARD_BORDER}; border-radius: 999px; background: #FFFFFF; color: {NAVY};
            font-size: 0.75rem; font-weight: 500; line-height: 1.45; white-space: nowrap; font-variant-numeric: tabular-nums;
        }}
        .pill-good {{ background: #E8F5F3; border-color: #BFE3DE; color: #1D6F65; }}
        .pill-warn {{ background: #FFF4E0; border-color: #F2D8A7; color: #8A5A00; }}
        .pill-good::before, .pill-warn::before {{ content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; }}

        /* Tabs: underline style (Streamlit 1.65 renders [data-testid="stTab"] with role="tab") */
        .stTabs [role="tablist"] {{ gap: 0; }}
        .stTabs [data-testid="stTab"] {{
            background: transparent; border-radius: 0; margin: 0; padding: 0.55rem 0.7rem;
            border-bottom: 2px solid transparent; color: {MUTED}; transition: color 0.15s, border-color 0.15s;
        }}
        .stTabs [data-testid="stTab"] p {{ color: inherit; font-size: 0.84rem; font-weight: 600; white-space: nowrap; }}
        .stTabs [data-testid="stTab"]:hover {{ color: {NAVY}; }}
        .stTabs [data-testid="stTab"][aria-selected="true"] {{ color: {NAVY}; border-bottom-color: {TEAL}; }}
        /* Replace Streamlit's sliding indicator with the static border above */
        .stTabs .react-aria-SelectionIndicator {{ display: none; }}
        [data-testid="stTabPanel"] {{ padding-top: 1.75rem; }}
        /* Section rhythm: larger gap between top-level blocks than inside cards */
        [data-testid="stTabPanel"] > [data-testid="stVerticalBlock"] {{ gap: 1.75rem; }}

        /* Cards (st.container(border=True, key="card_*")) */
        [class*="st-key-card_"] {{
            background-color: #FFFFFF; border: 1px solid {CARD_BORDER} !important; border-radius: 12px !important;
            box-shadow: 0 1px 2px rgba(15, 42, 67, 0.04), 0 1px 3px rgba(15, 42, 67, 0.06);
            padding: 1.5rem 1.5rem 1.5rem !important; gap: 1.1rem;
        }}
        .card-title {{ font-size: 1.02rem; font-weight: 600; letter-spacing: -0.01em; color: {NAVY}; }}
        .card-sub {{ font-size: 0.84rem; line-height: 1.5; color: {MUTED}; margin-top: 0.3rem; }}
        .card-head {{ padding-bottom: 0.6rem; }}
        /* The PNG has large baked-in fonts; cap its width so it doesn't dominate the page */
        .st-key-card_fig4 img {{ max-width: 920px !important; display: block; margin: 0 auto; }}
        .sub-head {{ font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: {MUTED}; margin-bottom: 0.6rem; }}
        [data-testid="stExpander"] details {{ border: 1px solid {CARD_BORDER}; border-radius: 8px; background: #FCFCFA; }}
        [data-testid="stExpander"] summary p {{ font-size: 0.82rem; font-weight: 600; color: {MUTED}; }}
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] li {{ font-size: 0.86rem; color: #33475B; }}
        [data-testid="stDataFrame"] {{ font-variant-numeric: tabular-nums; }}

        /* KPI cards */
        .kpi-card {{
            display: flex; flex-direction: column; gap: 0.5rem; min-height: 156px; height: 100%;
            background-color: #FFFFFF; border: 1px solid {CARD_BORDER}; border-radius: 12px; padding: 1.2rem 1.3rem;
            box-shadow: 0 1px 2px rgba(15, 42, 67, 0.04), 0 1px 3px rgba(15, 42, 67, 0.06);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }}
        .kpi-card:hover {{ transform: translateY(-2px); box-shadow: 0 6px 16px rgba(15, 42, 67, 0.08); }}
        .kpi-label {{ font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: {MUTED}; }}
        .kpi-value {{
            font-size: 1.75rem; font-weight: 650; letter-spacing: -0.02em; line-height: 1.15; color: {NAVY};
            font-variant-numeric: tabular-nums; white-space: nowrap;
        }}
        .kpi-spread {{ font-size: 0.95rem; font-weight: 500; color: {MUTED}; margin-left: 0.35rem; }}
        .kpi-foot {{ margin-top: auto; display: flex; flex-wrap: wrap; align-items: center; gap: 0.45rem 0.5rem; }}
        .kpi-caption {{ font-size: 0.78rem; line-height: 1.4; color: {MUTED}; }}
        .chip {{
            display: inline-block; padding: 0.1rem 0.45rem; border-radius: 6px; font-size: 0.74rem; font-weight: 600;
            font-variant-numeric: tabular-nums; white-space: nowrap;
        }}
        .chip-good {{ background: #E6F4EA; color: #1E6B3A; }}
        .chip-bad {{ background: #FCEDE5; color: #9A3412; }}

        /* Callouts (custom HTML + native st.info/success/warning share one look) */
        .callout {{
            border: 1px solid; border-left-width: 4px; border-radius: 8px; padding: 1.1rem 1.35rem;
            font-size: 0.9rem; line-height: 1.6;
        }}
        .callout-title {{ font-weight: 600; margin-bottom: 0.4rem; }}
        .callout ul {{ margin: 0.5rem 0 0.6rem 1.1rem; padding: 0; }}
        .callout li {{ margin-bottom: 0.2rem; }}
        .callout-info {{ background: #EEF5FC; border-color: #CFE0F3; border-left-color: {BLUE}; color: #163E66; }}
        .callout-warning {{ background: #FDF1EA; border-color: #F6D5C3; border-left-color: {HIGHLIGHT}; color: #7C2D12; }}
        .callout-success {{ background: #E9F6F4; border-color: #C4E6E1; border-left-color: {TEAL}; color: #145049; }}
        [data-testid="stAlertContainer"] {{
            border: 1px solid #CFE0F3; border-left: 4px solid {BLUE}; border-radius: 8px;
            background: #EEF5FC !important; color: #163E66 !important;
        }}
        [data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {{
            border-color: #C4E6E1; border-left-color: {TEAL}; background: #E9F6F4 !important; color: #145049 !important;
        }}
        [data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {{
            border-color: #F6D5C3; border-left-color: {HIGHLIGHT}; background: #FDF1EA !important; color: #7C2D12 !important;
        }}
        [data-testid="stAlertContainer"] p, [data-testid="stAlertContainer"] li {{ color: inherit; }}

        /* Thesis lead */
        .thesis {{
            font-size: 1.02rem; line-height: 1.6; color: {NAVY}; background: #FFFFFF;
            border: 1px solid {CARD_BORDER}; border-left: 4px solid {TEAL}; border-radius: 12px;
            padding: 1.1rem 1.5rem;
        }}
        .footer {{
            margin-top: 2.5rem; padding-top: 1.25rem; border-top: 1px solid {CARD_BORDER};
            font-size: 0.78rem; color: {MUTED};
        }}
        /* Offset Markdown's trailing -1rem margin so HTML blocks keep the 1rem block gap */
        .kpi-card, .thesis, .callout {{ margin-bottom: 1rem; }}
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
def _kv_rows(items) -> str:
    return "".join(
        f'<div class="kv"><span class="kv-k">{html.escape(str(k))}</span>'
        f'<span class="kv-v">{html.escape(", ".join(map(str, v)) if isinstance(v, list) else str(v))}</span></div>'
        for k, v in items
    )

with st.sidebar:
    st.markdown('<div class="side-title">About this run</div>', unsafe_allow_html=True)
    if run_meta:
        st.markdown(_kv_rows((k.replace("_", " "), v) for k, v in run_meta.items()), unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="side-title">Study</div>', unsafe_allow_html=True)
    st.markdown(
        _kv_rows([
            ("Benchmark", "CICIoMT2024"),
            ("Protocols", "Wi‑Fi, MQTT, Bluetooth"),
            ("Seeds", f"{', '.join(map(str, seeds))} (directional only)"),
            ("Mode", "Read‑only, runs fully offline"),
        ]),
        unsafe_allow_html=True
    )
    st.markdown("---")
    _cluster_fl_merged = bool(seeds) and all(
        find_run(data.get("runs", []), "cluster_fl", s) is not None for s in seeds
    )
    st.markdown('<div class="side-title">Cluster‑FL status</div>', unsafe_allow_html=True)
    if _cluster_fl_merged:
        st.markdown(
            pill("Merged", "good")
            + '<div class="side-note">Results in Detection, Explanation Divergence and Methods Compared</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(pill(CLUSTER_FL_STATUS, "warn"), unsafe_allow_html=True)

# ---- Header ----
_hero_pills = [
    pill("CICIoMT2024"),
    pill(f"{len(PROTOS)} protocols · " + " / ".join(PROTO_LABEL[p] for p in PROTOS)),
    pill(f"Seeds {', '.join(map(str, seeds))}"),
]
if run_meta.get("n_rounds") is not None:
    _hero_pills.append(pill(f"{run_meta['n_rounds']} rounds × {run_meta.get('n_local_epochs', '—')} local epochs"))
if run_meta.get("timestamp"):
    _hero_pills.append(pill(f"Results {run_meta['timestamp']}"))
_hero_pills.append(pill("Offline · read‑only", "good"))
st.markdown(
    '<div class="hero"><div class="hero-eyebrow">Research dashboard</div>'
    '<div class="hero-title">FedXplain-IoMT</div>'
    '<div class="hero-sub">Federated Learning &amp; Explainable AI for Heterogeneous Medical IoT</div>'
    f'<div class="pill-row">{"".join(_hero_pills)}</div></div>',
    unsafe_allow_html=True
)

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
    st.markdown('<div class="thesis">One shared detector cannot serve three structurally different hospital protocols: on this benchmark the shared model collapses on Bluetooth, no proximal strength rescues it, and only personalizing the final layer — or not sharing at all — does.</div>', unsafe_allow_html=True)

    # KPI cards row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4, gap="medium")

    # KPI 1: Best shared-model avg macro F1
    shared_methods = ["fedavg"] + [f"fedprox_mu{m}" for m in mus]
    best_shared_mean = -1.0
    best_shared_label = ""
    for method in shared_methods:
        seed_vals = []
        for seed in seeds:
            run = method_run(data, method, seed)
            if run:
                seed_vals.append(avg_macro_f1(run))
        if seed_vals:
            mean_val, _ = mean_spread(seed_vals)
            if mean_val > best_shared_mean:
                best_shared_mean = mean_val
                best_shared_label = method
    kpi1_value = fmt(best_shared_mean)
    kpi1_caption = method_label(best_shared_label)
    with kpi1:
        st.markdown(
            kpi_card("BEST SHARED-MODEL F1", kpi1_value, kpi1_caption,
                     tip=f"Best shared-model: {method_label(best_shared_label)} (avg macro F1, seed mean)"),
            unsafe_allow_html=True
        )

    # KPI 2: Local-only avg macro F1 + delta vs best shared
    local_seed_vals = []
    for seed in seeds:
        run = local_entry(data, seed)
        if run:
            local_seed_vals.append(avg_macro_f1(run))
    local_mean, local_spread = mean_spread(local_seed_vals)
    kpi2_value = f"{fmt(local_mean)} ± {local_spread:.4f}"
    kpi2_delta = f"{local_mean - best_shared_mean:+.4f} vs best shared"
    kpi2_caption = f"{len(seeds)}-seed mean"
    with kpi2:
        # Higher macro F1 than the best shared model is the good direction.
        st.markdown(
            kpi_card("LOCAL-ONLY F1", kpi2_value, kpi2_caption, kpi2_delta, local_mean >= best_shared_mean,
                     tip=f"{kpi2_delta} model ({method_label(best_shared_label)})"),
            unsafe_allow_html=True
        )

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
    kpi3_value = fmt(worst_bt_recall)
    kpi3_caption = f"FedProx μ={worst_mu}, seed {worst_seed}"
    with kpi3:
        st.markdown(
            kpi_card("BT BENIGN RECALL (WORST)", kpi3_value, kpi3_caption,
                     tip=f"Worst FedProx (μ={worst_mu}, seed={worst_seed}) vs local-only min {fmt(local_bt_min)}"),
            unsafe_allow_html=True
        )

    # KPI 4: FedAvg Wi‑Fi–Bluetooth Spearman (two‑seed mean)
    spear_seeds = []
    for seed in seeds:
        run = method_run(data, "fedavg", seed)
        if run:
            div_full = run.get("divergence_full", {})
            spear = div_full.get("wifi-bluetooth", {}).get("spearman")
            if spear is not None:
                spear_seeds.append(spear)
    kpi4_value = fmt_pm(spear_seeds)
    kpi4_caption = f"FedAvg · {len(seeds)}‑seed mean ± spread"
    with kpi4:
        st.markdown(
            kpi_card("WI‑FI–BT SPEARMAN", kpi4_value, kpi4_caption,
                     tip=f"FedAvg Wi‑Fi–Bluetooth Spearman ρ ({len(seeds)}‑seed mean ± spread)"),
            unsafe_allow_html=True
        )

    # Figure 4 (the image carries its own title and error-bar note)
    with card("fig4", "Figure 4 · Personalization trade‑off"):
        if os.path.exists(FIG4_PATH):
            st.image(FIG4_PATH, width="stretch")
        else:
            st.warning(f"Figure not found: {FIG4_PATH}")

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

    # Two‑column findings
    col1, col2 = st.columns(2, gap="medium")
    with col1, card("found", "What we found"):
        st.markdown(
            f"""
            - **Bluetooth collapse**: Shared models drop BT benign recall to as low as {fmt(worst_bt_recall)} (μ={worst_mu}, seed {worst_seed}), while local‑only stays at {fmt(local_bt_min)}.
            - **No safe μ**: Even μ={mu_min} worsens BT benign recall vs FedAvg ({smallest_mu_text}).
            - **FedPer’s trade‑off**: Recovers {fedper_gap_text} of the FedAvg→local‑only BT benign‑recall gap, with avg macro F1 {fmt(fedper_f1_mean)} ({fedper_vs_local_text} of local‑only {fmt(local_mean)}).
            - **Explanation consistency**: FedPer’s cross‑client similarity stays at FedAvg level (see the Explanation Divergence tab).
            """
        )
    with col2, card("not_proven", "What we didn’t prove yet"):
        st.markdown(
            """
            - **Statistical significance**: n=2 seeds provide directional trends only; no p‑values or confidence intervals.
            - **Full noise floor**: Preliminary (n=2) only; more seeds needed for robust variance estimates.
            - **Other aggregators**: SCAFFOLD, FedNova, etc. remain for future work.
            """
        )

# ===== TAB 1: Detection =====
with tab1:
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
    with card("unified", "Unified detection table (macro F1)", "Per‑client macro F1 by seed, then per‑seed averages"):
        st.dataframe(
            df_unified,
            hide_index=True,
            height=(len(df_unified) + 1) * 35 + 3,
            width="stretch"
        )
        st.caption("Centralized stores only pooled metrics: per‑client cells are empty and its Avg columns show pooled macro F1.")

    # Build BT benign recall table with styling
    bt_rows = []
    methods = ["fedavg"] + [f"fedprox_mu{m}" for m in mus] + ["fedper", "cluster_fl", "local_only", "centralized"]
    for method_key in methods:
        if method_key == "centralized":
            row = {"Method": "Centralized (pooled, all protocols)"}
            row.update({str(s): cent_benign.get(s) for s in seeds})
        else:
            row = {"Method": method_label(method_key)}
            row.update({str(s): bt_recall(data, method_key, s) for s in seeds})
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
        except (TypeError, ValueError):
            return ""
    # Apply styling to seed columns only
    seed_cols = [str(s) for s in seeds]
    styled_bt = df_bt.style.format(
        {c: lambda x: fmt(x) if isinstance(x, (float, int)) else x for c in df_bt.columns if c != "Method"}
    ).map(highlight_bad, subset=seed_cols)
    bt_col, callout_col = st.columns([3, 2], gap="medium")
    with bt_col, card("bt_recall", "Bluetooth benign‑recall table", "Values below 0.5 highlighted"):
        st.dataframe(styled_bt, hide_index=True, height=(len(df_bt) + 1) * 35 + 3, width="stretch")
        st.caption("< 0.5 means >50% false positives on benign traffic. Centralized stores only pooled benign recall, not a Bluetooth‑specific value.")

    # Bluetooth collapse callout
    bt_by_seed = {k: " / ".join(fmt(bt_recall(data, k, s)) for s in seeds) for k in ("fedavg", "fedper", "local_only")}
    with callout_col:
        callout(
            f"BT benign recall, seeds {' / '.join(map(str, seeds))}"
            "<ul>"
            f"<li>FedAvg: {bt_by_seed['fedavg']}</li>"
            f"<li>Worst FedProx (μ={worst_mu}, seed {worst_seed}): {fmt(worst_bt_recall)}</li>"
            f"<li>FedPer: {bt_by_seed['fedper']}</li>"
            f"<li>Local‑only: {bt_by_seed['local_only']}</li>"
            "</ul>"
            "The collapse is driven by sharing; the proximal term exacerbates it. "
            "Personalizing the final layer (FedPer) recovers most of the gap.",
            kind="warning",
            title="Bluetooth collapse"
        )

    det_card = card("det_detail", "Detailed per‑seed metrics", "Binary F1 and benign recall per client; collapsed rows highlighted")
    # Seed selector
    det_seed = det_card.columns([1, 4])[0].selectbox(
        "Seed",
        options=seeds,
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
            "Wi‑Fi binary F1": fmt(client_metric(run, "wifi", "f1")),
            "Wi‑Fi benign recall": fmt(client_metric(run, "wifi", "benign_recall")),
            "MQTT binary F1": fmt(client_metric(run, "mqtt", "f1")),
            "MQTT benign recall": fmt(client_metric(run, "mqtt", "benign_recall")),
            "Bluetooth binary F1": fmt(client_metric(run, "bluetooth", "f1")),
            "Bluetooth benign recall": fmt(client_metric(run, "bluetooth", "benign_recall")),
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
    det_card.dataframe(styled_det, hide_index=True, height=(len(df_det) + 1) * 35 + 3, width="stretch")
    _wifi_att = client_metric(method_run(data, "fedavg", det_seed), "wifi", "n_attack")
    _wifi_ben = client_metric(method_run(data, "fedavg", det_seed), "wifi", "n_benign")
    wifi_attack_share = f"{_wifi_att / (_wifi_att + _wifi_ben) * 100:.1f}%" if _wifi_att and _wifi_ben is not None else "—"
    det_card.caption(f"Wi‑Fi binary F1 is inflated by class imbalance ({wifi_attack_share} attack samples in the Wi‑Fi test split); macro F1 and per‑class precision/recall are used throughout.")

# ===== TAB 2: Explanation Divergence =====
with tab2:
    # Method and seed selectors
    div_methods = ["fedavg"] + [f"fedprox_mu{m}" for m in mus] + ["fedper", "local_only", "cluster_fl"]
    with card("div_select", "Explanation divergence across clients", "Choose a method and seed to inspect its explanations"):
        sel_method, sel_seed, _ = st.columns([2, 1, 3])
        div_method = sel_method.selectbox(
            "Method",
            options=div_methods,
            format_func=method_label,
            key="div_method"
        )
        div_seed = sel_seed.selectbox(
            "Seed",
            options=seeds,
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
                )
            )
        with col2:
            show_chart(
                go.Figure().update_layout(
                    title="Jaccard similarity heatmap (placeholder)",
                    plot_bgcolor="white",
                    paper_bgcolor="white"
                )
            )
    elif run is None:
        st.warning(f"No {method_label(div_method)} run found for seed {div_seed} in `{RESULTS_PATH}`.")
    else:
        # Charts
        col1, col2 = st.columns([3, 2], gap="medium")
        with col1, card("div_shap", "Top‑10 mean |SHAP| values", "Averaged across clients"):
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
                    xaxis_title="Mean |SHAP|",
                    yaxis_title="",
                    height=360,
                    margin=dict(l=8, r=8, t=8, b=16),
                    plot_bgcolor="white",
                    paper_bgcolor="white"
                )
                show_chart(fig_shap)
            else:
                st.warning("SHAP top‑10 data not available.")
        with col2, card("div_jaccard", "Jaccard similarity", "Top‑5 feature sets, per client pair"):
            # Jaccard similarity per client pair (1×3 heatmap)
            jaccard_pairwise = run.get("jaccard_pairwise", {})
            jac_vals = [jaccard_pairwise.get(p) for p in PAIRS]
            fig_heat = go.Figure(
                go.Heatmap(
                    z=[jac_vals],
                    x=[PAIR_LABEL[p] for p in PAIRS],
                    y=[""],
                    zmin=0,
                    zmax=1,
                    colorscale=[[0, "#FFFFFF"], [1, TEAL]],
                    text=[[fmt(v) for v in jac_vals]],
                    texttemplate="%{text}",
                    textfont={"color": "black"}
                )
            )
            fig_heat.update_layout(
                height=360,
                margin=dict(l=8, r=8, t=8, b=16),
                plot_bgcolor="white",
                paper_bgcolor="white"
            )
            show_chart(fig_heat)

        # Full‑metrics table
        ds_entry = data.get("divergence_summary", {}).get(run_key(run), {})
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
        with card("div_full", "Full divergence metrics", "Top‑5 Jaccard (J), Spearman (ρ) and cosine per client pair and seed, with seed‑to‑seed noise floor"):
            st.dataframe(
                df_full,
                hide_index=True,
                column_config={c: st.column_config.NumberColumn(format="%.4f") for c in df_full.columns if c != "Pair"},
                height=(len(df_full) + 1) * 35 + 3,
                width="stretch"
            )
            st.caption("|Δ| columns = seed‑to‑seed noise floor. A between‑method gap smaller than its noise‑floor cell is within seed noise.")

        # Divergence scope caption
        scope = ds_entry.get("scope", "Scope not available")
        callout(f"<b>Divergence scope</b>: {html.escape(str(scope))}", kind="info")

        # FedPer and local‑only comparison tables
        if div_method == "fedper":
            with card("div_fedper", "FedPer explanation similarity vs baselines", "Per client, against FedAvg and Local‑only"):
                comp_vs = {"FedAvg": run.get("fedper_vs_fedavg", {}), "Local‑only": run.get("fedper_vs_local_only", {})}
                df_comp = pd.DataFrame([
                    {
                        "Client": PROTO_LABEL[proto],
                        "Compared with": base,
                        "Jaccard": fmt(comp.get(proto, {}).get("top5_jaccard")),
                        "Spearman ρ": fmt(comp.get(proto, {}).get("spearman")),
                        "Cosine": fmt(comp.get(proto, {}).get("cosine")),
                    }
                    for proto in PROTOS for base, comp in comp_vs.items()
                ])
                st.dataframe(df_comp, hide_index=True, height=(len(df_comp) + 1) * 35 + 3, width="stretch")
        elif div_method == "local_only":
            with card("div_local", "Local‑only vs FedAvg", "Per‑client explanation similarity"):
                local_vs_global = data.get("local_vs_global", {})
                key = f"seed{div_seed}_local_vs_fedavg_seed{div_seed}"
                comp = local_vs_global.get(key, {})
                if comp:
                    df_local = pd.DataFrame([
                        {
                            "Client": PROTO_LABEL[proto],
                            "Jaccard": fmt(comp.get(f"{proto}_top5_jaccard")),
                            "Spearman ρ": fmt(comp.get(f"{proto}_spearman")),
                            "Cosine": fmt(comp.get(f"{proto}_cosine")),
                        }
                        for proto in PROTOS
                    ])
                    st.dataframe(df_local, hide_index=True, height=(len(df_local) + 1) * 35 + 3, width="stretch")
                else:
                    st.warning("Local‑vs‑global comparison data not available for this seed.")

# ===== TAB 3: Methods Compared =====
with tab3:
    # Centerpiece: BT benign recall vs μ
    fig_bt = go.Figure()
    # Lines per seed (FedAvg/FedProx family → BLUE; seeds distinguished by dash)
    for i, seed in enumerate(seeds):
        # Collect points: FedAvg (μ=0) + FedProx mus
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
                    marker=dict(size=8)
                )
            )
    # Reference lines: local‑only (dashed teal) and FedPer (dotted amber)
    local_bt_means = [v for s in seeds if (v := bt_recall(data, "local_only", s)) is not None]
    local_bt_mean = sum(local_bt_means) / len(local_bt_means) if local_bt_means else None
    if local_bt_mean is not None:
        fig_bt.add_hline(
            y=local_bt_mean,
            line_dash="dash",
            line_color=TEAL,
            annotation_text=f"Local‑only ({fmt(local_bt_mean)})",
            annotation_position="top left"
        )
    fedper_bt_means = [v for s in seeds if (v := bt_recall(data, "fedper", s)) is not None]
    fedper_bt_mean = sum(fedper_bt_means) / len(fedper_bt_means) if fedper_bt_means else None
    if fedper_bt_mean is not None:
        fig_bt.add_hline(
            y=fedper_bt_mean,
            line_dash="dot",
            line_color=AMBER,
            annotation_text=f"FedPer ({fmt(fedper_bt_mean)})",
            annotation_position="top right"
        )
    fig_bt.update_layout(
        xaxis_title="Method",
        yaxis_title="Benign recall",
        yaxis=dict(range=[0, 1.05]),
        hovermode="x unified",
        height=420,
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    with card("bt_vs_mu", "Bluetooth benign recall vs μ", "FedAvg and FedProx per seed (held‑out test), with local‑only and FedPer reference lines"):
        show_chart(fig_bt)

    # Secondary chart: avg macro F1 per method
    fig_f1 = go.Figure()
    methods_for_f1 = ["fedavg"] + [f"fedprox_mu{m}" for m in mus] + ["fedper", "cluster_fl", "local_only", "centralized"]
    x_labels = []
    y_means = []
    y_errors = []  # half spread for error bars
    for method_key in methods_for_f1:
        if method_key == "centralized":
            # Only pooled macro F1 is stored for centralized
            seed_vals = [v for v in cent_f1.values() if v is not None]
            if seed_vals:
                mean_val, spread = mean_spread(seed_vals)
                x_labels.append("Centralized (pooled)")
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
    # Assign colors: FedAvg/FedProx → BLUE, FedPer → AMBER, Cluster-FL → NAVY, Local‑only → TEAL, Centralized → GREY
    colors = []
    for label in x_labels:
        if label.startswith("FedProx") or label == "FedAvg":
            colors.append(BLUE)
        elif label == "FedPer":
            colors.append(AMBER)
        elif label == "Cluster-FL":
            colors.append(NAVY)
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
            text=[fmt(y) for y in y_means],
            # Inside the bar so labels don't collide with the error bars
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="white"),
        )
    )
    fig_f1.update_layout(
        xaxis_title="Method",
        yaxis_title="Macro F1",
        yaxis=dict(range=[0, 1.0]),
        uniformtext_minsize=8,
        uniformtext_mode="hide",
        height=420,
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    with card("f1_methods", "Average macro F1 per method", "Seed mean; error bars span seed min–max"):
        show_chart(fig_f1)

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
    with card("findings", "Written findings"):
        st.markdown(
            f"""
            - **No μ saves Bluetooth**: Even the gentlest proximal term (μ={mu_min}) lowers Bluetooth benign recall below FedAvg ({smallest_mu_text}). By μ={mu_max} the model flags {flag_range_text} of Bluetooth benign traffic as attack. Local‑only sits at {fmt(local_bt_min)} — the collapse is caused by *sharing*, and proximal anchoring only deepens it.
            - **Local‑only beats all sharing**: Local‑only achieves {fmt(local_mean)} avg macro F1 vs {fmt(best_shared_mean)} for the best shared model ({method_label(best_shared_label)}) — a {local_mean - best_shared_mean:+.4f} absolute gain. It matches the pooled centralized reference ({cent_f1_text}) within {cent_gap_text} points and beats it on Bluetooth benign recall ({fmt(local_bt_min)} vs pooled {cent_benign_text}).
            - **FedPer’s trade‑off position**: FedPer recovers {fmt(fedper_bt_mean)} Bluetooth benign recall (seed mean) — {fedper_gap_text} of the FedAvg→local‑only gap. Its avg macro F1 ({fmt(fedper_f1_mean)}) is {fedper_vs_local_text} of local‑only, and cross‑client explanation similarity remains at FedAvg level (see the Explanation Divergence tab).
            - **Cluster‑FL resolves the collapse structurally**: Protocol‑aligned clusters ({'{Wi‑Fi}'}, {'{MQTT, Bluetooth}'}) avoid inter‑protocol gradient conflict entirely — {fmt(cluster_f1_mean)} avg macro F1 with {fmt(cluster_bt_mean)} Bluetooth benign recall, without a private per‑client head.
            """
        )

# ===== TAB 4: Methodology & Limitations =====
with tab4:
    _mu_list = ", ".join(map(str, mus))
    meth_card = card("methodology", "Study scale & methodology")
    scale_col, methods_col = meth_card.columns([3, 2], gap="large")
    scale_col.markdown('<div class="sub-head">Setup</div>', unsafe_allow_html=True)
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
    methods_col.markdown('<div class="sub-head">Methods compared</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="sub-head">Verified</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="sub-head">Planned / Open</div>', unsafe_allow_html=True)
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
        cf_card.success(
            f"""
            **Status**: Merged ({len(seeds)} seeds) — protocol‑aligned clusters {{Wi‑Fi}} and {{MQTT, Bluetooth}}, one shared model per cluster.

            - **Avg macro F1** (seeds {_cf_seed_txt}): {' / '.join(fmt(v) for v in _cf_f1)} — seed mean **{fmt(_cf_mu)}**
            - **Bluetooth benign recall** (seeds {_cf_seed_txt}): {' / '.join(fmt(v) for v in _cf_bt)} — seed mean **{fmt(_cf_bt_m)}**
            """
        )
        cf_card.expander("Details").markdown(
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
            "Wi‑Fi Top‑5 Jaccard": _jac,
            "MQTT Top‑5 Jaccard": _jac,
            "Bluetooth Top‑5 Jaccard": _jac,
            "Max pairwise Jaccard |diff|": st.column_config.NumberColumn(format="%.4f"),
        },
        height=(len(df_stab) + 1) * 35 + 3,
        width="stretch"
    )
    if shared_status is not None:
        stab_card.caption(f"Status (all aggregators): {shared_status}")

    arch_card = card("architecture", "System architecture diagram")
    if os.path.exists(ARCH_IMG_PATH):
        arch_card.image(ARCH_IMG_PATH, width="stretch")
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
        GEN_COLORS = {"local_only": TEAL, "fedavg": BLUE, "fedper": AMBER, "cluster_fl": NAVY,
                      "fedper_head": AMBER, "fedper_head_unseen": HIGHLIGHT}
        GEN_LABELS = {"fedper_head": "FedPer head (trunk saw target data)",
                      "fedper_head_unseen": "FedPer head (trunk without target)"}

        # ---- Experiment 1: Zero-Day ----
        zd_card = card("zero_day", "Experiment 1 — Zero‑day (novel attack) detection", "Attack recall on attack types never seen in the target client's local training set")
        zd = gen_data.get("zero_day", {})
        zd_targets = list(zd.get("targets", {}).keys())
        zd_target = zd_card.columns([1, 3])[0].selectbox(
            "Target client",
            options=zd_targets,
            format_func=lambda p: PROTO_LABEL.get(p, p),
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

        fig_zd = go.Figure()
        for m in zd_methods:
            per_attack = zd_summary[m].get("per_attack", {})
            fig_zd.add_trace(go.Bar(
                name=GEN_LABELS.get(m, method_label(m)),
                x=attacks,
                y=[per_attack.get(a, {}).get("mean") for a in attacks],
                marker_color=GEN_COLORS.get(m, GREY),
            ))
        fig_zd.update_layout(
            barmode="group",
            xaxis_title="Unseen attack type (from other protocols' held‑out test splits)",
            yaxis_title="Attack recall (seed mean)",
            yaxis=dict(range=[0, 1.05]),
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=480,
        )
        with zd_card:
            show_chart(fig_zd)

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
            },
            width="stretch",
        )
        zd_card.expander("How to read this table").markdown(
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
        ds_target = ds_card.columns([1, 3])[0].selectbox(
            "Target client",
            options=list(ds_targets.keys()),
            format_func=lambda p: PROTO_LABEL.get(p, p),
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
                line=dict(color=GEN_COLORS.get(arm, GREY)),
            ))
        fig_ds.update_layout(
            xaxis=dict(title="Training samples available at the clinic (N, log scale)", type="log"),
            yaxis=dict(title="Macro F1 on full held‑out test set", range=[0, 1.05]),
            hovermode="x unified",
            height=440,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        with ds_card:
            show_chart(fig_ds)
        ds_cfg = ds.get("config", {})
        ds_card.expander("Method details").markdown(
            f"Seeds {ds_cfg.get('seeds')} × {ds_cfg.get('repeats_per_seed')} stratified draws per N; "
            f"{ds_cfg.get('epochs')} epochs with a floor of {ds_cfg.get('min_optimizer_steps')} optimizer steps for every arm; "
            "scaler refit on the N‑sample subset.\n\n"
            + "\n".join(f"- **{GEN_LABELS.get(k, method_label(k))}**: {v}." for k, v in ds.get("methods", {}).items())
        )

        # ---- Takeaway (computed from the JSON, not hard-coded) ----
        tk_card = card("takeaway", "Takeaway")
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
        tk_card.caption("Two seeds — directional only.")
        tk_card.expander("Interpretation guide").markdown(
            "FL is justified here only where its zero‑day recall exceeds local‑only with a positive "
            "detection margin, and where the head‑only curve (especially the trunk trained without the target client) "
            "stays above local‑only at small N."
        )

# ---- Footer ----
st.markdown(
    '<div class="footer">FedXplain‑IoMT • Experimental research framework • '
    f"Dashboard rendered with Python {'.'.join(map(str, sys.version_info[:3]))} • "
    f"Study results timestamp: {html.escape(str(run_meta.get('timestamp', '—')))}</div>",
    unsafe_allow_html=True
)