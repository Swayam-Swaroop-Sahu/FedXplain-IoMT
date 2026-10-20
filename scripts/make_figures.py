"""Generate the four publication figures for the study.

All figures are derived from results/study_results.json only - no training,
no SHAP computation - and are fully regenerable:
    python scripts/make_figures.py
Writes 300-dpi PNGs into figures/.

Palette (validated with the dataviz six-checks validator on the #F5F5F2
surface): federated methods blue #2A78D6, local-only teal #2A9D8F, FedPer
amber #C98500 (amber carries a contrast WARN -> every amber mark is
direct-labeled), chrome/text navy #0F2A43, warm-grey surface #F5F5F2.
With n=2 seeds, "error bars" are the honest min-max range across seeds.
"""

import json
import os
import sys
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MU_VALUES, PROTOCOLS, SEEDS, STUDY_RESULTS_PATH  # noqa: E402

# --- palette (validated; see module docstring) ---
BLUE = "#2A78D6"      # shared-model federated methods (FedAvg, FedProx)
TEAL = "#2A9D8F"      # local-only control
AMBER = "#C98500"     # FedPer (personalized)
NAVY = "#0F2A43"      # text / chrome
SURFACE = "#F5F5F2"   # figure background
GRID = "#D8D8D2"      # recessive gridlines

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

PROTOCOL_TITLES = {"wifi": "Wi-Fi", "mqtt": "MQTT", "bluetooth": "Bluetooth"}
METHOD_LABELS = {
    "fedavg": "FedAvg",
    "fedper": "FedPer",
    "cluster_fl": "Cluster-FL",
    "local_only": "Local-only",
}
PAIR_TITLES = {"wifi-mqtt": "Wi-Fi vs MQTT", "wifi-bluetooth": "Wi-Fi vs BT", "mqtt-bluetooth": "MQTT vs BT"}


def _method_label(method: str) -> str:
    """Human-readable method name (fedprox_mu0.01 -> 'FedProx (mu=0.01)')."""
    if method in METHOD_LABELS:
        return METHOD_LABELS[method]
    mu = method.replace("fedprox_mu", "")
    return f"FedProx ($\\mu$={mu})"


def _method_color(method: str) -> str:
    """Series color by method family: blue = shared-model FL, teal = local-only, amber = FedPer, purple = Cluster-FL."""
    if method == "local_only":
        return TEAL
    if method == "fedper":
        return AMBER
    if method == "cluster_fl":
        return "#8338EC"
    return BLUE


def _load_runs() -> dict:
    with open(STUDY_RESULTS_PATH, "r", encoding="utf-8") as f:
        study = json.load(f)
    runs = study.get("runs", [])
    by_seed_method: dict[tuple[int, str], dict] = {}
    for r in runs:
        method = r["aggregator"] if r["aggregator"] != "fedprox" else f"fedprox_mu{r['mu']}"
        by_seed_method[(r["seed"], method)] = r
    for r in study.get("local_only", []):
        by_seed_method[(r["seed"], "local_only")] = r
    return {"study": study, "by_seed_method": by_seed_method}


def _seed_values(
    by_seed_method: dict, method: str, getter: Callable[[dict], float]
) -> list[float]:
    """Collect a metric across seeds for one method (order follows SEEDS)."""
    return [getter(by_seed_method[(seed, method)]) for seed in SEEDS]


def _mean_range(values: list[float]) -> tuple[float, float]:
    """Mean and half-width of the min-max range (n=2 seeds: honest error bar)."""
    return float(np.mean(values)), float(max(values) - min(values)) / 2.0


def _collect_metrics(
    by_seed_method: dict, methods: list[str], getter: Callable[[dict], float]
) -> dict[str, tuple[float, float, list[float]]]:
    """Per method: (mean, half-range, per-seed values) for a scalar metric."""
    out: dict[str, tuple[float, float, list[float]]] = {}
    for m in methods:
        vals = _seed_values(by_seed_method, m, getter)
        mean, half = _mean_range(vals)
        out[m] = (mean, half, vals)
    return out


def _apply_style(ax: plt.Axes) -> None:
    """Recessive axes/grid styling shared by all figures."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y" if getattr(ax, "_y_is_value", True) else "x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=NAVY, labelsize=10)


# ---------------------------------------------------------------------------
# Figure 1: macro F1 per client per aggregator (faceted by client, one
# horizontal bar per method; identity by y-label, color by method family)
# ---------------------------------------------------------------------------
def fig1_detection(study: dict, by_seed_method: dict) -> str:
    has_cluster_fl = any(r.get("aggregator") == "cluster_fl" for r in study.get("runs", []))
    methods = ["fedavg"] + [f"fedprox_mu{m}" for m in MU_VALUES] + ["fedper"]
    if has_cluster_fl:
        methods.append("cluster_fl")
    methods.append("local_only")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.0), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, proto in zip(axes, PROTOCOLS):
        getter = lambda r, p=proto: r["client_metrics"][p]["f1_macro"]  # noqa: E731
        stats = _collect_metrics(by_seed_method, methods, getter)
        y = np.arange(len(methods))[::-1]
        means = [stats[m][0] for m in methods]
        errs = [stats[m][1] for m in methods]
        colors = [_method_color(m) for m in methods]
        ax._y_is_value = False  # horizontal bars: value axis is x
        ax.barh(y, means, xerr=errs, height=0.62, color=colors,
                error_kw=dict(ecolor=NAVY, elinewidth=1.2, capsize=3))
        for yi, m in zip(y, methods):
            mean = stats[m][0]
            # Place the label past the error whisker so bars with a wide
            # min-max range never collide with their own label.
            ax.text(mean + errs[list(methods).index(m)] + 0.015, yi,
                    f"{mean:.3f}", va="center", ha="left", fontsize=9, color=NAVY)
        ax.set_yticks(y)
        ax.set_yticklabels([_method_label(m) for m in methods], fontsize=10)
        ax.set_xlim(0, 1.12)
        ax.set_title(PROTOCOL_TITLES[proto], fontsize=12, color=NAVY, pad=10)
        ax.set_xlabel("Macro F1 (test set)", fontsize=10, color=NAVY)
        _apply_style(ax)

    fig.suptitle(
        "Per-client detection quality by method (mean across seeds 42 & 7; bars show min-max range)",
        fontsize=13, color=NAVY, y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(FIGURES_DIR, "fig1_detection_macro_f1.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 2: Spearman divergence heatmap (methods x client pairs, seed mean),
# sequential blue ramp, with the seed-to-seed noise floor shown alongside
# ---------------------------------------------------------------------------
def fig2_divergence_heatmap(study: dict, by_seed_method: dict) -> str:
    has_cluster_fl = any(r.get("aggregator") == "cluster_fl" for r in study.get("runs", []))
    methods = ["fedavg"] + [f"fedprox_mu{m}" for m in MU_VALUES] + ["fedper"]
    if has_cluster_fl:
        methods.append("cluster_fl")
    methods.append("local_only")
    pairs = [f"{a}-{b}" for a, b in zip(PROTOCOLS[:-1], PROTOCOLS[1:])] + [
        f"{PROTOCOLS[0]}-{PROTOCOLS[-1]}"
    ]
    pair_keys = ["wifi-mqtt", "wifi-bluetooth", "mqtt-bluetooth"]
    matrix = np.zeros((len(methods), len(pair_keys)))
    noise = np.zeros_like(matrix)
    for i, m in enumerate(methods):
        for j, pk in enumerate(pair_keys):
            vals = [
                by_seed_method[(seed, m)]["divergence_full"][pk]["spearman"]
                for seed in SEEDS
            ]
            matrix[i, j] = np.mean(vals)
            noise[i, j] = np.abs(vals[0] - vals[1])

    fig, (ax, axn) = plt.subplots(
        1, 2, figsize=(10.5, 4.4), gridspec_kw={"width_ratios": [3, 3]}
    )
    fig.patch.set_facecolor(SURFACE)

    im = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=0.8, aspect="auto")
    ax.set_xticks(range(len(pair_keys)))
    ax.set_xticklabels([PAIR_TITLES[p] for p in pair_keys], fontsize=10, color=NAVY)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([_method_label(m) for m in methods], fontsize=10, color=NAVY)
    ax.set_title("Spearman agreement of explanations\n(mean across seeds)",
                 fontsize=11, color=NAVY, pad=10)
    for i in range(len(methods)):
        for j in range(len(pair_keys)):
            val = matrix[i, j]
            color = "white" if val > 0.55 else NAVY
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Noise floor panel: seed-to-seed |diff| of the same statistic
    axn.imshow(noise, cmap="Greys", vmin=0.0, vmax=max(0.12, noise.max()), aspect="auto")
    axn.set_xticks(range(len(pair_keys)))
    axn.set_xticklabels([PAIR_TITLES[p] for p in pair_keys], fontsize=10, color=NAVY)
    axn.set_yticks(range(len(methods)))
    axn.set_yticklabels([])
    axn.set_title("Noise floor: seed-to-seed\n|difference| of Spearman",
                  fontsize=11, color=NAVY, pad=10)
    for i in range(len(methods)):
        for j in range(len(pair_keys)):
            shade = noise[i, j] / max(0.12, noise.max())
            color = "white" if shade > 0.55 else NAVY
            axn.text(j, i, f"{noise[i, j]:.2f}", ha="center", va="center",
                     fontsize=9, color=color)
    axn.tick_params(length=0)
    for spine in axn.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.ax.tick_params(labelsize=9, colors=NAVY)
    cbar.outline.set_edgecolor(GRID)

    fig.suptitle(
        "Cross-client explanation divergence - any between-method gap smaller than its noise-floor cell is within seed noise",
        fontsize=12, color=NAVY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = os.path.join(FIGURES_DIR, "fig2_spearman_divergence_heatmap.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 3: Bluetooth benign recall across the FedProx mu sweep ("no safe mu")
# ---------------------------------------------------------------------------
def fig3_mu_sweep(study: dict, by_seed_method: dict) -> str:
    xticks = ["FedAvg\n($\\mu$=0)"] + [str(m) for m in MU_VALUES]
    x = np.arange(len(xticks))
    recalls = [
        _seed_values(by_seed_method, "fedavg", lambda r: r["client_metrics"]["bluetooth"]["benign_recall"])
    ] + [
        _seed_values(by_seed_method, f"fedprox_mu{m}",
                     lambda r: r["client_metrics"]["bluetooth"]["benign_recall"])
        for m in MU_VALUES
    ]
    means = [np.mean(v) for v in recalls]
    lows = [np.mean(v) - min(v) for v in recalls]
    highs = [max(v) - np.mean(v) for v in recalls]

    lo_vals = _seed_values(by_seed_method, "local_only",
                           lambda r: r["client_metrics"]["bluetooth"]["benign_recall"])
    lo_mean = float(np.mean(lo_vals))

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Local-only reference: the level the collapse never approaches
    ax.axhline(lo_mean, color=TEAL, linestyle="--", linewidth=2)
    ax.text(len(xticks) - 1 + 0.12, lo_mean, f"Local-only ({lo_mean:.3f})",
            va="center", ha="left", fontsize=10, color=NAVY)

    # FedAvg -> FedProx sweep: one connected series (mu=0 IS FedAvg)
    ax.errorbar(x, means, yerr=[lows, highs], color=BLUE, linewidth=2,
                marker="o", markersize=7, capsize=4,
                ecolor=NAVY, elinewidth=1.2)
    for xi, mean in zip(x, means):
        ha = "left" if xi < len(xticks) - 1 else "right"
        off = 0.045 if xi < len(xticks) - 1 else -0.045
        ax.text(xi + off, mean, f"{mean:.3f}", va="bottom" if mean < 0.5 else "top",
                ha=ha, fontsize=9, color=NAVY)

    ax.annotate(
        "benign traffic flagged as attack\n(recall collapses as $\\mu$ rises)",
        xy=(3, means[3]), xytext=(1.6, 0.62),
        fontsize=10, color=NAVY, ha="left",
        arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.2),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(xticks, fontsize=10)
    ax.set_xlabel("FedProx proximal term $\\mu$ (FedAvg is $\\mu$=0)", fontsize=11, color=NAVY)
    ax.set_ylabel("Bluetooth benign recall (higher = better)", fontsize=11, color=NAVY)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlim(-0.4, len(xticks) - 0.2)
    ax.set_title(
        "No safe $\\mu$: Bluetooth benign recall under the FedProx sweep\n"
        "(mean across seeds 42 & 7; error bars show min-max range)",
        fontsize=12, color=NAVY, pad=12,
    )
    _apply_style(ax)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig3_bluetooth_mu_sweep.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 4: trade-off scatter - detection quality vs explanation consistency
# ---------------------------------------------------------------------------
def fig4_tradeoff(study: dict, by_seed_method: dict) -> str:
    has_cluster_fl = any(r.get("aggregator") == "cluster_fl" for r in study.get("runs", []))
    methods = ["fedavg"] + [f"fedprox_mu{m}" for m in MU_VALUES] + ["fedper"]
    if has_cluster_fl:
        methods.append("cluster_fl")
    methods.append("local_only")

    def avg_macro_f1(r: dict) -> float:
        return float(np.mean([r["client_metrics"][p]["f1_macro"] for p in PROTOCOLS]))

    def mean_spearman(r: dict) -> float:
        return float(np.mean([r["divergence_full"][p]["spearman"] for p in r["divergence_full"]]))

    xs = _collect_metrics(by_seed_method, methods, mean_spearman)
    ys = _collect_metrics(by_seed_method, methods, avg_macro_f1)

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for m in methods:
        xmean, xhalf, _ = xs[m]
        ymean, yhalf, _ = ys[m]
        ax.errorbar(xmean, ymean, xerr=xhalf, yerr=yhalf,
                    fmt="o", markersize=9, color=_method_color(m),
                    ecolor=NAVY, elinewidth=1.0, capsize=3, zorder=3)

    # Direct labels (amber FedPer relief obligation; all methods labeled)
    label_offsets = {
        "fedavg": (0.012, 0.008),
        "fedprox_mu0.001": (0.012, 0.010),
        "fedprox_mu0.01": (-0.012, 0.012),
        "fedprox_mu0.05": (-0.012, 0.008),
        "fedprox_mu0.1": (-0.012, -0.014),
        "fedper": (0.014, 0.006),
        "cluster_fl": (0.014, 0.006),
        "local_only": (0.014, -0.031),
    }
    for m in methods:
        dx, dy = label_offsets[m]
        ha = "left" if dx > 0 else "right"
        ax.text(xs[m][0] + dx, ys[m][0] + dy, _method_label(m),
                fontsize=10, color=NAVY, ha=ha, va="bottom" if dy >= 0 else "top")

    # Quadrant midlines make the 2x2 legible; both captions stay inside the
    # axes so the saved canvas is not stretched by out-of-bounds text.
    ax.set_ylim(0.66, 1.0)
    ax.axhline(0.85, color=GRID, linewidth=1, linestyle=":")
    ax.axvline(0.45, color=GRID, linewidth=1, linestyle=":")
    ax.text(0.455, 0.985, "high quality / high agreement", fontsize=9, color=NAVY, alpha=0.75)
    ax.text(0.255, 0.672, "low quality / high agreement", fontsize=9, color=NAVY, alpha=0.75)

    ax.set_xlabel("Cross-client explanation consistency (mean Spearman across pairs)",
                  fontsize=11, color=NAVY)
    ax.set_ylabel("Detection quality (avg macro F1 across clients)", fontsize=11, color=NAVY)
    ax.set_title(
        "The personalization trade-off: detection quality vs explanation consistency\n"
        "(mean across seeds 42 & 7; error bars show min-max range)",
        fontsize=12, color=NAVY, pad=12,
    )
    _apply_style(ax)
    # NOTE: tight_layout collapses this axes to a sliver on some matplotlib
    # versions (observed 3.11.2: axes box shrank to 1.6% of figure height),
    # so the layout is set explicitly instead.
    fig.subplots_adjust(left=0.11, right=0.96, top=0.85, bottom=0.13)
    path = os.path.join(FIGURES_DIR, "fig4_tradeoff_scatter.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def main() -> None:
    data = _load_runs()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    paths = [
        fig1_detection(data["study"], data["by_seed_method"]),
        fig2_divergence_heatmap(data["study"], data["by_seed_method"]),
        fig3_mu_sweep(data["study"], data["by_seed_method"]),
        fig4_tradeoff(data["study"], data["by_seed_method"]),
    ]
    for p in paths:
        print(f"[OK] wrote {p}")


if __name__ == "__main__":
    main()
