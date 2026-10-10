"""SHAP explanation generation and cross-client divergence analysis."""

import os
import sys

# Ensure numba JIT does not trigger Windows Application Control DLL block
os.environ["NUMBA_DISABLE_JIT"] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from scipy.stats import spearmanr

# Allow running as `python src/explain.py` from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    DATA_DIR,
    MODELS_DIR,
    PROTOCOLS,
    RESULTS_DIR,
    SHAP_ATTACK_SAMPLE_SIZE,
    SHAP_BACKGROUND_SIZE,
)
from src.model import IoMTMLP
from src.utils import get_feature_names, get_input_dim, load_and_preprocess, set_seed


def _spearman_cosine(a: np.ndarray, b: np.ndarray) -> tuple[float | None, float | None]:
    """Spearman rank correlation and cosine similarity between two attribution vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return None, None
    spearman = float(spearmanr(a, b)[0])
    if not np.isfinite(spearman):
        spearman = None
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    cosine = float(np.dot(a, b) / denom) if denom > 0 else None
    return spearman, cosine


def compute_pairwise_divergence(
    client_top5: dict[str, list[str]],
    client_full_vectors: dict[str, dict[str, list]],
) -> tuple[dict[str, float], dict[str, list[str]], dict[str, dict[str, float | None]]]:
    """Compute pairwise Jaccard, shared features, and full-vector divergence metrics."""
    pairs = [("wifi", "mqtt"), ("wifi", "bluetooth"), ("mqtt", "bluetooth")]
    jaccard_pairwise: dict[str, float] = {}
    shared_features: dict[str, list[str]] = {}
    divergence_full: dict[str, dict[str, float | None]] = {}

    for p1, p2 in pairs:
        pair_label = f"{p1}-{p2}"
        if not client_top5.get(p1) or not client_top5.get(p2):
            jaccard_pairwise[pair_label] = 0.0
            shared_features[pair_label] = []
        else:
            set1 = set(client_top5[p1])
            set2 = set(client_top5[p2])
            shared = sorted(list(set1.intersection(set2)))
            union = set1.union(set2)
            jaccard = float(len(shared) / len(union)) if union else 0.0
            jaccard_pairwise[pair_label] = round(jaccard, 4)
            shared_features[pair_label] = shared

        # Full-vector divergence (Spearman / cosine) requires both clients' vectors
        vec1, vec2 = client_full_vectors.get(p1), client_full_vectors.get(p2)
        if vec1 is None or vec2 is None:
            divergence_full[pair_label] = {"spearman": None, "cosine": None}
            continue
        # Align by feature name so vector order cannot skew the comparison
        names1 = list(vec1["feature_names"])
        names2 = list(vec2["feature_names"])
        common = [n for n in names1 if n in set(names2)]
        a = np.array([vec1["values"][names1.index(n)] for n in common])
        b = np.array([vec2["values"][names2.index(n)] for n in common])
        spearman, cosine = _spearman_cosine(a, b)
        divergence_full[pair_label] = {
            "spearman": round(spearman, 4) if spearman is not None else None,
            "cosine": round(cosine, 4) if cosine is not None else None,
        }

    return jaccard_pairwise, shared_features, divergence_full


def explain_client(
    model: torch.nn.Module,
    proto: str,
    feature_names: list[str],
    data_dir: str = DATA_DIR,
    seed: int = 42,
    bg_size: int = SHAP_BACKGROUND_SIZE,
    attack_sample_size: int = SHAP_ATTACK_SAMPLE_SIZE,
) -> tuple[list[tuple[str, float]], list[str], dict[str, float]]:
    """Compute SHAP top-10/top-5 features and the full mean-|SHAP| vector for one client.

    Returns (top10, top5, full_mean_shap) where full_mean_shap is a dict with
    'feature_names' and 'values' (45-element lists), or empty lists/dicts if the
    client has no attack samples.
    """
    set_seed(seed)
    X_train, X_test, y_train, y_test = load_and_preprocess(
        proto, data_dir=data_dir, random_state=seed
    )

    # Background set: random samples from X_train where y_train == 0 (benign)
    benign_indices = torch.where(y_train == 0)[0]
    if len(benign_indices) == 0:
        bg_tensor = X_train[:bg_size]
    elif len(benign_indices) <= bg_size:
        bg_tensor = X_train[benign_indices]
    else:
        perm = torch.randperm(len(benign_indices))[:bg_size]
        bg_tensor = X_train[benign_indices[perm]]

    # Attack samples: from X_test where y_test == 1
    attack_indices = torch.where(y_test == 1)[0]
    n_attack = len(attack_indices)

    if n_attack == 0:
        print(f"[WARNING] Protocol '{proto}' has ZERO attack samples. Skipping SHAP.")
        return [], [], {}

    sample_size = min(attack_sample_size, n_attack)
    perm = torch.randperm(n_attack)[:sample_size]
    attack_tensor = X_test[attack_indices[perm]]

    print(f"Protocol: {proto.upper():<9} | Background: {bg_tensor.shape[0]} benign | Attack: {attack_tensor.shape[0]} samples")

    # Compute SHAP values with DeepExplainer
    explainer = shap.DeepExplainer(model, bg_tensor)
    shap_values = explainer.shap_values(attack_tensor)

    if isinstance(shap_values, list):
        shap_arr = shap_values[0]
    else:
        shap_arr = shap_values
    shap_arr = np.squeeze(shap_arr)

    mean_shap = np.abs(shap_arr).mean(axis=0)
    mean_shap = np.squeeze(mean_shap)

    # Rank descending, keep top 10
    ranked_indices = np.argsort(mean_shap)[::-1]
    top10_idx = ranked_indices[:10]
    top10_feats = [feature_names[i] for i in top10_idx]
    top10_vals = [float(mean_shap[i]) for i in top10_idx]

    top10 = list(zip(top10_feats, top10_vals))
    top5 = top10_feats[:5]
    full_mean_shap = {
        "feature_names": list(feature_names),
        "values": [float(v) for v in mean_shap],
    }

    print(f"  Top 5 features for {proto.upper()}: {top10_feats[:5]}\n")
    return top10, top5, full_mean_shap


def compute_shap_divergence(
    model_or_path: str | torch.nn.Module = os.path.join(MODELS_DIR, "fedavg_global.pth"),
    data_dir: str = DATA_DIR,
    seed: int = 42,
    bg_size: int = SHAP_BACKGROUND_SIZE,
    attack_sample_size: int = SHAP_ATTACK_SAMPLE_SIZE,
    results_dir: str = RESULTS_DIR,
    plot_prefix: str = "",
    title_suffix: str = "FedAvg Global Model",
    save_plots: bool = True,
) -> dict:
    """Compute SHAP explanations per client and cross-client divergence."""
    set_seed(seed)
    os.makedirs(results_dir, exist_ok=True)

    input_dim = get_input_dim("wifi", data_dir=data_dir)
    if isinstance(model_or_path, str):
        if not os.path.exists(model_or_path):
            raise FileNotFoundError(f"Model file not found at {model_or_path}")
        model = IoMTMLP(input_dim)
        model.load_state_dict(torch.load(model_or_path, weights_only=True))
    else:
        model = model_or_path
    model.eval()

    feature_names = get_feature_names("wifi", data_dir=data_dir)

    client_top10: dict[str, list[tuple[str, float]]] = {}
    client_top5: dict[str, list[str]] = {}
    client_full_vectors: dict[str, dict[str, list]] = {}

    print(f"Computing SHAP explanations (data_dir={data_dir}, bg={bg_size}, attack={attack_sample_size}, seed={seed})...\n")

    for proto in PROTOCOLS:
        top10, top5, full_vec = explain_client(
            model,
            proto,
            feature_names,
            data_dir=data_dir,
            seed=seed,
            bg_size=bg_size,
            attack_sample_size=attack_sample_size,
        )
        client_top10[proto] = top10
        client_top5[proto] = top5
        client_full_vectors[proto] = full_vec

    # ------------------------------------------------------------------
    # Top-5 Jaccard overlap + full-vector divergence (Spearman / cosine)
    # ------------------------------------------------------------------
    jaccard_pairwise, shared_features, divergence_full = compute_pairwise_divergence(
        client_top5, client_full_vectors
    )

    # ------------------------------------------------------------------
    # Save plots if requested
    # ------------------------------------------------------------------
    if save_plots:
        for proto in PROTOCOLS:
            if not client_top10.get(proto):
                continue

            top10_feats, top10_vals = zip(*client_top10[proto])
            y_pos = np.arange(len(top10_feats))
            feats_rev = list(top10_feats)[::-1]
            vals_rev = list(top10_vals)[::-1]

            fig, ax = plt.subplots(figsize=(8, 5))
            color = "#3b82f6" if "fedprox" not in plot_prefix else "#0284c7"
            edgecolor = "#1d4ed8" if "fedprox" not in plot_prefix else "#0369a1"
            bars = ax.barh(y_pos, vals_rev, color=color, edgecolor=edgecolor, alpha=0.85)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feats_rev, fontsize=9)
            ax.set_xlabel("Mean |SHAP Value|", fontsize=10, fontweight="bold")
            ax.set_title(f"Top 10 SHAP Features - {proto.upper()} Client ({title_suffix})", fontsize=11, fontweight="bold")
            ax.grid(axis="x", linestyle="--", alpha=0.5)

            max_val = max(vals_rev) if vals_rev else 1.0
            ax.set_xlim(0, max_val * 1.15)
            for bar in bars:
                width = bar.get_width()
                ax.text(width + (max_val * 0.015), bar.get_y() + bar.get_height() / 2, f"{width:.4f}",
                        ha="left", va="center", fontsize=8, color="#1e293b")

            plt.tight_layout()
            prefix_str = f"_{plot_prefix}" if plot_prefix else ""
            plot_path = os.path.join(results_dir, f"shap_top10_{proto}{prefix_str}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

        # Cross-client comparison plot
        valid_protos = [p for p in PROTOCOLS if client_top5.get(p)]
        if valid_protos:
            all_top5_unique = sorted(list(set(feat for p in valid_protos for feat in client_top5[p])))
            color_palette = plt.cm.tab20(np.linspace(0, 1, max(len(all_top5_unique), 1)))
            feature_color_map = {feat: color_palette[i] for i, feat in enumerate(all_top5_unique)}

            n_plots = len(valid_protos)
            fig, axes = plt.subplots(1, n_plots, figsize=(5.5 * n_plots, 4.8), sharey=False)
            if n_plots == 1:
                axes = [axes]

            for ax, proto in zip(axes, valid_protos):
                top10_feats, top10_vals = zip(*client_top10[proto])
                top5_feats = list(top10_feats[:5])[::-1]
                top5_vals = list(top10_vals[:5])[::-1]
                y_pos = np.arange(len(top5_feats))
                bar_colors = [feature_color_map[f] for f in top5_feats]

                bars = ax.barh(y_pos, top5_vals, color=bar_colors, edgecolor="#334155", linewidth=0.8, alpha=0.9)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(top5_feats, fontsize=10, fontweight="normal")
                ax.set_xlabel("Mean |SHAP Value|", fontsize=10)
                ax.set_title(f"{proto.upper()} Client", fontsize=12, fontweight="bold")
                ax.grid(axis="x", linestyle="--", alpha=0.5)

                max_val = max(top5_vals) if top5_vals else 1.0
                ax.set_xlim(0, max_val * 1.15)
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + (max_val * 0.015), bar.get_y() + bar.get_height() / 2, f"{width:.3f}",
                            ha="left", va="center", fontsize=8, color="#0f172a")

            fig.suptitle(f"Cross-Client SHAP Explanation Divergence ({title_suffix})", fontsize=13, fontweight="bold", y=1.02)
            plt.tight_layout()
            comp_name = f"shap_cross_client_comparison_{plot_prefix}.png" if plot_prefix else "shap_cross_client_comparison.png"
            comparison_plot_path = os.path.join(results_dir, comp_name)
            plt.savefig(comparison_plot_path, dpi=300, bbox_inches="tight")
            plt.close()

    # Output divergence table
    print("=" * 65)
    print(f"{'Client Pair':<18} | {'Shared Top-5 Features':<28} | {'Jaccard Index':>13}")
    print("-" * 65)
    for pair_lbl, jaccard in jaccard_pairwise.items():
        shared_str = ", ".join(shared_features[pair_lbl]) if shared_features[pair_lbl] else "None"
        print(f"{pair_lbl:<18} | {shared_str:<28} | {jaccard:>13.4f}")
    print("=" * 65 + "\n")

    return {
        "client_top10": client_top10,
        "client_top5": client_top5,
        "shap_full_vectors_per_client": client_full_vectors,
        "jaccard_pairwise": jaccard_pairwise,
        "divergence_full": divergence_full,
        "shared_features": shared_features,
    }


def main() -> None:
    compute_shap_divergence(
        model_or_path=os.path.join(MODELS_DIR, "fedavg_global.pth"),
        data_dir=DATA_DIR,
        title_suffix="FedAvg Global Model",
    )


if __name__ == "__main__":
    main()
