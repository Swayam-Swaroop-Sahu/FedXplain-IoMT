"""SHAP explanation generation and cross-client divergence analysis."""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch

# Allow running as `python src/explain.py` from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import set_seed, load_and_preprocess, get_input_dim, get_feature_names
from src.model import IoMTMLP

PROTOCOLS = ["wifi", "mqtt", "bluetooth"]
RESULTS_DIR = "results"
MODEL_PATH = "models/fedavg_global.pth"


def main() -> None:
    set_seed(42)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load federated global model
    input_dim = get_input_dim("wifi")
    model = IoMTMLP(input_dim)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please run src/run_federated.py first.")

    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    feature_names = get_feature_names("wifi")

    client_top10 = {}
    client_top5 = {}
    client_mean_shap = {}

    print("Computing SHAP explanations per client...\n")

    for proto in PROTOCOLS:
        X_train, X_test, y_train, y_test = load_and_preprocess(proto)

        # Background set: 50 random samples from X_train where y_train == 0 (benign)
        benign_indices = torch.where(y_train == 0)[0]
        if len(benign_indices) == 0:
            bg_tensor = X_train[:50]
        elif len(benign_indices) < 50:
            bg_tensor = X_train[benign_indices]
        else:
            perm = torch.randperm(len(benign_indices))[:50]
            bg_tensor = X_train[benign_indices[perm]]

        # Attack samples: from X_test where y_test == 1
        attack_indices = torch.where(y_test == 1)[0]
        n_attack = len(attack_indices)

        if n_attack == 0:
            print(f"[WARNING] Protocol '{proto}' has ZERO attack samples in its test set. Skipping SHAP.")
            client_top10[proto] = None
            client_top5[proto] = None
            client_mean_shap[proto] = None
            continue

        if n_attack < 50:
            attack_tensor = X_test[attack_indices]
        else:
            sample_size = min(100, n_attack)
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

        client_top10[proto] = list(zip(top10_feats, top10_vals))
        client_top5[proto] = top10_feats[:5]
        client_mean_shap[proto] = mean_shap

        print(f"  Top 5 features for {proto.upper()}: {top10_feats[:5]}\n")

    # ------------------------------------------------------------------
    # 2. Per-client plots: horizontal bar chart of top 10 features
    # ------------------------------------------------------------------
    for proto in PROTOCOLS:
        if client_top10[proto] is None:
            continue

        top10_feats, top10_vals = zip(*client_top10[proto])
        # Invert so highest importance is at the top
        y_pos = np.arange(len(top10_feats))
        feats_rev = list(top10_feats)[::-1]
        vals_rev = list(top10_vals)[::-1]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(y_pos, vals_rev, color="#3b82f6", edgecolor="#1d4ed8", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feats_rev, fontsize=9)
        ax.set_xlabel("Mean |SHAP Value|", fontsize=10, fontweight="bold")
        ax.set_title(f"Top 10 SHAP Features - {proto.upper()} Client (Attack Samples)", fontsize=11, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        max_val = max(vals_rev) if vals_rev else 1.0
        ax.set_xlim(0, max_val * 1.15)
        # Value annotations on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + (max_val * 0.015), bar.get_y() + bar.get_height() / 2, f"{width:.4f}",
                    ha="left", va="center", fontsize=8, color="#1e293b")

        plt.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, f"shap_top10_{proto}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {plot_path}")

    # ------------------------------------------------------------------
    # 3. Cross-client comparison plot: top 5 features side by side
    # ------------------------------------------------------------------
    valid_protos = [p for p in PROTOCOLS if client_top5[p] is not None]
    if valid_protos:
        # Collect all unique top-5 features across clients for consistent coloring
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

        fig.suptitle("Cross-Client SHAP Explanation Divergence (FedAvg Global Model)", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        comparison_plot_path = os.path.join(RESULTS_DIR, "shap_cross_client_comparison.png")
        plt.savefig(comparison_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {comparison_plot_path}")

    # ------------------------------------------------------------------
    # 4. Top-5 Jaccard overlap
    # ------------------------------------------------------------------
    pairs = [("wifi", "mqtt"), ("wifi", "bluetooth"), ("mqtt", "bluetooth")]
    report_rows = []

    for p1, p2 in pairs:
        pair_label = f"{p1}-{p2}"
        if client_top5.get(p1) is None or client_top5.get(p2) is None:
            report_rows.append((pair_label, "N/A", "N/A"))
        else:
            set1 = set(client_top5[p1])
            set2 = set(client_top5[p2])
            shared = sorted(list(set1.intersection(set2)))
            shared_str = ", ".join(shared) if shared else "None"
            union = set1.union(set2)
            jaccard = len(shared) / len(union) if union else 0.0
            report_rows.append((pair_label, shared_str, f"{jaccard:.4f}"))

    # Console output
    print("\n" + "=" * 65)
    print(f"{'Client Pair':<18} | {'Shared Top-5 Features':<28} | {'Jaccard Index':>13}")
    print("-" * 65)
    for pair_lbl, shared, jaccard in report_rows:
        print(f"{pair_lbl:<18} | {shared:<28} | {jaccard:>13}")
    print("=" * 65)
    print("Lower Jaccard indicates higher explanation divergence across clients.\n")

    # Save markdown report
    report_md_path = os.path.join(RESULTS_DIR, "divergence_report.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# SHAP Cross-Client Explanation Divergence Report\n\n")
        f.write("| Client Pair | Shared Top-5 Features | Jaccard Index |\n")
        f.write("| :--- | :--- | :--- |\n")
        for pair_lbl, shared, jaccard in report_rows:
            f.write(f"| {pair_lbl} | {shared} | {jaccard} |\n")
        f.write("\nLower Jaccard indicates higher explanation divergence across clients.\n")

    print(f"Saved: {report_md_path}")
    print("SHAP explanation pipeline completed successfully.")


if __name__ == "__main__":
    main()
