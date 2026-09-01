"""Master experiment runner for Phase A: Scaled-up study runs across seeds and aggregators."""

import json
import os
import sys
import time

# Ensure numba JIT does not trigger Windows Application Control DLL block
os.environ["NUMBA_DISABLE_JIT"] = "1"

import numpy as np

# Allow running as `python src/run_study_experiments.py` from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.centralized import main as run_centralized
from src.config import (
    DATA_DIR,
    MODELS_DIR,
    MU_VALUES,
    N_LOCAL_EPOCHS,
    N_ROUNDS,
    PROTOCOLS,
    RESULTS_DIR,
    SEEDS,
    SHAP_ATTACK_SAMPLE_SIZE,
    SHAP_BACKGROUND_SIZE,
    STUDY_RESULTS_PATH,
)
from src.explain import compute_shap_divergence
from src.run_federated import train_fedavg
from src.run_federated_prox import train_fedprox


def run_all_study_experiments() -> dict:
    """Run all 6 federated configurations + centralized baseline + SHAP."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 80)
    print("PHASE A: SCALED-UP EXPERIMENT RUN (CICIoMT2024 Study Scale)")
    print(f"Data Dir: {DATA_DIR} | Rounds: {N_ROUNDS} | Epochs: {N_LOCAL_EPOCHS} | Seeds: {SEEDS} | Mus: {MU_VALUES}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Centralized Baseline Run
    # ------------------------------------------------------------------
    print("\n>>> [1/7] Running Centralized Baseline...")
    t0_cent = time.time()
    cent_metrics = run_centralized(data_dir=DATA_DIR, seed=42)
    print(f">>> Centralized Baseline completed in {time.time() - t0_cent:.1f}s")

    # ------------------------------------------------------------------
    # 2. Federated Runs (2 seeds x 3 aggregators = 6 runs)
    # ------------------------------------------------------------------
    runs_data = []
    run_idx = 1
    total_fed_runs = len(SEEDS) * (1 + len(MU_VALUES))

    for seed in SEEDS:
        # FedAvg run
        print(f"\n>>> [{run_idx + 1}/{total_fed_runs + 1}] Running FedAvg (seed={seed})...")
        t0 = time.time()
        model_path_avg = os.path.join(MODELS_DIR, f"study_fedavg_seed{seed}.pth")
        fedavg_res = train_fedavg(
            data_dir=DATA_DIR,
            seed=seed,
            n_rounds=N_ROUNDS,
            n_epochs=N_LOCAL_EPOCHS,
            save_path=model_path_avg,
        )
        print(f"    FedAvg (seed={seed}) training completed in {time.time() - t0:.1f}s. Computing SHAP...")

        # SHAP for FedAvg
        shap_res_avg = compute_shap_divergence(
            model_or_path=model_path_avg,
            data_dir=DATA_DIR,
            seed=seed,
            bg_size=SHAP_BACKGROUND_SIZE,
            attack_sample_size=SHAP_ATTACK_SAMPLE_SIZE,
            plot_prefix=f"study_fedavg_seed{seed}" if seed != 42 else "study_fedavg",
            title_suffix=f"Study FedAvg (seed={seed})",
            save_plots=(seed == 42),  # save visual artifacts for primary seed
        )

        runs_data.append({
            "seed": seed,
            "aggregator": "fedavg",
            "mu": None,
            "client_metrics": fedavg_res["client_metrics"],
            "shap_top10_per_client": shap_res_avg["client_top10"],
            "shap_top5_per_client": shap_res_avg["client_top5"],
            "jaccard_pairwise": shap_res_avg["jaccard_pairwise"],
            "shared_features": shap_res_avg["shared_features"],
        })
        run_idx += 1

        # FedProx runs
        for mu in MU_VALUES:
            print(f"\n>>> [{run_idx + 1}/{total_fed_runs + 1}] Running FedProx (mu={mu}, seed={seed})...")
            t0 = time.time()
            model_path_prox = os.path.join(MODELS_DIR, f"study_fedprox_mu{mu}_seed{seed}.pth")
            fedprox_res = train_fedprox(
                data_dir=DATA_DIR,
                seed=seed,
                mu=mu,
                n_rounds=N_ROUNDS,
                n_epochs=N_LOCAL_EPOCHS,
                save_path=model_path_prox,
            )
            print(f"    FedProx (mu={mu}, seed={seed}) training completed in {time.time() - t0:.1f}s. Computing SHAP...")

            shap_res_prox = compute_shap_divergence(
                model_or_path=model_path_prox,
                data_dir=DATA_DIR,
                seed=seed,
                bg_size=SHAP_BACKGROUND_SIZE,
                attack_sample_size=SHAP_ATTACK_SAMPLE_SIZE,
                plot_prefix=f"study_fedprox_mu{mu}_seed{seed}" if seed != 42 else f"study_fedprox_mu{mu}",
                title_suffix=f"Study FedProx (mu={mu}, seed={seed})",
                save_plots=(seed == 42),
            )

            runs_data.append({
                "seed": seed,
                "aggregator": "fedprox",
                "mu": mu,
                "client_metrics": fedprox_res["client_metrics"],
                "shap_top10_per_client": shap_res_prox["client_top10"],
                "shap_top5_per_client": shap_res_prox["client_top5"],
                "jaccard_pairwise": shap_res_prox["jaccard_pairwise"],
                "shared_features": shap_res_prox["shared_features"],
            })
            run_idx += 1

    # ------------------------------------------------------------------
    # 3. Compute Preliminary Noise Floor across Seeds (n=2)
    # ------------------------------------------------------------------
    noise_floor = {}
    aggregators = [("fedavg", None)] + [("fedprox", mu) for mu in MU_VALUES]

    for agg_name, mu in aggregators:
        key_label = f"{agg_name}_mu{mu}" if mu is not None else agg_name
        agg_runs = [r for r in runs_data if r["aggregator"] == agg_name and r["mu"] == mu]

        per_client_top5_stability = {}
        for proto in PROTOCOLS:
            top5_s1 = set(agg_runs[0]["shap_top5_per_client"].get(proto, []))
            top5_s2 = set(agg_runs[1]["shap_top5_per_client"].get(proto, []))
            overlap = len(top5_s1.intersection(top5_s2)) / max(len(top5_s1.union(top5_s2)), 1)
            per_client_top5_stability[proto] = {
                "seed42_top5": agg_runs[0]["shap_top5_per_client"].get(proto, []),
                "seed7_top5": agg_runs[1]["shap_top5_per_client"].get(proto, []),
                "seed_to_seed_jaccard": round(overlap, 4),
            }

        pairwise_spread = {}
        pairs = ["wifi-mqtt", "wifi-bluetooth", "mqtt-bluetooth"]
        for pair in pairs:
            j1 = agg_runs[0]["jaccard_pairwise"].get(pair, 0.0)
            j2 = agg_runs[1]["jaccard_pairwise"].get(pair, 0.0)
            pairwise_spread[pair] = {
                "seed42_jaccard": j1,
                "seed7_jaccard": j2,
                "abs_diff": round(abs(j1 - j2), 4),
                "mean_jaccard": round((j1 + j2) / 2.0, 4),
            }

        noise_floor[key_label] = {
            "per_client_top5_stability": per_client_top5_stability,
            "pairwise_divergence_spread": pairwise_spread,
            "status": "preliminary (n=2 seeds), not the full noise floor",
        }

    # ------------------------------------------------------------------
    # 4. POC vs Study Comparative Summary Data
    # ------------------------------------------------------------------
    poc_summary = {
        "scale": "POC (sampled subsets: wifi ~5k, mqtt ~3k, bt ~2k)",
        "fedavg_f1_macro": 0.8560,
        "fedprox_mu0.01_f1_macro": 0.8974,
        "jaccard_wifi_mqtt": {"fedavg": 0.6667, "fedprox_mu0.01": 0.4286},
        "jaccard_wifi_bluetooth": {"fedavg": 0.1111, "fedprox_mu0.01": 0.1111},
        "jaccard_mqtt_bluetooth": {"fedavg": 0.1111, "fedprox_mu0.01": 0.0000},
    }

    final_payload = {
        "run_metadata": {
            "n_rounds": N_ROUNDS,
            "n_local_epochs": N_LOCAL_EPOCHS,
            "data_dir": DATA_DIR,
            "seeds": SEEDS,
            "mu_values": MU_VALUES,
            "shap_bg_size": SHAP_BACKGROUND_SIZE,
            "shap_attack_size": SHAP_ATTACK_SAMPLE_SIZE,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "centralized": cent_metrics,
        "poc_baseline_reference": poc_summary,
        "runs": runs_data,
        "preliminary_noise_floor": noise_floor,
    }

    with open(STUDY_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)
    print(f"\n[OK] Saved all study experiment results to {STUDY_RESULTS_PATH}")

    # ------------------------------------------------------------------
    # 5. Print Comparative Summary Table (POC vs Study)
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("SUMMARY COMPARISON: POC SCALE vs. STUDY SCALE")
    print("=" * 90)

    # Primary seed=42 runs
    s42_runs = {f"{r['aggregator']}_{r['mu']}": r for r in runs_data if r["seed"] == 42}
    fedavg_s42 = s42_runs["fedavg_None"]
    fedprox_s42_001 = s42_runs["fedprox_0.01"]
    fedprox_s42_01 = s42_runs["fedprox_0.1"]

    print(f"{'Setting / Metric':<30} | {'POC Scale (5 rnds)':<22} | {'Study Scale (10 rnds, Seed 42)':<30}")
    print("-" * 90)

    # Performance
    poc_avg_mac = 0.8560
    study_avg_mac = np.mean([m["f1_macro"] for m in fedavg_s42["client_metrics"].values()])
    print(f"{'FedAvg Macro F1 (Avg)':<30} | {poc_avg_mac:<22.4f} | {study_avg_mac:<30.4f}")

    poc_prox_mac = 0.8974
    study_prox001_mac = np.mean([m["f1_macro"] for m in fedprox_s42_001["client_metrics"].values()])
    print(f"{'FedProx (mu=0.01) Macro F1':<30} | {poc_prox_mac:<22.4f} | {study_prox001_mac:<30.4f}")

    study_prox01_mac = np.mean([m["f1_macro"] for m in fedprox_s42_01["client_metrics"].values()])
    print(f"{'FedProx (mu=0.1) Macro F1':<30} | {'—':<22} | {study_prox01_mac:<30.4f}")

    # Per-client Macro F1s
    print("-" * 90)
    for proto in PROTOCOLS:
        p_avg = fedavg_s42["client_metrics"][proto]["f1_macro"]
        p_prox = fedprox_s42_001["client_metrics"][proto]["f1_macro"]
        print(f"  {proto.upper()} FedAvg / FedProx0.01:   | {'—':<22} | {p_avg:.4f} / {p_prox:.4f}")

    # Divergence Jaccard
    print("-" * 90)
    pairs = [("wifi-mqtt", "0.6667 / 0.4286"), ("wifi-bluetooth", "0.1111 / 0.1111"), ("mqtt-bluetooth", "0.1111 / 0.0000")]
    for pair, poc_val in pairs:
        j_avg = fedavg_s42["jaccard_pairwise"][pair]
        j_prox = fedprox_s42_001["jaccard_pairwise"][pair]
        print(f"  Jaccard {pair:<16} | {poc_val:<22} | {j_avg:.4f} (Avg) / {j_prox:.4f} (Prox0.01)")

    print("=" * 90)
    print("\nPreliminary 2-seed noise floor summary:")
    for agg_key, nf in noise_floor.items():
        print(f"  [{agg_key}] Seed-to-seed top-5 Jaccards: " +
              ", ".join([f"{proto}={d['seed_to_seed_jaccard']}" for proto, d in nf["per_client_top5_stability"].items()]))
    print()

    return final_payload


if __name__ == "__main__":
    run_all_study_experiments()
