"""Generate publication-ready figures from results/raw JSON files."""

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_ablation_bytes_and_latency():
    """Generates bytes/token and latency plots from Experiment 3 ablation JSONs."""
    exp3_files = glob.glob("results/raw/exp3/*_seed*.json")
    if not exp3_files:
        print("No Exp-3 files found to plot.")
        return

    # Aggregate across seeds
    variant_data = {}
    for fpath in exp3_files:
        with open(fpath) as f:
            data = json.load(f)
        variants = data.get("variants", {})
        for vname, vinfo in variants.items():
            sys_info = vinfo["systems"]
            if vname not in variant_data:
                variant_data[vname] = {
                    "wall_time": [],
                    "tok_s": [],
                    "dram_mb": [],
                    "ssd_mb": [],
                    "peak_rss": [],
                }
            variant_data[vname]["wall_time"].append(sys_info["wall_time_sec"])
            variant_data[vname]["tok_s"].append(sys_info["tok_per_sec"])
            variant_data[vname]["dram_mb"].append(sys_info["dram_bytes"] / (1024 * 1024))
            variant_data[vname]["ssd_mb"].append(sys_info["ssd_bytes"] / (1024 * 1024))
            variant_data[vname]["peak_rss"].append(sys_info["peak_rss_mb"])

    labels = list(variant_data.keys())
    dram_means = [np.mean(variant_data[k]["dram_mb"]) for k in labels]
    ssd_means = [np.mean(variant_data[k]["ssd_mb"]) for k in labels]
    wall_means = [np.mean(variant_data[k]["wall_time"]) for k in labels]
    wall_stds = [np.std(variant_data[k]["wall_time"]) for k in labels]

    Path("figures").mkdir(exist_ok=True)
    Path("paper/figures").mkdir(parents=True, exist_ok=True)

    # 1. Stacked Bar Chart of DRAM vs SSD bytes moved
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    x = np.arange(len(labels))
    width = 0.55

    ax.bar(x, dram_means, width, label="DRAM Bytes (MB)", color="#1f77b4", edgecolor="black")
    ax.bar(x, ssd_means, width, bottom=dram_means, label="SSD Bytes (MB)", color="#ff7f0e", edgecolor="black")

    ax.set_ylabel("Total Bytes Moved (MB)", fontsize=11, fontweight="bold")
    ax.set_title("Memory Hierarchy Byte Movement by Policy (Exp-3)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()

    fig.savefig("figures/bytes_per_method.png")
    fig.savefig("paper/figures/bytes_per_method.png")
    plt.close(fig)
    print("Saved figures/bytes_per_method.png")

    # 2. Wall clock latency vs predicted time (Exp-2)
    exp2_files = glob.glob("results/raw/exp2/*_seed*.json")
    if exp2_files:
        with open(exp2_files[0]) as f:
            e2 = json.load(f)
        modes = list(e2["modes"].keys())
        measured = [e2["modes"][m]["cold"]["wall_time_sec"] for m in modes]
        predicted = [e2["modes"][m]["cold"]["predicted_time_sec"] for m in modes]

        fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
        x_m = np.arange(len(modes))
        w = 0.35
        ax.bar(x_m - w/2, measured, w, label="Measured Wall Time (s)", color="#2ca02c", edgecolor="black")
        ax.bar(x_m + w/2, predicted, w, label="Predicted I/O Time (s)", color="#d62728", edgecolor="black")

        ax.set_ylabel("Time (seconds)", fontsize=11, fontweight="bold")
        ax.set_title("Cost Model Validation: Measured vs Predicted I/O Time", fontsize=12, fontweight="bold")
        ax.set_xticks(x_m)
        ax.set_xticklabels(modes, fontsize=10)
        ax.legend(frameon=True)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()

        fig.savefig("figures/predicted_vs_wall.png")
        fig.savefig("paper/figures/predicted_vs_wall.png")
        plt.close(fig)
        print("Saved figures/predicted_vs_wall.png")


if __name__ == "__main__":
    plot_ablation_bytes_and_latency()
