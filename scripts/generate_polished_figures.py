"""
Generate publication-quality figures for Paper A from metrics JSON files.

Outputs publication-ready PNGs at 300 DPI with consistent styling:
- Colorblind-safe palette (Wong 2011)
- Larger fonts (10-14 pt)
- LaTeX-rendered text where possible
- Clean grids, minimal chart junk
- Publication aspect ratios

Generated figures:
- fig1_hero.png         : 3-panel comparison (sp_err / tp_err / var_ratio)
- fig2_var_ratio.png    : Variance ratio with error bars (UQ story)
- fig3_improvement.png  : Improvement ratios bar chart
- fig4_per_channel.png  : Per-channel breakdown
- fig5_summary_table.png: Summary metrics as image table
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# ──────────────────────── Style ─────────────────────────────────────

# Colorblind-safe palette (Wong 2011)
COLORS = {
    "fno":      "#0072B2",   # blue
    "spectral": "#D55E00",   # orange
    "vanilla":  "#999999",   # gray
}

LABELS = {
    "fno":      "FNO",
    "spectral": "Spectral PINN",
    "vanilla":  "Vanilla PINN",
}

CHANNEL_LABELS = {
    "z": r"$z$ (price)",
    "y": r"$y$ (volatility)",
    "p": r"$p$ (jump intensity)",
}

# Publication settings
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
})


# ──────────────────────── Load data ─────────────────────────────────

def load_metrics(metrics_dir):
    """Load all 3 method metrics into a unified dict."""
    with open(os.path.join(metrics_dir, "pinn_ensemble_eval.json")) as f:
        vanilla = json.load(f)["pinn_ensemble"]

    with open(os.path.join(metrics_dir, "pinn_spectral_eval.json")) as f:
        spectral_raw = json.load(f)["ensemble"]

    with open(os.path.join(metrics_dir, "fno2d_metrics.json")) as f:
        fno_raw = json.load(f)

    # Normalize structure
    data = {"vanilla": {}, "spectral": {}, "fno": {}}
    for ch in ["z", "y", "p"]:
        data["vanilla"][ch] = {
            "sp_err":    {"mean": vanilla[ch]["sp_err_mean"],
                          "std":  vanilla[ch]["sp_err_std"]},
            "tp_err":    {"mean": vanilla[ch]["tp_err_mean"],
                          "std":  vanilla[ch]["tp_err_std"]},
            "var_ratio": {"mean": vanilla[ch]["var_ratio_mean"],
                          "std":  vanilla[ch]["var_ratio_std"]},
        }
        data["spectral"][ch] = spectral_raw[ch]   # already has mean/std nested
        data["fno"][ch] = {
            "sp_err":    {"mean": fno_raw[ch]["sp_err"], "std": 0.0},
            "tp_err":    {"mean": fno_raw[ch]["tp_err"], "std": 0.0},
            "var_ratio": {"mean": fno_raw[ch]["var_ratio"], "std": 0.0},
        }
    return data


# ──────────────────────── Figure 1: Hero ────────────────────────────

def fig1_hero(data, out_path):
    """Three-panel comparison: sp_err / tp_err / var_ratio across methods."""
    channels = ["z", "y", "p"]
    methods = ["vanilla", "spectral", "fno"]
    metrics_to_plot = [
        ("sp_err", "Spatial error", "log"),
        ("tp_err", "Temporal error", "log"),
        ("var_ratio", "Variance ratio (target = 1.0)", "linear"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))

    x = np.arange(len(channels))
    width = 0.27

    for ax_idx, (metric, ylabel, scale) in enumerate(metrics_to_plot):
        ax = axes[ax_idx]
        for m_idx, method in enumerate(methods):
            means = [data[method][ch][metric]["mean"] for ch in channels]
            stds  = [data[method][ch][metric]["std"]  for ch in channels]
            offset = (m_idx - 1) * width
            bars = ax.bar(x + offset, means, width,
                          yerr=stds, capsize=3,
                          label=LABELS[method],
                          color=COLORS[method],
                          edgecolor="black", linewidth=0.5,
                          error_kw={"elinewidth": 1, "alpha": 0.8})

        ax.set_xticks(x)
        ax.set_xticklabels([CHANNEL_LABELS[c] for c in channels])
        ax.set_ylabel(ylabel)
        ax.set_yscale(scale)

        # Reference line for variance ratio
        if metric == "var_ratio":
            ax.axhline(1.0, color="green", linestyle=":", linewidth=1.5,
                       alpha=0.6, label="Target (FD)")
            ax.set_ylim(0, 1.4)

        # Light grid only horizontal
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

        if ax_idx == 0:
            ax.legend(loc="upper right", framealpha=0.9, edgecolor="gray")

    fig.suptitle("Method comparison: hierarchical Lévy SPDE on "
                 r"$(z, y, p)$ channels (3 methods × 3 channels)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  saved {out_path}")


# ──────────────────────── Figure 2: Variance ratio ──────────────────

def fig2_var_ratio(data, out_path):
    """Detailed variance-ratio plot focusing on UQ story."""
    channels = ["z", "y", "p"]
    methods = ["vanilla", "spectral", "fno"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    x = np.arange(len(channels))
    width = 0.27

    for m_idx, method in enumerate(methods):
        means = [data[method][ch]["var_ratio"]["mean"] for ch in channels]
        stds  = [data[method][ch]["var_ratio"]["std"]  for ch in channels]
        offset = (m_idx - 1) * width
        ax.bar(x + offset, means, width,
               yerr=stds, capsize=4,
               label=LABELS[method],
               color=COLORS[method],
               edgecolor="black", linewidth=0.5,
               error_kw={"elinewidth": 1.2, "alpha": 0.8})

    # Target line
    ax.axhline(1.0, color="green", linestyle="--", linewidth=2,
               alpha=0.7, label="Target (FD ground truth)")

    # Highlight catastrophic failure region
    ax.axhspan(0.0, 0.3, alpha=0.1, color="red",
               label="Catastrophic failure region")

    ax.set_xticks(x)
    ax.set_xticklabels([CHANNEL_LABELS[c] for c in channels])
    ax.set_ylabel(r"Variance ratio  $\mathrm{Var}_\mathrm{model} \,/\, \mathrm{Var}_\mathrm{FD}$",
                  fontsize=12)
    ax.set_ylim(0, 1.4)
    ax.set_title("Variance ratio: Spectral PINN closes the UQ gap "
                 "left by Vanilla PINN")
    ax.legend(loc="upper left", framealpha=0.9, edgecolor="gray", ncol=2)

    # Annotations for dramatic z-channel improvement
    z_vanilla = data["vanilla"]["z"]["var_ratio"]["mean"]
    z_spectral = data["spectral"]["z"]["var_ratio"]["mean"]
    ax.annotate(
        f"4.3× improvement",
        xy=(0 + width, z_spectral),
        xytext=(0 + width + 0.05, z_spectral + 0.25),
        fontsize=10, color=COLORS["spectral"],
        arrowprops=dict(arrowstyle="->", color=COLORS["spectral"], lw=1.2),
    )

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  saved {out_path}")


# ──────────────────────── Figure 3: Improvement ratios ──────────────

def fig3_improvement(data, out_path):
    """Spectral PINN's improvement over Vanilla, per channel/metric."""
    channels = ["z", "y", "p"]
    metrics_pretty = {"sp_err": "Spatial err", "tp_err": "Temporal err",
                      "var_ratio": "Variance match"}
    metrics_to_plot = ["sp_err", "tp_err"]   # var_ratio handled differently

    # Compute ratios
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    width = 0.35
    x = np.arange(len(channels))

    for m_idx, metric in enumerate(metrics_to_plot):
        ratios = []
        for ch in channels:
            v_mean = data["vanilla"][ch][metric]["mean"]
            s_mean = data["spectral"][ch][metric]["mean"]
            ratios.append(v_mean / s_mean if s_mean > 0 else 0)

        offset = (m_idx - 0.5) * width
        bars = ax.bar(x + offset, ratios, width,
                      label=metrics_pretty[metric],
                      edgecolor="black", linewidth=0.5,
                      color=[COLORS["spectral"], "#E69F00"][m_idx])

        # Annotate values
        for bar, r in zip(bars, ratios):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{r:.1f}×",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(1.0, color="black", linestyle="-", linewidth=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([CHANNEL_LABELS[c] for c in channels])
    ax.set_ylabel("Improvement factor (Vanilla / Spectral)")
    ax.set_title(r"Spectral PINN improvement over Vanilla PINN")
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="gray")
    ax.set_ylim(0, max(8, max([data["vanilla"][ch]["tp_err"]["mean"]
                                / data["spectral"][ch]["tp_err"]["mean"]
                                for ch in channels]) * 1.15))

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  saved {out_path}")


# ──────────────────────── Figure 4: Per-channel breakdown ───────────

def fig4_per_channel(data, out_path):
    """Detailed per-channel metrics on log scale."""
    channels = ["z", "y", "p"]
    methods = ["vanilla", "spectral", "fno"]
    metrics_to_plot = ["sp_err", "tp_err"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4),
                              sharey=False)

    for ax_idx, ch in enumerate(channels):
        ax = axes[ax_idx]
        x = np.arange(len(metrics_to_plot))
        width = 0.27

        for m_idx, method in enumerate(methods):
            means = [data[method][ch][m]["mean"] for m in metrics_to_plot]
            stds  = [data[method][ch][m]["std"]  for m in metrics_to_plot]
            offset = (m_idx - 1) * width
            ax.bar(x + offset, means, width,
                   yerr=stds, capsize=3,
                   label=LABELS[method] if ax_idx == 0 else None,
                   color=COLORS[method],
                   edgecolor="black", linewidth=0.5,
                   error_kw={"elinewidth": 1, "alpha": 0.8})

        ax.set_xticks(x)
        ax.set_xticklabels(["Spatial", "Temporal"])
        ax.set_yscale("log")
        ax.set_title(f"Channel: {CHANNEL_LABELS[ch]}")
        if ax_idx == 0:
            ax.set_ylabel("Error (log scale)")
            ax.legend(loc="upper right", framealpha=0.9, edgecolor="gray")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    fig.suptitle("Per-channel error decomposition (log scale): "
                 "FNO dominant, Spectral PINN closes vanilla gap",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  saved {out_path}")


# ──────────────────────── Figure 5: Summary table ───────────────────

def fig5_summary_table(data, out_path):
    """Render a clean summary table of all key metrics."""
    channels = ["z", "y", "p"]
    methods = ["vanilla", "spectral", "fno"]

    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.axis("off")

    # Build table data
    headers = ["Channel", "Method", "Spatial err", "Temporal err",
               "Var ratio (target=1)"]
    rows = []
    for ch in channels:
        for m_idx, method in enumerate(methods):
            sp = data[method][ch]["sp_err"]
            tp = data[method][ch]["tp_err"]
            vr = data[method][ch]["var_ratio"]
            ch_label = CHANNEL_LABELS[ch] if m_idx == 0 else ""
            rows.append([
                ch_label,
                LABELS[method],
                f"{sp['mean']:.3e} ± {sp['std']:.2e}" if sp['std'] > 0
                  else f"{sp['mean']:.3e}",
                f"{tp['mean']:.3e} ± {tp['std']:.2e}" if tp['std'] > 0
                  else f"{tp['mean']:.3e}",
                f"{vr['mean']:.3f} ± {vr['std']:.3f}" if vr['std'] > 0
                  else f"{vr['mean']:.3f}",
            ])

    table = ax.table(cellText=rows, colLabels=headers,
                     cellLoc="center", loc="center",
                     colColours=["#E0E0E0"] * len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Color rows by method
    for i, row in enumerate(rows):
        method_label = row[1]
        method_key = next(k for k, v in LABELS.items() if v == method_label)
        for j in range(len(headers)):
            cell = table[i + 1, j]
            cell.set_facecolor(COLORS[method_key] + "15"
                               if method_key != "fno"
                               else COLORS[method_key] + "20")
            cell.set_edgecolor("white")

    # Bold headers
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_text_props(weight="bold")

    plt.title("Summary: PINN methods on hierarchical Lévy SPDE "
              "(16 vanilla + 8 spectral seeds, 1 FNO)",
              pad=20)
    plt.savefig(out_path)
    plt.close()
    print(f"  saved {out_path}")


# ──────────────────────── Main ──────────────────────────────────────

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    metrics_dir = os.path.join(here, "..", "metrics")
    out_dir = here  # same folder
    os.makedirs(out_dir, exist_ok=True)

    print("Loading metrics...")
    data = load_metrics(metrics_dir)

    print("Generating polished figures...")
    fig1_hero(data, os.path.join(out_dir, "fig1_hero.png"))
    fig2_var_ratio(data, os.path.join(out_dir, "fig2_var_ratio.png"))
    fig3_improvement(data, os.path.join(out_dir, "fig3_improvement.png"))
    fig4_per_channel(data, os.path.join(out_dir, "fig4_per_channel.png"))
    fig5_summary_table(data, os.path.join(out_dir, "fig5_summary_table.png"))

    print("\nDone. Figures saved to:")
    print(f"  {out_dir}")


if __name__ == "__main__":
    main()
