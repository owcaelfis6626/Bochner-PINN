"""
Generate publication-quality figures for the Spectral PINN paper, with
B-PINN baseline included. Outputs PDF + PNG in results_eval/figures/paper/.

Figures produced:
  fig_variance_gap.{pdf,png}    -- variance ratio bar chart, 3 methods, all problems
  fig_logpsd_bars.{pdf,png}     -- log-PSD MSE bar chart on log scale (the 113x gap)
  fig_y_bimodality.{pdf,png}    -- per-seed strip plot of hierarchical y-channel
  fig_method_compare.{pdf,png}  -- 2x2 small-multiples summary
"""

from __future__ import annotations
import os, json, csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SPDE = os.path.dirname(ROOT)
RES_EVAL = os.path.join(ROOT, "results_eval")
OUT = os.path.join(RES_EVAL, "figures", "paper")
os.makedirs(OUT, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
COL = {"vanilla": "#7f7f7f", "bpinn": "#d62728", "spectral": "#1f77b4"}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200)
    plt.close(fig)
    print(f"  wrote {name}.{{pdf,png}}")


# ── Load data ──────────────────────────────────────────────────────────

def load_burgers_per_seed():
    """vanilla + spectral, per-seed."""
    rows = list(csv.DictReader(open(os.path.join(RES_EVAL, "metrics_burgers_vg.csv"))))
    out = {"vanilla": {"vr": [], "sp": [], "tp": []},
           "spectral": {"vr": [], "sp": [], "tp": []}}
    for r in rows:
        m = out[r["mode"]]
        m["vr"].append(float(r["var_ratio"]))
        m["sp"].append(float(r["sp_err"]))
        m["tp"].append(float(r["tp_err"]))
    return out


def load_burgers_bpinn_per_seed():
    j = json.load(open(os.path.join(RES_EVAL, "bpinn_baseline_metrics.json")))
    out = {"vr": [], "sp": [], "tp": []}
    for s in j["burgers_vg"]:
        out["vr"].append(j["burgers_vg"][s]["var_ratio"])
        out["sp"].append(j["burgers_vg"][s]["log_sp_mse"])
        out["tp"].append(j["burgers_vg"][s]["log_tp_mse"])
    return out


def load_hier_per_seed():
    """spectral per-seed per-channel; vanilla mean/std per channel."""
    j = json.load(open(os.path.join(SPDE, "results_pinn_2d_spectral",
                                    "pinn_spectral_eval.json")))
    out = {ch: {"vr": [], "sp": [], "tp": []} for ch in ("z", "y", "p")}
    for sd, d in j["per_seed"].items():
        for ch in ("z", "y", "p"):
            for ts, m in d[ch].items():
                out[ch]["vr"].append(m["var_ratio"])
                out[ch]["sp"].append(m["sp_err"])
                out[ch]["tp"].append(m["tp_err"])

    j2 = json.load(open(os.path.join(SPDE, "results_pinn_2d", "pinn_ensemble_eval.json")))
    vanilla = {}
    for ch in ("z", "y", "p"):
        vanilla[ch] = {
            "vr": (j2["pinn_ensemble"][ch]["var_ratio_mean"],
                   j2["pinn_ensemble"][ch]["var_ratio_std"]),
            "sp": (j2["pinn_ensemble"][ch]["sp_err_mean"],
                   j2["pinn_ensemble"][ch]["sp_err_std"]),
            "tp": (j2["pinn_ensemble"][ch]["tp_err_mean"],
                   j2["pinn_ensemble"][ch]["tp_err_std"]),
        }
    return out, vanilla


def load_hier_bpinn_per_seed():
    j = json.load(open(os.path.join(RES_EVAL, "bpinn_baseline_metrics.json")))
    out = {ch: {"vr": [], "sp": [], "tp": []} for ch in ("z", "y", "p")}
    for key, m in j["main_2d"].items():
        seed, ch = key.rsplit("_", 1)
        out[ch]["vr"].append(m["var_ratio"])
        out[ch]["sp"].append(m["log_sp_mse"])
        out[ch]["tp"].append(m["log_tp_mse"])
    return out


# ── Figure 1: variance gap closure ─────────────────────────────────────

def fig_variance_gap():
    burg = load_burgers_per_seed()
    burg_b = load_burgers_bpinn_per_seed()
    hier, hier_v = load_hier_per_seed()
    hier_b = load_hier_bpinn_per_seed()

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 3.3))
    cats = ["Burgers-VG", "Hier. $z$", "Hier. $y$", "Hier. $p$"]

    def m_and_s(values):
        return np.mean(values), np.std(values)

    vanilla_means = [
        m_and_s(burg["vanilla"]["vr"]),
        hier_v["z"]["vr"],
        hier_v["y"]["vr"],
        hier_v["p"]["vr"],
    ]
    bpinn_means = [
        m_and_s(burg_b["vr"]),
        m_and_s(hier_b["z"]["vr"]),
        m_and_s(hier_b["y"]["vr"]),
        m_and_s(hier_b["p"]["vr"]),
    ]
    spectral_means = [
        m_and_s(burg["spectral"]["vr"]),
        m_and_s(hier["z"]["vr"]),
        m_and_s(hier["y"]["vr"]),
        m_and_s(hier["p"]["vr"]),
    ]

    x = np.arange(len(cats))
    w = 0.27
    for i, (label, data, c) in enumerate([
        ("Vanilla PINN",   vanilla_means,  COL["vanilla"]),
        ("B-PINN",         bpinn_means,    COL["bpinn"]),
        ("Spectral PINN",  spectral_means, COL["spectral"]),
    ]):
        means = [d[0] for d in data]
        stds  = [d[1] for d in data]
        ax.bar(x + (i - 1) * w, means, w, yerr=stds, capsize=3,
               label=label, color=c, edgecolor="black", linewidth=0.4)

    ax.axhline(1.0, ls="--", lw=0.8, color="black", alpha=0.5)
    ax.text(len(cats) - 0.5, 1.03, "target = 1.0", fontsize=8,
            ha="right", va="bottom", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel(r"variance ratio $\mathrm{Var}(u_\theta)/\mathrm{Var}(u_{\mathrm{FD}})$")
    ax.set_ylim(-0.05, 1.6)
    ax.set_title("Variance gap closure across methods and problems")
    ax.legend(frameon=False, loc="upper right")

    save(fig, "fig_variance_gap")


# ── Figure 2: log-PSD MSE on log-y axis ────────────────────────────────

def fig_logpsd_bars():
    burg = load_burgers_per_seed()
    burg_b = load_burgers_bpinn_per_seed()
    hier, hier_v = load_hier_per_seed()
    hier_b = load_hier_bpinn_per_seed()

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), sharey=True)

    def m_and_s(values):
        return np.mean(values), np.std(values)

    cats = ["Burgers-VG", "Hier. $z$", "Hier. $y$", "Hier. $p$"]
    sp_v = [m_and_s(burg["vanilla"]["sp"]), hier_v["z"]["sp"],
            hier_v["y"]["sp"], hier_v["p"]["sp"]]
    sp_b = [m_and_s(burg_b["sp"]), m_and_s(hier_b["z"]["sp"]),
            m_and_s(hier_b["y"]["sp"]), m_and_s(hier_b["p"]["sp"])]
    sp_s = [m_and_s(burg["spectral"]["sp"]), m_and_s(hier["z"]["sp"]),
            m_and_s(hier["y"]["sp"]), m_and_s(hier["p"]["sp"])]
    tp_v = [m_and_s(burg["vanilla"]["tp"]), hier_v["z"]["tp"],
            hier_v["y"]["tp"], hier_v["p"]["tp"]]
    tp_b = [m_and_s(burg_b["tp"]), m_and_s(hier_b["z"]["tp"]),
            m_and_s(hier_b["y"]["tp"]), m_and_s(hier_b["p"]["tp"])]
    tp_s = [m_and_s(burg["spectral"]["tp"]), m_and_s(hier["z"]["tp"]),
            m_and_s(hier["y"]["tp"]), m_and_s(hier["p"]["tp"])]

    x = np.arange(len(cats)); w = 0.27
    for ax, (data_v, data_b, data_s, title) in zip(
        axes,
        [(sp_v, sp_b, sp_s, "log-spatial PSD MSE"),
         (tp_v, tp_b, tp_s, "log-temporal PSD MSE")]):
        for i, (label, data, c) in enumerate([
            ("Vanilla PINN",   data_v, COL["vanilla"]),
            ("B-PINN",         data_b, COL["bpinn"]),
            ("Spectral PINN",  data_s, COL["spectral"]),
        ]):
            means = np.array([d[0] for d in data])
            stds  = np.array([d[1] for d in data])
            ax.bar(x + (i - 1) * w, means, w,
                   yerr=stds, capsize=3,
                   label=label, color=c, edgecolor="black", linewidth=0.4)
        ax.set_yscale("log")
        ax.set_xticks(x); ax.set_xticklabels(cats)
        ax.set_title(title)
        ax.set_ylabel("MSE  (lower = better)")

    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Spectrum matching across methods", y=1.02)

    save(fig, "fig_logpsd_bars")


# ── Figure 3: y-channel bimodality strip plot ──────────────────────────

def fig_y_outlier():
    """Per-seed strip plot for y channel: 7 of 8 seeds converge low,
    one seed lands high. Honest framing of the std=0.44 outlier."""
    j = json.load(open(os.path.join(SPDE, "results_pinn_2d_spectral",
                                    "pinn_spectral_eval.json")))
    seeds = sorted(j["per_seed"].keys(), key=int)
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.2))
    for i, sd in enumerate(seeds):
        vrs = [m["var_ratio"] for m in j["per_seed"][sd]["y"].values()]
        outlier = np.mean(vrs) > 1.0
        c = "#d62728" if outlier else COL["spectral"]
        x = np.full(len(vrs), i) + np.random.default_rng(int(sd)).uniform(-0.12, 0.12, len(vrs))
        ax.scatter(x, vrs, s=28, alpha=0.85, color=c,
                   edgecolor="black", linewidth=0.4,
                   label="outlier seed" if outlier and i == 0 else None)
    ax.axhline(1.0, ls="--", lw=0.8, color="black", alpha=0.5)
    ax.text(len(seeds) - 0.5, 1.03, "target = 1.0", fontsize=8,
            ha="right", va="bottom", alpha=0.7)
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f"s{sd}" for sd in seeds])
    ax.set_xlabel("training seed")
    ax.set_ylabel(r"variance ratio (4 test seeds per training seed)")
    ax.set_ylim(0.0, 2.2)
    ax.set_title(r"$y$-channel of hierarchical SPDE: 7/8 seeds converge low, seed 1 is an outlier")

    # Annotate
    ax.annotate("seed 1 converges to\na different attractor\n(var. ratio $\\approx$ 1.96)",
                xy=(0, 1.96), xytext=(2.5, 1.85),
                fontsize=9, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8))

    save(fig, "fig_y_outlier")


# ── Figure 4: 2x2 method-compare summary ───────────────────────────────

def fig_method_compare():
    """Compact summary: var_ratio + log-PSD on Burgers-VG and Hier-z."""
    burg = load_burgers_per_seed()
    burg_b = load_burgers_bpinn_per_seed()
    hier, hier_v = load_hier_per_seed()
    hier_b = load_hier_bpinn_per_seed()

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5))

    def panel(ax, vals, ylabel, title, log=False, target=None):
        labels = ["Vanilla", "B-PINN", "Spectral"]
        colors = [COL["vanilla"], COL["bpinn"], COL["spectral"]]
        means = [v[0] for v in vals]
        stds  = [v[1] for v in vals]
        ax.bar(labels, means, yerr=stds, capsize=4, color=colors,
               edgecolor="black", linewidth=0.5)
        if log:
            ax.set_yscale("log")
        if target is not None:
            ax.axhline(target, ls="--", lw=0.8, color="black", alpha=0.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)

    def ms(x): return (np.mean(x), np.std(x))

    # Burgers-VG var_ratio
    panel(axes[0, 0],
          [ms(burg["vanilla"]["vr"]), ms(burg_b["vr"]), ms(burg["spectral"]["vr"])],
          "variance ratio", "Burgers-VG", target=1.0)
    # Burgers-VG log-spat PSD
    panel(axes[0, 1],
          [ms(burg["vanilla"]["sp"]), ms(burg_b["sp"]), ms(burg["spectral"]["sp"])],
          "log-spatial PSD MSE", "Burgers-VG (log scale)", log=True)
    # Hier z var_ratio
    panel(axes[1, 0],
          [hier_v["z"]["vr"], ms(hier_b["z"]["vr"]), ms(hier["z"]["vr"])],
          "variance ratio", r"Hierarchical SPDE — $z$ channel", target=1.0)
    # Hier z log-spat PSD
    panel(axes[1, 1],
          [hier_v["z"]["sp"], ms(hier_b["z"]["sp"]), ms(hier["z"]["sp"])],
          "log-spatial PSD MSE", r"Hier. $z$ channel (log scale)", log=True)

    fig.suptitle("Method comparison: variance ratio (left) and spectrum matching (right)",
                 y=1.0)
    save(fig, "fig_method_compare")


def main():
    fig_variance_gap()
    fig_logpsd_bars()
    fig_y_outlier()
    fig_method_compare()
    print(f"\nAll figures in {OUT}")


if __name__ == "__main__":
    main()
