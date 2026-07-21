"""
Distributional spectral evaluation for the trained FNO.

For each test seed: predict (Z, Y, P) given (noise, jumps, IC) inputs,
collect into an ensemble, then compare the ensemble's spatial / temporal
PSD against the FD ensemble (ground truth) — same metric the PINN side
uses, so numbers are directly comparable.

Outputs:
  - JSON with per-channel sp_err, tp_err, var_ratio
  - PNG comparing FD vs FNO PSDs (z, y, p panels)
"""

import os
import json
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fno_model import FNO2d
from fno_dataset import build_split, normalize_per_channel
from config_fno import FNOConfig


CHANNEL_NAMES = ["z", "y", "p"]


def _ensemble_psd_gpu(field_ens: torch.Tensor) -> tuple:
    """field_ens: (N, C, Nx, Nt) GPU tensor -> (sp[C, kx], tp[C, w]) CPU numpy.

    Uses torch.fft.rfft so computation stays on GPU (seconds, not minutes).
    """
    Fx = torch.fft.rfft(field_ens, dim=-2).abs() ** 2   # (N, C, kx, Nt)
    sp = Fx.mean(dim=(0, 3)).cpu().numpy()               # (C, kx)

    Ft = torch.fft.rfft(field_ens, dim=-1).abs() ** 2   # (N, C, Nx, w)
    tp = Ft.mean(dim=(0, 2)).cpu().numpy()               # (C, w)
    return sp, tp


def evaluate(fno_cfg: FNOConfig, ckpt_path: str, device: torch.device,
             out_dir: str = None):
    out_dir = out_dir or fno_cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)

    blob = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = FNO2d(
        in_channels=fno_cfg.in_channels,
        out_channels=fno_cfg.out_channels,
        modes_x=fno_cfg.modes_x,
        modes_t=fno_cfg.modes_t,
        width=fno_cfg.width,
        n_layers=fno_cfg.n_layers,
        activation=fno_cfg.activation,
    ).to(device)
    model.load_state_dict(blob["model"])
    model.eval()

    x_stats = {k: v.to(device) for k, v in blob["x_stats"].items()}
    x_stats_cpu = {k: v.cpu() for k, v in blob["x_stats"].items()}

    # ── Test split ─────────────────────────────────────────────────
    X_te, Y_te = build_split(fno_cfg, fno_cfg.n_test,
                             fno_cfg.base_seed_test, "test")
    X_te_n, _ = normalize_per_channel(X_te, x_stats_cpu)

    # ── Inference ──────────────────────────────────────────────────
    print(f"  [eval] running FNO on {fno_cfg.n_test} test seeds ...")
    t0 = time.time()
    preds = []
    with torch.no_grad():
        for i in range(0, fno_cfg.n_test, fno_cfg.batch_size):
            xb = X_te_n[i:i + fno_cfg.batch_size].to(device)
            pb = model(xb).cpu().numpy()
            preds.append(pb)
    Y_pred = np.concatenate(preds, axis=0)
    Y_true = Y_te.numpy()
    print(f"  [eval] inference done in {time.time()-t0:.1f}s")

    # ── PSDs ───────────────────────────────────────────────────────
    sp_p, tp_p = _ensemble_psd(Y_pred)
    sp_t, tp_t = _ensemble_psd(Y_true)

    eps = 1e-12
    metrics = {}
    for c, name in enumerate(CHANNEL_NAMES):
        sp_err = float(np.mean(
            (np.log(sp_p[c] + eps) - np.log(sp_t[c] + eps)) ** 2))
        tp_err = float(np.mean(
            (np.log(tp_p[c] + eps) - np.log(tp_t[c] + eps)) ** 2))
        var_p = float(Y_pred[:, c].var())
        var_t = float(Y_true[:, c].var())
        var_ratio = var_p / max(var_t, 1e-15)
        metrics[name] = {
            "sp_err": sp_err,
            "tp_err": tp_err,
            "var_ratio": var_ratio,
            "var_pred": var_p,
            "var_true": var_t,
        }
        print(f"    [{name}]  sp_err={sp_err:.4e}  "
              f"tp_err={tp_err:.4e}  var_ratio={var_ratio:.3f}")

    metrics["epoch_loaded"] = int(blob.get("epoch", -1))
    metrics["val_total_loaded"] = float(blob.get("val_total", float("nan")))

    with open(os.path.join(out_dir, "fno_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Plot ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    for c, name in enumerate(CHANNEL_NAMES):
        axes[c, 0].loglog(sp_t[c], "k-", lw=1.5, label="FD ensemble")
        axes[c, 0].loglog(sp_p[c], "r--", lw=1.5, label="FNO ensemble")
        axes[c, 0].set_title(f"Spatial PSD  ({name})")
        axes[c, 0].set_xlabel("wavenumber k")
        axes[c, 0].set_ylabel("|FFT|²")
        axes[c, 0].grid(True, alpha=0.3, which="both")
        if c == 0:
            axes[c, 0].legend()

        axes[c, 1].loglog(tp_t[c], "k-", lw=1.5, label="FD ensemble")
        axes[c, 1].loglog(tp_p[c], "r--", lw=1.5, label="FNO ensemble")
        axes[c, 1].set_title(f"Temporal PSD  ({name})")
        axes[c, 1].set_xlabel("frequency ω")
        axes[c, 1].grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fno_psd_comparison.png"),
                bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  [eval] saved -> {out_dir}/fno_metrics.json, fno_psd_comparison.png")

    return metrics
