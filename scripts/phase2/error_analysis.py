"""
Error distribution analysis for trained Spectral-PINN models.

For each (problem, mode, seed) trained checkpoint we:
    * Predict on full (Nx, Ny, Nt) grid.
    * Compute pointwise absolute error |u_PINN - u_FD|.
    * Time-average to get spatial heatmap (Nx, Ny).
    * Space-average to get temporal profile (Nt,).
    * Per-mode quantile statistics (q25, q50, q75, q95, q99) of the
      pointwise error across the whole spacetime cube.
    * Save spatial heatmap PNG, temporal profile PNG, and a CSV row.

This produces the "where does the error live" diagnostic that the
paper's discussion section references when claiming the spectral loss
preferentially fixes high-frequency / late-time errors.

Output:
    error_analysis/<problem>_<mode>_seed<s>_spatial.png
    error_analysis/<problem>_<mode>_seed<s>_temporal.png
    error_analysis/<problem>_<mode>_seed<s>_freq.png
    error_analysis/summary.csv

Usage:
    python error_analysis.py --problem ac      --mode vanilla  --seeds 0 1 2
    python error_analysis.py --problem ac      --mode spectral --seeds 0 1 2
    python error_analysis.py --problem burgers --mode vanilla  --seeds 0 1 2
    ...
"""

from __future__ import annotations

import argparse
import csv
import os
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _get_problem(name, mode):
    if name == "ac":
        if mode == "vanilla":
            from config_pinn_ac import ConfigPINNAC as C
        else:
            from config_pinn_ac import ConfigPINNACSpectral as C
        from pinn_models import ACPINNSystem as Sys
        from trainer_ac import _build_realisation as B
        from eval_ac    import _full_grid_predict
        return C, Sys, B, _full_grid_predict, "ac", False
    if name == "heston":
        if mode == "vanilla":
            from config_pinn_heston import ConfigPINNHeston as C
        else:
            from config_pinn_heston import ConfigPINNHestonSpectral as C
        from pinn_models    import HestonPINNSystem as Sys
        from trainer_heston import _build_realisation as B
        from eval_heston    import _full_grid_predict
        return C, Sys, B, _full_grid_predict, "heston", True
    if name == "burgers":
        if mode == "vanilla":
            from config_pinn_burgers import ConfigPINNBurgers as C
        else:
            from config_pinn_burgers import ConfigPINNBurgersSpectral as C
        from pinn_models     import BurgersPINNSystem as Sys
        from trainer_burgers import _build_realisation as B
        from eval_burgers    import _full_grid_predict
        return C, Sys, B, _full_grid_predict, "burgers", False
    if name == "rda":
        if mode == "vanilla":
            from config_pinn_rda import ConfigPINNRDA as C
        else:
            from config_pinn_rda import ConfigPINNRDASpectral as C
        from pinn_models import RDAPINNSystem as Sys
        from trainer_rda import _build_realisation as B
        from eval_rda    import _full_grid_predict
        return C, Sys, B, _full_grid_predict, "rda", False
    if name == "fracdiff":
        if mode == "vanilla":
            from config_pinn_fracdiff import ConfigPINNFracDiff as C
        else:
            from config_pinn_fracdiff import ConfigPINNFracDiffSpectral as C
        from pinn_models      import FracDiffPINNSystem as Sys
        from trainer_fracdiff import _build_realisation as B
        from eval_fracdiff    import _full_grid_predict
        return C, Sys, B, _full_grid_predict, "fracdiff", False
    raise ValueError(name)


def _spectral_radial_psd(U: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Radially-averaged spatial PSD over time.  U shape: (Nx, Ny, Nt)."""
    Nx, Ny, Nt = U.shape
    Uhat = torch.fft.fftn(U, dim=(0, 1))
    psd  = (Uhat.abs() ** 2).mean(dim=-1).cpu().numpy()  # (Nx, Ny)

    kx = np.fft.fftfreq(Nx) * Nx
    ky = np.fft.fftfreq(Ny) * Ny
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    Kmag = np.sqrt(KX ** 2 + KY ** 2)

    K_max = int(np.floor(min(Nx, Ny) / 2))
    bins  = np.arange(K_max + 1)
    radial = np.zeros(K_max)
    for k in range(K_max):
        mask = (Kmag >= bins[k]) & (Kmag < bins[k + 1])
        if mask.any():
            radial[k] = psd[mask].mean()
    return bins[:K_max], radial


def _save_spatial_heatmap(err_xy, x_grid, y_grid, out_path, title):
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=150)
    im = ax.imshow(err_xy.T,
                   origin="lower",
                   extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]],
                   cmap="magma",
                   aspect="auto")
    plt.colorbar(im, ax=ax, label=r"$\langle |u_{PINN} - u_{FD}| \rangle_t$")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def _save_temporal_profile(err_t, t_grid, out_path, title):
    fig, ax = plt.subplots(figsize=(6.0, 3.5), dpi=150)
    ax.plot(t_grid, err_t, color="C3", lw=1.8)
    ax.set_xlabel("t"); ax.set_ylabel(r"$\langle |u_{PINN}-u_{FD}| \rangle_{x,y}$")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def _save_freq_psd(k, p_pinn, p_fd, out_path, title):
    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=150)
    ax.loglog(k[1:], p_fd[1:],   "k-",  lw=1.8, label="FD ground truth")
    ax.loglog(k[1:], p_pinn[1:], "C0--", lw=1.8, label="PINN")
    ax.set_xlabel("|k| (radial wavenumber)"); ax.set_ylabel("spatial PSD")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True,
                    choices=["ac", "heston", "burgers", "rda", "fracdiff"])
    ap.add_argument("--mode", choices=["vanilla", "spectral"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ckpt_dir", type=str, default=None)
    ap.add_argument("--out_dir",  type=str, default="error_analysis")
    args = ap.parse_args()

    Cfg, Sys, build_realisation, full_grid_predict, prefix, is_heston = \
        _get_problem(args.problem, args.mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = Cfg()
    ckpt_dir = args.ckpt_dir or cfg.output_dir
    os.makedirs(args.out_dir, exist_ok=True)

    summary_csv = os.path.join(args.out_dir, "summary.csv")
    summary_exists = os.path.exists(summary_csv)

    rows = []
    for seed in args.seeds:
        ckpt = os.path.join(
            ckpt_dir, f"{prefix}_{args.mode}_s{seed}.pt"
        )
        if not os.path.exists(ckpt):
            print(f"  [skip] {ckpt} missing", flush=True)
            continue
        print(f"  [analyse] {ckpt}", flush=True)
        st = torch.load(ckpt, map_location=device)
        system = Sys(cfg).to(device)
        system.load_state_dict(st["model"])
        system.eval()

        _, grids, fd = build_realisation(cfg, seed, device)
        x_grid, y_grid, t_grid = grids

        # We always reduce to a single scalar field for analysis.
        # For Heston we run analysis on S only (asset price is the
        # primary observable in finance); v could be done separately.
        if is_heston:
            pred_pair = full_grid_predict(system, grids, device)
            pred = pred_pair[0]   # S
            fd_field = fd[0]      # S_fd
            tag = f"{prefix}_S"
        else:
            pred = full_grid_predict(system, grids, device)
            fd_field = fd
            tag = prefix

        err = (pred - fd_field).abs()                    # (Nx, Ny, Nt)
        err_xy_avg = err.mean(dim=-1).cpu().numpy()       # spatial heatmap
        err_t_avg  = err.mean(dim=(0, 1)).cpu().numpy()    # temporal profile

        # Quantiles over the whole spacetime cube
        flat = err.reshape(-1).cpu().numpy()
        q25, q50, q75, q95, q99 = np.quantile(
            flat, [0.25, 0.50, 0.75, 0.95, 0.99]
        )

        # Spatial PSD
        k, p_pinn = _spectral_radial_psd(pred.cpu())
        _, p_fd   = _spectral_radial_psd(fd_field.cpu())

        # Save plots
        spatial_png  = os.path.join(
            args.out_dir,
            f"{tag}_{args.mode}_seed{seed}_spatial.png")
        temporal_png = os.path.join(
            args.out_dir,
            f"{tag}_{args.mode}_seed{seed}_temporal.png")
        freq_png     = os.path.join(
            args.out_dir,
            f"{tag}_{args.mode}_seed{seed}_freqPSD.png")
        title = f"{tag} | {args.mode} | seed {seed}"
        _save_spatial_heatmap(err_xy_avg, x_grid, y_grid,
                              spatial_png, title + "  - spatial |error|")
        _save_temporal_profile(err_t_avg, t_grid,
                               temporal_png, title + "  - temporal |error|")
        _save_freq_psd(k, p_pinn, p_fd,
                       freq_png, title + "  - radial spatial PSD")

        rows.append({
            "problem":  args.problem,
            "mode":     args.mode,
            "seed":     seed,
            "mean_err": float(flat.mean()),
            "q25": q25, "q50": q50, "q75": q75, "q95": q95, "q99": q99,
            "max_err": float(flat.max()),
            "spatial_png":  os.path.basename(spatial_png),
            "temporal_png": os.path.basename(temporal_png),
            "freq_png":     os.path.basename(freq_png),
        })
        print(f"    mean={flat.mean():.4e} q50={q50:.4e} "
              f"q95={q95:.4e} max={flat.max():.4e}", flush=True)

    if not rows:
        print("[error_analysis] no checkpoints processed")
        return

    keys = list(rows[0].keys())
    write_header = not summary_exists
    with open(summary_csv, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[error_analysis] appended {len(rows)} rows -> {summary_csv}",
          flush=True)


if __name__ == "__main__":
    main()
