"""
Heston PINN evaluation:
    * sp_err  — log10 spatial-PSD MSE   (PINN vs FD)
    * tp_err  — log10 temporal-PSD MSE  (PINN vs FD)
    * var_ratio — Var(PINN) / Var(FD)

Reads checkpoints saved by main_heston.py and writes a CSV summary.

Usage:
    python eval_heston.py --mode vanilla  --seeds 0 1 2 3
    python eval_heston.py --mode spectral --seeds 0 1 2 3
"""

from __future__ import annotations

import argparse
import csv
import os
import numpy as np
import torch

try:
    from .config_pinn_heston import ConfigPINNHeston, ConfigPINNHestonSpectral
    from .pinn_models import HestonPINNSystem
    from .trainer_heston import (
        _build_realisation, _spatial_psd_xy, _temporal_psd,
    )
except ImportError:
    from config_pinn_heston import ConfigPINNHeston, ConfigPINNHestonSpectral
    from pinn_models import HestonPINNSystem
    from trainer_heston import (
        _build_realisation, _spatial_psd_xy, _temporal_psd,
    )


@torch.no_grad()
def _full_grid_predict(system, grids, device, batch=8192):
    x_grid, y_grid, t_grid = grids
    Nx, Ny, Nt = len(x_grid), len(y_grid), len(t_grid)
    XX, YY, TT = np.meshgrid(x_grid, y_grid, t_grid, indexing="ij")
    flat_x = torch.tensor(XX.reshape(-1, 1), dtype=torch.float32, device=device)
    flat_y = torch.tensor(YY.reshape(-1, 1), dtype=torch.float32, device=device)
    flat_t = torch.tensor(TT.reshape(-1, 1), dtype=torch.float32, device=device)
    N = flat_x.shape[0]
    S_out = torch.empty(N, 1, device=device)
    V_out = torch.empty(N, 1, device=device)
    for i in range(0, N, batch):
        xb = flat_x[i:i+batch]; yb = flat_y[i:i+batch]; tb = flat_t[i:i+batch]
        S_out[i:i+batch] = system.net_S(xb, yb, tb)
        V_out[i:i+batch] = system.net_v(xb, yb, tb)
    return (S_out.reshape(Nx, Ny, Nt),
            V_out.reshape(Nx, Ny, Nt))


def _metrics(pred, target):
    sp_p = _spatial_psd_xy(pred.unsqueeze(0)).squeeze(0)
    sp_t = _spatial_psd_xy(target.unsqueeze(0)).squeeze(0)
    tp_p = _temporal_psd(pred.unsqueeze(0)).squeeze(0)
    tp_t = _temporal_psd(target.unsqueeze(0)).squeeze(0)
    sp_err = ((torch.log10(sp_p.clamp(min=1e-12))
               - torch.log10(sp_t.clamp(min=1e-12))) ** 2).mean().item()
    tp_err = ((torch.log10(tp_p.clamp(min=1e-12))
               - torch.log10(tp_t.clamp(min=1e-12))) ** 2).mean().item()
    var_ratio = (pred.var() / target.var().clamp(min=1e-12)).item()
    return sp_err, tp_err, var_ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["vanilla", "spectral"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--ckpt_dir", type=str, default=None)
    ap.add_argument("--csv_out",  type=str, default=None)
    args = ap.parse_args()

    cfg = ConfigPINNHestonSpectral() if args.mode == "spectral" else ConfigPINNHeston()
    ckpt_dir = args.ckpt_dir or cfg.output_dir
    csv_out  = args.csv_out  or os.path.join(ckpt_dir, "metrics.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for seed in args.seeds:
        ckpt = os.path.join(ckpt_dir, f"heston_{args.mode}_s{seed}.pt")
        if not os.path.exists(ckpt):
            print(f"  [skip] {ckpt} missing", flush=True)
            continue
        print(f"  [eval] {ckpt}", flush=True)
        st = torch.load(ckpt, map_location=device)
        system = HestonPINNSystem(cfg).to(device)
        system.load_state_dict(st["model"])
        system.eval()

        _, grids, fields = _build_realisation(cfg, seed, device)
        S_fd, V_fd = fields
        S_pred, V_pred = _full_grid_predict(system, grids, device)

        sp_S, tp_S, vr_S = _metrics(S_pred, S_fd)
        sp_v, tp_v, vr_v = _metrics(V_pred, V_fd)
        rows.append({
            "seed": seed,
            "sp_err_S": sp_S, "tp_err_S": tp_S, "var_ratio_S": vr_S,
            "sp_err_v": sp_v, "tp_err_v": tp_v, "var_ratio_v": vr_v,
        })
        print(f"    S: sp={sp_S:.3f} tp={tp_S:.3f} vr={vr_S:.3f}  "
              f"v: sp={sp_v:.3f} tp={tp_v:.3f} vr={vr_v:.3f}", flush=True)

    if not rows:
        print("[eval_heston] no checkpoints found")
        return

    keys = list(rows[0].keys())
    with open(csv_out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[eval_heston] wrote {csv_out}", flush=True)

    # Summary
    arr = {k: np.array([r[k] for r in rows]) for k in keys if k != "seed"}
    print("\n[eval_heston] mean +/- std across seeds:")
    for k, v in arr.items():
        print(f"  {k:14s} {v.mean():8.4f} +/- {v.std():.4f}")


if __name__ == "__main__":
    main()
