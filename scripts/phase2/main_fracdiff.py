"""
Fractional Diffusion PINN multi-seed launcher.
"""

from __future__ import annotations

import argparse
import os
import time
import torch

try:
    from .config_pinn_fracdiff import ConfigPINNFracDiff, ConfigPINNFracDiffSpectral
    from .trainer_fracdiff import train_one
except ImportError:
    from config_pinn_fracdiff import ConfigPINNFracDiff, ConfigPINNFracDiffSpectral
    from trainer_fracdiff import train_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["vanilla", "spectral"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    args = ap.parse_args()

    if args.mode == "vanilla":
        cfg = ConfigPINNFracDiff()
    else:
        cfg = ConfigPINNFracDiffSpectral()
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main_fracdiff] mode={args.mode} device={device} "
          f"seeds={args.seeds} epochs={cfg.num_epochs}", flush=True)
    print(f"[main_fracdiff] output_dir={cfg.output_dir}", flush=True)

    os.makedirs(cfg.output_dir, exist_ok=True)

    overall_t0 = time.time()
    for seed in args.seeds:
        print(f"\n{'='*60}\n  Seed {seed} (mode={args.mode})\n{'='*60}",
              flush=True)
        save_path = os.path.join(cfg.output_dir,
                                 f"fracdiff_{args.mode}_s{seed}.pt")
        train_one(cfg, seed=seed, device=device, mode=args.mode,
                  save_path=save_path)
    print(f"\n[main_fracdiff] all seeds done in "
          f"{(time.time()-overall_t0)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
