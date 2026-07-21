"""
Spectral-loss ablation runner.

For a given problem class, runs the training loop with different
combinations of the three spectral loss components disabled, so we can
isolate which loss component drives the improvement over vanilla PINN.

Variants:
    full      : w_spec_xy=*, w_spec_t=*, w_var=*    (all three)
    xy_only   : w_spec_xy=*, w_spec_t=0, w_var=0
    t_only    : w_spec_xy=0, w_spec_t=*, w_var=0
    var_only  : w_spec_xy=0, w_spec_t=0, w_var=*
    xy_t      : w_spec_xy=*, w_spec_t=*, w_var=0     (no variance)
    xy_var    : w_spec_xy=*, w_spec_t=0, w_var=*     (no temporal)
    t_var     : w_spec_xy=0, w_spec_t=*, w_var=*     (no spatial)

(``vanilla`` = none of these is run separately by main_<problem>.py.)

Usage:
    python run_ablation.py --problem ac      --variants xy_only t_only var_only --seeds 0 1 2
    python run_ablation.py --problem burgers --variants xy_only t_only var_only --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import os
import time
import torch


# Map problem -> (cfg_class, trainer.train_one, ckpt_prefix)
def _get_problem(name):
    if name == "ac":
        from config_pinn_ac import ConfigPINNACSpectral as C
        from trainer_ac import train_one as T
        return C, T, "ac"
    if name == "heston":
        from config_pinn_heston import ConfigPINNHestonSpectral as C
        from trainer_heston import train_one as T
        return C, T, "heston"
    if name == "burgers":
        from config_pinn_burgers import ConfigPINNBurgersSpectral as C
        from trainer_burgers import train_one as T
        return C, T, "burgers"
    if name == "rda":
        from config_pinn_rda import ConfigPINNRDASpectral as C
        from trainer_rda import train_one as T
        return C, T, "rda"
    if name == "fracdiff":
        from config_pinn_fracdiff import ConfigPINNFracDiffSpectral as C
        from trainer_fracdiff import train_one as T
        return C, T, "fracdiff"
    raise ValueError(f"unknown problem: {name}")


# Ablation variant -> (w_spec_xy_factor, w_spec_t_factor, w_var_factor)
# Factors multiply the original config weight; 0 = disabled.
ABLATIONS = {
    "full":     (1.0, 1.0, 1.0),
    "xy_only":  (1.0, 0.0, 0.0),
    "t_only":   (0.0, 1.0, 0.0),
    "var_only": (0.0, 0.0, 1.0),
    "xy_t":     (1.0, 1.0, 0.0),
    "xy_var":   (1.0, 0.0, 1.0),
    "t_var":    (0.0, 1.0, 1.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem",
                    choices=["ac", "heston", "burgers", "rda", "fracdiff"],
                    required=True)
    ap.add_argument("--variants", type=str, nargs="+",
                    default=["xy_only", "t_only", "var_only"],
                    choices=list(ABLATIONS.keys()))
    ap.add_argument("--seeds",  type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--root_out", type=str, default="results_ablation")
    args = ap.parse_args()

    Cfg, train_one, prefix = _get_problem(args.problem)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ablation] problem={args.problem} device={device} "
          f"variants={args.variants} seeds={args.seeds}", flush=True)

    overall_t0 = time.time()
    for variant in args.variants:
        fxy, ft, fvar = ABLATIONS[variant]
        var_dir = os.path.join(args.root_out, f"{prefix}_{variant}")
        os.makedirs(var_dir, exist_ok=True)
        print(f"\n{'#'*70}\n# ablation: {prefix}/{variant}  "
              f"(xy={fxy}, t={ft}, var={fvar})\n{'#'*70}", flush=True)

        for seed in args.seeds:
            cfg = Cfg()
            if args.epochs:
                cfg.num_epochs = args.epochs
            # apply ablation by zeroing spectral loss weights
            cfg.w_spec_xy = cfg.w_spec_xy * fxy
            cfg.w_spec_t  = cfg.w_spec_t  * ft
            cfg.w_var     = cfg.w_var     * fvar

            # also pin output_dir so checkpoints land in the variant dir
            cfg.output_dir = var_dir

            save_path = os.path.join(
                var_dir, f"{prefix}_{variant}_s{seed}.pt"
            )
            print(f"\n--- seed {seed} ---", flush=True)
            train_one(cfg, seed=seed, device=device, mode="spectral",
                      save_path=save_path,
                      ablation={
                          "variant": variant,
                          "w_spec_xy": cfg.w_spec_xy,
                          "w_spec_t":  cfg.w_spec_t,
                          "w_var":     cfg.w_var,
                      })

    print(f"\n[ablation] all variants done in "
          f"{(time.time() - overall_t0)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
