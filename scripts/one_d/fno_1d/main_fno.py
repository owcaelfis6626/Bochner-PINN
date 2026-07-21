"""
End-to-end driver for the FNO operator.

Run with:
    python main_fno.py             # full pipeline (data -> train -> eval)
    python main_fno.py --eval-only # skip training, evaluate fno_best.pt
    python main_fno.py --quick     # tiny config for smoke-testing
"""

import argparse
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_fno import FNOConfig
from fno_train import train
from fno_eval import evaluate


def quick_cfg() -> FNOConfig:
    cfg = FNOConfig()
    cfg.n_train = 16
    cfg.n_val = 4
    cfg.n_test = 16
    cfg.epochs = 5
    cfg.modes_x = 8
    cfg.modes_t = 12
    cfg.width = 16
    cfg.n_layers = 2
    cfg.batch_size = 4
    cfg.log_every = 1
    cfg.output_dir = "results_fno_quick"
    cfg.data_dir = "data_fno_quick"
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = quick_cfg() if args.quick else FNOConfig()
    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"[FNO]  modes=({cfg.modes_x},{cfg.modes_t})  width={cfg.width}  "
          f"layers={cfg.n_layers}  in={cfg.in_channels} out={cfg.out_channels}")
    print(f"       train={cfg.n_train}  val={cfg.n_val}  test={cfg.n_test}  "
          f"epochs={cfg.epochs}")
    print(f"       jumps={'VG+Gamma' if cfg.use_jumps else 'OFF'}")

    if not args.eval_only:
        print("\n[1] Training FNO ...")
        log_path = os.path.join(cfg.output_dir, "fno_train.json")
        os.makedirs(cfg.output_dir, exist_ok=True)
        train(cfg, device, log_path=log_path)

    print("\n[2] Evaluating FNO ...")
    ckpt_path = os.path.join(cfg.output_dir, "fno_best.pt")
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"  [error] checkpoint not found: {ckpt_path}")
    metrics = evaluate(cfg, ckpt_path, device, out_dir=cfg.output_dir)

    print("\n[3] Summary")
    for ch in ("z", "y", "p"):
        m = metrics[ch]
        print(f"  {ch}:  sp_err={m['sp_err']:.4e}  "
              f"tp_err={m['tp_err']:.4e}  var_ratio={m['var_ratio']:.3f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
