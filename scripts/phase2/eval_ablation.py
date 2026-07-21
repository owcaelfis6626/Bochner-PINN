"""
Aggregate ablation results across seeds.

Walks results_ablation/<problem>_<variant>/ folders, evaluates each
seed's checkpoint on the FD ground truth, and writes a summary CSV
containing mean +/- std for sp_err / tp_err / var_ratio per variant.

Usage:
    python eval_ablation.py --problem ac      --variants xy_only t_only var_only full
    python eval_ablation.py --problem burgers --variants xy_only t_only var_only full
"""

from __future__ import annotations

import argparse
import csv
import os
import numpy as np
import torch


def _get_problem(name):
    if name == "ac":
        from config_pinn_ac    import ConfigPINNACSpectral as C
        from pinn_models       import ACPINNSystem as Sys
        from trainer_ac        import _build_realisation as B
        from eval_ac           import _full_grid_predict, _metrics
        return C, Sys, B, _full_grid_predict, _metrics, "ac"
    if name == "heston":
        from config_pinn_heston import ConfigPINNHestonSpectral as C
        from pinn_models        import HestonPINNSystem as Sys
        from trainer_heston     import _build_realisation as B
        from eval_heston        import _full_grid_predict, _metrics
        return C, Sys, B, _full_grid_predict, _metrics, "heston"
    if name == "burgers":
        from config_pinn_burgers import ConfigPINNBurgersSpectral as C
        from pinn_models         import BurgersPINNSystem as Sys
        from trainer_burgers     import _build_realisation as B
        from eval_burgers        import _full_grid_predict, _metrics
        return C, Sys, B, _full_grid_predict, _metrics, "burgers"
    if name == "rda":
        from config_pinn_rda import ConfigPINNRDASpectral as C
        from pinn_models     import RDAPINNSystem as Sys
        from trainer_rda     import _build_realisation as B
        from eval_rda        import _full_grid_predict, _metrics
        return C, Sys, B, _full_grid_predict, _metrics, "rda"
    if name == "fracdiff":
        from config_pinn_fracdiff import ConfigPINNFracDiffSpectral as C
        from pinn_models           import FracDiffPINNSystem as Sys
        from trainer_fracdiff      import _build_realisation as B
        from eval_fracdiff         import _full_grid_predict, _metrics
        return C, Sys, B, _full_grid_predict, _metrics, "fracdiff"
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True,
                    choices=["ac", "heston", "burgers", "rda", "fracdiff"])
    ap.add_argument("--variants", type=str, nargs="+",
                    default=["full", "xy_only", "t_only", "var_only",
                             "xy_t", "xy_var", "t_var"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--root", type=str, default="results_ablation")
    ap.add_argument("--csv_out", type=str, default=None)
    args = ap.parse_args()

    Cfg, Sys, build_realisation, full_grid_predict, metrics, prefix = \
        _get_problem(args.problem)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = Cfg()
    csv_out = args.csv_out or os.path.join(
        args.root, f"summary_{args.problem}.csv")

    summary_rows = []
    detail_rows = []

    for variant in args.variants:
        var_dir = os.path.join(args.root, f"{prefix}_{variant}")
        if not os.path.isdir(var_dir):
            print(f"  [skip] {var_dir} missing", flush=True)
            continue
        sp_list, tp_list, vr_list = [], [], []
        for seed in args.seeds:
            ckpt = os.path.join(var_dir,
                                f"{prefix}_{variant}_s{seed}.pt")
            if not os.path.exists(ckpt):
                print(f"    [skip] {ckpt} missing", flush=True)
                continue
            print(f"  [eval] {ckpt}", flush=True)
            st = torch.load(ckpt, map_location=device)
            system = Sys(cfg).to(device)
            system.load_state_dict(st["model"])
            system.eval()

            _, grids, fd = build_realisation(cfg, seed, device)

            if args.problem == "heston":
                Sp, Vp = full_grid_predict(system, grids, device)
                S_fd, V_fd = fd
                sp_S, tp_S, vr_S = metrics(Sp, S_fd)
                sp_V, tp_V, vr_V = metrics(Vp, V_fd)
                sp = 0.5 * (sp_S + sp_V)
                tp = 0.5 * (tp_S + tp_V)
                vr = 0.5 * (vr_S + vr_V)
            else:
                pred = full_grid_predict(system, grids, device)
                sp, tp, vr = metrics(pred, fd)

            sp_list.append(sp); tp_list.append(tp); vr_list.append(vr)
            detail_rows.append({
                "problem": args.problem, "variant": variant, "seed": seed,
                "sp_err": sp, "tp_err": tp, "var_ratio": vr,
            })
            print(f"    sp={sp:.3f} tp={tp:.3f} vr={vr:.3f}", flush=True)

        if sp_list:
            sp_a = np.array(sp_list); tp_a = np.array(tp_list); vr_a = np.array(vr_list)
            summary_rows.append({
                "problem": args.problem, "variant": variant,
                "n_seeds": len(sp_list),
                "sp_mean":  sp_a.mean(),  "sp_std":  sp_a.std(),
                "tp_mean":  tp_a.mean(),  "tp_std":  tp_a.std(),
                "vr_mean":  vr_a.mean(),  "vr_std":  vr_a.std(),
            })

    if not summary_rows:
        print("[eval_ablation] nothing to write")
        return

    keys = list(summary_rows[0].keys())
    with open(csv_out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    print(f"\n[eval_ablation] wrote {csv_out}", flush=True)

    detail_csv = csv_out.replace(".csv", "_detail.csv")
    if detail_rows:
        keys = list(detail_rows[0].keys())
        with open(detail_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in detail_rows:
                w.writerow(r)
        print(f"[eval_ablation] wrote {detail_csv}", flush=True)

    print("\n[eval_ablation] summary:")
    print(f"{'variant':12s} {'sp_mean':>9s} {'sp_std':>8s} "
          f"{'tp_mean':>9s} {'tp_std':>8s} {'vr_mean':>9s} {'vr_std':>8s}")
    for r in summary_rows:
        print(f"{r['variant']:12s} "
              f"{r['sp_mean']:9.4f} {r['sp_std']:8.4f} "
              f"{r['tp_mean']:9.4f} {r['tp_std']:8.4f} "
              f"{r['vr_mean']:9.4f} {r['vr_std']:8.4f}")


if __name__ == "__main__":
    main()
