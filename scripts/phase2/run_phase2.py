"""
Master sequential launcher for the Phase-2 expansion:

   1. Burgers   vanilla   (4 seeds) + eval
   2. Burgers   spectral  (4 seeds) + eval
   3. RDA       vanilla   (4 seeds) + eval
   4. RDA       spectral  (4 seeds) + eval
   5. FracDiff  vanilla   (4 seeds) + eval
   6. FracDiff  spectral  (4 seeds) + eval
   7. Ablation study (AC + Burgers, 3 variants x 3 seeds)
   8. Error-distribution analysis on every trained checkpoint

This produces the data needed to push Paper B from 7.5/10 to 8/10:
    * 4th, 5th and 6th problem class (generalisation breadth).
    * Spectral-loss ablation (which component matters).
    * Spatial / temporal / frequency error decomposition.

Run:
    python run_phase2.py
        # uses defaults: 4 seeds, 5000 epochs each
    python run_phase2.py --skip burgers_vanilla   # if you already have it
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


HERE = os.path.dirname(os.path.abspath(__file__))


def _run(cmd):
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=HERE)
    dt = (time.time() - t0) / 3600
    print(f">>> exit={rc}  elapsed={dt:.2f} h", flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--ablation_seeds", type=int, nargs="+",
                    default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--ablation_epochs", type=int, default=3000,
                    help="Ablation runs use fewer epochs to stay tractable.")
    ap.add_argument("--skip", type=str, nargs="*", default=[],
                    help="Stage names to skip "
                         "(burgers_vanilla, burgers_spectral, rda_vanilla, "
                         "rda_spectral, fracdiff_vanilla, fracdiff_spectral, "
                         "ablation, error_analysis)")
    args = ap.parse_args()

    seeds_str = [str(s) for s in args.seeds]
    epochs_arg = ["--epochs", str(args.epochs)] if args.epochs else []
    ab_seeds_str = [str(s) for s in args.ablation_seeds]

    # ---------------- Stage 1-6: training + eval --------------------
    plan = [
        ("burgers_vanilla",   "main_burgers.py",  "vanilla",
                              "eval_burgers.py"),
        ("burgers_spectral",  "main_burgers.py",  "spectral",
                              "eval_burgers.py"),
        ("rda_vanilla",       "main_rda.py",      "vanilla",
                              "eval_rda.py"),
        ("rda_spectral",      "main_rda.py",      "spectral",
                              "eval_rda.py"),
        ("fracdiff_vanilla",  "main_fracdiff.py", "vanilla",
                              "eval_fracdiff.py"),
        ("fracdiff_spectral", "main_fracdiff.py", "spectral",
                              "eval_fracdiff.py"),
    ]

    overall_t0 = time.time()
    for name, train_script, mode, eval_script in plan:
        if name in args.skip:
            print(f"\n=== SKIP {name} ===", flush=True)
            continue
        print(f"\n{'#'*70}\n#  {name.upper()}\n{'#'*70}", flush=True)

        train_cmd = [sys.executable, train_script,
                     "--mode", mode, "--seeds", *seeds_str, *epochs_arg]
        rc = _run(train_cmd)
        if rc != 0:
            print(f"!! training {name} failed (rc={rc}); continuing.",
                  flush=True)
            continue

        eval_cmd = [sys.executable, eval_script,
                    "--mode", mode, "--seeds", *seeds_str]
        _run(eval_cmd)

    # ---------------- Stage 7: ablation -----------------------------
    if "ablation" not in args.skip:
        print(f"\n{'#'*70}\n#  ABLATION (AC + BURGERS)\n{'#'*70}", flush=True)
        ablation_variants = ["xy_only", "t_only", "var_only", "full"]
        for problem in ["ac", "burgers"]:
            cmd = [sys.executable, "run_ablation.py",
                   "--problem", problem,
                   "--variants", *ablation_variants,
                   "--seeds", *ab_seeds_str,
                   "--epochs", str(args.ablation_epochs)]
            _run(cmd)
            eval_cmd = [sys.executable, "eval_ablation.py",
                        "--problem", problem,
                        "--variants", *ablation_variants,
                        "--seeds", *ab_seeds_str]
            _run(eval_cmd)

    # ---------------- Stage 8: error analysis -----------------------
    if "error_analysis" not in args.skip:
        print(f"\n{'#'*70}\n#  ERROR DISTRIBUTION ANALYSIS\n{'#'*70}",
              flush=True)
        for problem, mode in [
            ("ac",        "vanilla"),  ("ac",        "spectral"),
            ("heston",    "vanilla"),  ("heston",    "spectral"),
            ("burgers",   "vanilla"),  ("burgers",   "spectral"),
            ("rda",       "vanilla"),  ("rda",       "spectral"),
            ("fracdiff",  "vanilla"),  ("fracdiff",  "spectral"),
        ]:
            cmd = [sys.executable, "error_analysis.py",
                   "--problem", problem, "--mode", mode,
                   "--seeds", *seeds_str]
            _run(cmd)

    print(f"\n{'#'*70}\n#  PHASE-2 ALL DONE — total wallclock "
          f"{(time.time()-overall_t0)/3600:.2f} h\n{'#'*70}", flush=True)


if __name__ == "__main__":
    main()
