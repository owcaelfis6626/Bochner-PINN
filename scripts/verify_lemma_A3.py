"""Numerical verification of Lemma A3 (notes/zolotarev_extension_v1.md).

Claim: the law-level driver-smoothing error of the stochastic convolution
vanishes as the spline grid h -> 0, with rate

  (E ||u* - u_bar||^s)^{1/s}  ~  h^{1/2}    (VG, s = 2)
  (E ||u* - u_bar||^s)^{1/s}  ~  h^{1/alpha} (symmetric alpha-stable, s < alpha)

where, per mode with rate lambda:
  u*(t)    = int_0^t exp(-lambda (t-s)) dL(s)        (true driver)
  u_bar(t) = int_0^t exp(-lambda (t-s)) Ltilde'(s) ds (linear-spline driver)
and Ltilde interpolates the path L on the coarse grid of width h.

This is the pathwise-coupled quantity that feeds Lemma A1; the experiment
verifies that the Theta(1) "jump floor" of the residual formulation is
absent at the solution-law level.

Single mode, T = 1, fine grid 2^16. CPU, float64, no training.
"""

import numpy as np
from scipy.signal import lfilter

RNG = np.random.default_rng(0)

T       = 1.0
N_F     = 2 ** 16          # fine grid
DT      = T / N_F
LAMBDAS = [4.0, 25.0]
MS      = [16, 64, 256, 1024, 4096]   # coarse cell = m fine steps, h = m*DT

# VG (theta=0 symmetric, sigma=1, nu=0.5): m2 = sigma^2 = 1
VG_SIGMA, VG_NU = 1.0, 0.5
M2 = VG_SIGMA ** 2
N_REP_VG = 256

# alpha-stable
ALPHA   = 1.5
S_MOM   = 1.2              # s < alpha for the moment metric
N_REP_AS = 1024

BATCH = 64


def vg_increments(n_rep, n_f, rng):
    """Symmetric VG increments on the fine grid. (n_rep, n_f)."""
    dG = rng.gamma(shape=DT / VG_NU, scale=VG_NU, size=(n_rep, n_f))
    return VG_SIGMA * np.sqrt(dG) * rng.standard_normal((n_rep, n_f))


def stable_increments(n_rep, n_f, rng):
    """Symmetric alpha-stable increments, scale DT^{1/alpha}. CMS algorithm."""
    V = rng.uniform(-np.pi / 2, np.pi / 2, size=(n_rep, n_f))
    W = rng.exponential(1.0, size=(n_rep, n_f))
    X = (np.sin(ALPHA * V) / np.cos(V) ** (1.0 / ALPHA)
         * (np.cos((1.0 - ALPHA) * V) / W) ** ((1.0 - ALPHA) / ALPHA))
    return DT ** (1.0 / ALPHA) * X


def conv_gap_sq(dL, lam, m):
    """||u* - u_bar||_{L2(0,T)}^2 per replication.

    dL : (n_rep, N_F) fine-grid driver increments
    m  : coarse cell width in fine steps (h = m*DT)
    Returns (n_rep,)
    """
    a = np.exp(-lam * DT)

    # u*: recursion u_{j+1} = a u_j + dL_j  (jump at right endpoint)
    u_star = lfilter([1.0], [1.0, -a], dL, axis=1)   # (n_rep, N_F)

    # spline driver: cell increments / h, constant density per coarse cell
    n_c = N_F // m
    cell_inc = dL.reshape(dL.shape[0], n_c, m).sum(axis=2)   # (n_rep, n_c)
    dens = np.repeat(cell_inc / (m * DT), m, axis=1)          # (n_rep, N_F)
    u_bar = lfilter([1.0], [1.0, -a], dens * DT, axis=1)

    return ((u_star - u_bar) ** 2).sum(axis=1) * DT          # (n_rep,)


def run_class(name, gen, n_rep, metrics):
    print(f"\n=== {name} ===")
    header = "lambda      h     " + "  ".join(f"{k:>12s}" for k in metrics)
    print(header)
    results = {lam: {k: [] for k in metrics} for lam in LAMBDAS}
    for lam in LAMBDAS:
        for m in MS:
            h = m * DT
            gaps = []
            for start in range(0, n_rep, BATCH):
                nb = min(BATCH, n_rep - start)
                dL = gen(nb, N_F, RNG)
                gaps.append(conv_gap_sq(dL, lam, m))
            g2 = np.concatenate(gaps)           # ||.||^2 per rep
            gn = np.sqrt(g2)                    # ||.|| per rep
            vals = {}
            if "rms" in metrics:
                vals["rms"] = float(np.sqrt(g2.mean()))
            if "ratio_m2Th" in metrics:
                vals["ratio_m2Th"] = float(g2.mean() / (M2 * T * h))
            if "median" in metrics:
                vals["median"] = float(np.median(gn))
            if "Ls_mean" in metrics:
                vals["Ls_mean"] = float((gn ** S_MOM).mean() ** (1.0 / S_MOM))
            for k in metrics:
                results[lam][k].append(vals[k])
            print(f"{lam:6.1f} {h:9.2e} " +
                  "  ".join(f"{vals[k]:12.4e}" for k in metrics))

    print("\n--- log-log slopes vs h ---")
    hs = np.array([m * DT for m in MS])
    for lam in LAMBDAS:
        for k in metrics:
            if k == "ratio_m2Th":
                continue
            y = np.array(results[lam][k])
            # fit only the resolved regime lambda*h <= 1
            mask = lam * hs <= 1.0
            slope = np.polyfit(np.log(hs[mask]), np.log(y[mask]), 1)[0]
            print(f"  lambda={lam:5.1f}  {k:>8s}: slope = {slope:+.3f}")
    return results


def main():
    print(f"Lemma A3 verification  T={T}  N_fine=2^16  "
          f"VG(sigma={VG_SIGMA}, nu={VG_NU})  alpha={ALPHA}, s={S_MOM}")
    print(f"Predictions: VG rms ~ h^0.5 ; "
          f"stable median & L^s mean ~ h^(1/alpha) = h^{1.0/ALPHA:.3f}")

    run_class("Variance-Gamma (s=2)", vg_increments, N_REP_VG,
              ["rms", "ratio_m2Th"])
    run_class(f"symmetric {ALPHA}-stable", stable_increments, N_REP_AS,
              ["median", "Ls_mean"])


if __name__ == "__main__":
    main()
