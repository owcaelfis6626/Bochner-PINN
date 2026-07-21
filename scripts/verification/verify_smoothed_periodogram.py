"""
Verification of Lemma S1.4 (single-realisation kernel-smoothed
periodogram consistency) for the rewritten Paper 1.

Approach: at each grid size N, sweep the smoothing bandwidth h and
report (i) the raw-periodogram MSE (h=0), (ii) the MSE-optimal smoothed
MSE, and (iii) the bandwidth at which the optimum occurs. The lemma
predicts:
  - raw periodogram MSE does not vanish with N (variance per mode is
    Theta(mu(k)^2), independent of N);
  - smoothed-periodogram MSE at the MSE-optimal bandwidth decays
    polynomially in N at rate N^{-2 s_x / (2 s_x + d)};
  - the optimal bandwidth grows with N.

Spectrum mu(k) = (1+k^2)^{-s} on the integer lattice. Each realisation
samples the periodogram from its known chi^2_2 distribution at each
mode. ~1 minute CPU runtime.
"""
import numpy as np

s = 1.0
d = 1
N_values = [64, 128, 256, 512, 1024, 2048]
N_real = 2048
h_grid = list(range(0, 65))   # h=0 is the raw periodogram

print(f"Lemma S1.4 verification (s={s}, d={d}, {N_real} realisations per N)")
print(f"{'N':>6}  {'raw MSE':>10}  {'opt MSE':>10}  {'h_opt':>6}  "
      f"{'opt/raw':>8}  {'theory rate hint':>18}")

theory_exp = -s / (2*s + d)
results = {}

for N in N_values:
    k = np.fft.fftfreq(N, d=1.0/N)
    mu = (1.0 + np.abs(k)**2) ** (-s)
    rng = np.random.default_rng(0)
    # Sample N_real periodograms: at each mode, draw chi^2_2 * mu(k)/2
    z2 = rng.standard_normal((N_real, N))**2 + rng.standard_normal((N_real, N))**2
    raw = z2 * mu / 2   # E[raw] = mu

    best_mse = float("inf"); best_h = 0
    raw_mse = float(np.mean((raw - mu)**2))
    for h in h_grid:
        if 2*h + 1 > N // 2:    # kernel must be smaller than half the signal
            continue
        if h == 0:
            mse = raw_mse
        else:
            kernel = np.ones(2*h + 1) / (2*h + 1)
            sm = np.empty_like(raw)
            for r_i in range(raw.shape[0]):
                sm[r_i] = np.convolve(raw[r_i], kernel, mode="same")
            mse = float(np.mean((sm - mu)**2))
        if mse < best_mse:
            best_mse = mse; best_h = h
    rel_improve = best_mse / raw_mse
    results[N] = (raw_mse, best_mse, best_h)
    print(f"{N:>6}  {raw_mse:>10.5f}  {best_mse:>10.5f}  {best_h:>6d}  "
          f"{rel_improve:>8.3f}  {'N^'+f'{theory_exp:.3f}':>18}")

print()
print("Decay of MSE-optimal smoothed estimator (theory: N^{-2 s/(2s+d)})")
print(f"{'N_lo->N_hi':>14}  {'raw decay':>10}  {'smooth decay':>14}  {'theory':>10}")
N_list = list(results.keys())
for i in range(len(N_list)-1):
    N_lo, N_hi = N_list[i], N_list[i+1]
    raw_decay = results[N_hi][0] / results[N_lo][0]
    smooth_decay = results[N_hi][1] / results[N_lo][1]
    theory = (N_hi / N_lo) ** (-2 * s / (2*s + d))
    print(f"  {N_lo:>4d}->{N_hi:<5d}  {raw_decay:>10.3f}  {smooth_decay:>14.3f}  {theory:>10.3f}")

print()
total_raw_decay = results[N_list[-1]][0] / results[N_list[0]][0]
total_smooth_decay = results[N_list[-1]][1] / results[N_list[0]][1]
theory_total = (N_list[-1] / N_list[0]) ** (-2 * s / (2*s + d))
print(f"Total decay N={N_list[0]} -> N={N_list[-1]}:")
print(f"  raw MSE decay:       {total_raw_decay:.3f}  (expected close to 1)")
print(f"  smoothed MSE decay:  {total_smooth_decay:.3f}  (theory {theory_total:.3f})")
print(f"  smoothed/theory:     {total_smooth_decay / theory_total:.3f}")
verdict = "PASS" if (total_raw_decay > 0.5 and total_smooth_decay < 0.5 * total_raw_decay) else "FAIL"
print(f"  Lemma S1.4 verification: {verdict}")
