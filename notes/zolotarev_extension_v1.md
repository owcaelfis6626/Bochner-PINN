# Full-law convergence for non-Gaussian (Lévy) fields via Zolotarev metrics

**Status: v1 proof program — 2026-06-09.**
Replaces Prop. `levy-ito` ($4\sqrt J$ floor) and resolves the α-stable case left open in §3 (`subsec:levy-ext`).
References on hand: Zolotarev 1976 (`for claude/sm2908_eng.pdf`), Bołbotowski–Bouchitté 2025 (`for claude/2511.00232v1.pdf`).

---

## 0. The three obstacles (precise restatement)

The current paper proves (Prop. 3.1) for Gaussian fields:
$$W_2(\mathcal L(u_\theta), \mathcal L(u_{\rm FD})) \le C_1\sqrt\varepsilon + C_2 N_s^{-\beta_{\rm sr}} + C_3\sqrt\delta.$$

For Lévy drivers it stalls at:

* **(O1) Gaussian-projection gap.** PSD + variance losses see only second-order
  structure. Prop. `levy-ito` pays an additive $4\sqrt J$
  ($J$ = jump $L^2$ energy) that training cannot reduce.
* **(O2) Spline-smoothing jump floor.** The pathwise Mishra–Molinaro leg
  evaluates the residual against a spline-interpolated driver $\tilde L$. For
  unbounded-variation drivers the *pathwise residual* of the true solution
  against $\tilde L$ is $\Theta(1)$ in $L^2$, uncontrolled by grid refinement
  (Remark after Lemma S1.6).
* **(O3) α-stable failure.** For $\alpha < 2$: $J = \infty$ (A7 fails), second
  moments are infinite, $W_2 = \infty$, and the spectral measure does not exist
  classically. The current technique gives nothing.

**Root diagnosis.** (O2) is a *strong-norm artifact*: a spline cannot track a
jump pathwise, but in law the smoothing is a small perturbation (Lemma A3
below). (O3) is a *metric mismatch*: the right metric must be finite under
$s$-moments only, $s < \alpha$, and ideal (compatible with convolution
structure). Both fixes point to the Zolotarev metric $\zeta_s$.

---

## 1. Continuous problem and conventions

**SPDE.** On $D = \mathbb T^d$ (periodised, consistent with (A1)), $t \in [0,T]$:
$$\partial_t u = \mathcal A u + \sigma\,\dot\xi + \dot L, \qquad u(\cdot,0) = u_0 \ \text{(deterministic)},$$
$\mathcal A$ = second-order elliptic generator (heat-type), semigroup $S_t$,
per-mode symbol $-\lambda_k$, $\lambda_k \asymp |k|^2$.

**Bandlimited drivers (standing assumption, = sub-grid honesty of (A3)).**
$\dot\xi$, $\dot L$ are supported on Fourier modes $|k| \le K$, $J_K := \#\{k\}$.
Per mode $k$: $L_k$ is a one-dimensional Lévy process; across modes independent
(or jointly Lévy with cross-independence after diagonalisation — same proofs).

**Driver classes.**
* **(VG):** Variance-Gamma — infinite activity, finite variation, all moments
  finite. Jump second moment $m_2 := \int z^2\,\nu_{\rm VG}(dz) < \infty$.
* **(αS):** symmetric α-stable, $\alpha \in (1,2)$. $\mathbb E|L_k(t)|^p < \infty$
  iff $p < \alpha$. Itô (compensated) convention for the jump integral.

**Mild solution.**
$$u(t) = S_t u_0 + \sigma\!\int_0^t\! S_{t-s}\, dW_s + \int_0^t\! S_{t-s}\, dL_s
\;\in\; H := L^2(D \times [0,T]).$$
Per mode: $\hat u_k(t) = e^{-\lambda_k t}\hat u_{0,k} + \int_0^t e^{-\lambda_k(t-s)}dL_k(s) + (\text{Gaussian part})$.

**The coupling structure (this is what makes the theorem possible).**
The experiments share one driver realisation: $u_{\rm FD} = \Psi_N(\xi, L)$
(numerical solver), $u_\theta = \Phi_\theta(\tilde\xi, \tilde L)$ where
$\tilde L = \mathcal S_h L$ is the spline smoothing of the *same* increments at
temporal resolution $h$. Hence $(u_\theta, u_{\rm FD})$ live on one probability
space with a canonical coupling. All bounds below exploit this; the
*ensemble/no-coupling* setting is deferred to §7 (open leg).

**Intermediate objects.**
* $\bar u :=$ exact mild solution driven by $(\tilde\xi, \tilde L)$
  (smooth-in-time driver, pathwise classical).
* $u^\star :=$ exact mild solution driven by the true $(\xi, L)$.

The error chain: $u_\theta \leftrightarrow \bar u$ (residual, Lemma A2)
$\leftrightarrow u^\star$ (driver smoothing in law, Lemma A3)
$\leftrightarrow u_{\rm FD}$ (numerical error, cited).

---

## 2. The Zolotarev metric on $H$

For $s \in (1,2]$, write $s = 1 + \beta$, $\beta \in (0,1]$. Define
$$\Lambda_s := \{f : H \to \mathbb R \ \text{Fréchet-}C^1,\
\|f'(x) - f'(y)\|_{H^*} \le \|x - y\|^{\beta}\ \forall x,y\},$$
$$\zeta_s(\mu, \nu) := \sup_{f \in \Lambda_s}\Big|\int f\,d\mu - \int f\,d\nu\Big|.$$

Properties (Zolotarev 1976, valid on separable Banach spaces):
* **(Z1) Ideality of order $s$:** $\zeta_s(X + Z, Y + Z) \le \zeta_s(X, Y)$ for
  $Z \perp (X,Y)$, and $\zeta_s(cX, cY) = c^s \zeta_s(X,Y)$.
* **(Z2) Finiteness:** $\zeta_s(\mu,\nu) < \infty$ requires *equal means* (in
  the Pettis sense — $\Lambda_s$ contains all linear functionals) and finite
  $s$-th absolute moments.
* **(Z3) Convergence:** $\zeta_s$ metrises weak convergence + $s$-moment
  convergence on mean-matched classes.

**Mean handling.** $\mathbb E u_\theta \ne \mathbb E u_{\rm FD}$ exactly, so we
state results for the *centred* laws plus an explicit mean term:
$$\rho_s(X, Y) := \zeta_s\big(\mathcal L(X - \mathbb EX),\ \mathcal L(Y - \mathbb EY)\big) \;+\; \|\mathbb EX - \mathbb EY\|_H.$$
$\rho_s$ metrises the same topology and every bound below controls both pieces.

**Transfer back to Wasserstein (why $\zeta_s$ is not a compromise).**
* **(T1)** [Bołbotowski–Bouchitté 2025, Thm. 3.1] On $\mathcal P_2(\mathbb R^d)$,
  any $d$: $\tfrac14 W_2^2 \le \zeta_2$, constant sharp, dimension-free.
* **(T2)** [Rio 2009] On $\mathbb R$, $s \in (1,2]$: $W_s^s \le c_s\,\zeta_s$.

---

## 3. Main results

### Lemma A1 (Taylor-coupling bound for $\zeta_s$)

*Let $X, Y \in L^s(\mathbb P; H)$ with $\mathbb EX = \mathbb EY$. Then*
$$\zeta_s(\mathcal L X, \mathcal L Y)
\;\le\; \big(\mathbb E\|Y\|^s\big)^{\frac{s-1}{s}}\big(\mathbb E\|X - Y\|^s\big)^{\frac1s}
\;+\; \mathbb E\|X - Y\|^s.$$

**Proof.** Fix $f \in \Lambda_s$. Subtract the affine part: $g(x) := f(x) - f(0) - \langle f'(0), x\rangle$.
Then $g \in \Lambda_s$, $g'(0) = 0$, and since $\mathbb EX = \mathbb EY$ the affine part cancels:
$\mathbb E f(X) - \mathbb E f(Y) = \mathbb E g(X) - \mathbb E g(Y)$.
From $g'(0)=0$ and the Hölder bound, $\|g'(x)\| \le \|x\|^{\beta}$ ($\beta = s-1$).
By the fundamental theorem of calculus along the segment $[Y, X]$:
$$|g(X) - g(Y)| \le \int_0^1 \|g'(Y + t(X-Y))\|\,dt\,\|X-Y\|
\le \int_0^1 (\|Y\| + t\|X-Y\|)^{\beta} dt\,\|X-Y\|.$$
Since $\beta \le 1$, $(a+b)^\beta \le a^\beta + b^\beta$, so the integrand is
$\le \|Y\|^\beta + \|X-Y\|^\beta$, giving
$$|g(X)-g(Y)| \le \|Y\|^{s-1}\|X-Y\| + \|X-Y\|^s.$$
Take expectations; Hölder with exponents $\tfrac{s}{s-1}, s$ on the first term. $\blacksquare$

*Remark.* No independence, no Gaussianity, no structure beyond moments: the
canonical coupling does all the work. For $s=2$ this reads
$\zeta_2 \le \sigma_Y\,\|X-Y\|_{L^2(\mathbb P;H)} + \|X-Y\|^2_{L^2(\mathbb P;H)}$.

### Lemma A2 (residual stability in $L^s$, replaces MM-in-$L^2$)

*Linear case, rigorous.* $u_\theta - \bar u$ solves the deterministic parabolic
equation with source $r_\theta := \partial_t u_\theta - \mathcal A u_\theta - \sigma\tilde\xi - \dot{\tilde L}$
(both objects see the same smoothed driver, which is pathwise a function).
Parabolic stability pathwise: $\|u_\theta - \bar u\|_H \le C_{\rm stab}\,\big(\|r_\theta\|_{H'} + \|u_\theta(0) - u_0\|\big)$.
Raising to the $s$-th power and taking expectations:
$$\mathbb E\|u_\theta - \bar u\|_H^s \;\le\; C_{\rm stab}^s\,\delta_s,
\qquad \delta_s := \mathbb E\big(\|r_\theta\|_{H'} + \|u_\theta(0)-u_0\|\big)^s.$$
$\delta_s$ is exactly a trainable residual loss with exponent $s$ — see §6.

*Semilinear case (Burgers-VG), conditional.* Same statement under the
well-posedness hypothesis of MM Thm. 3.4 (stability constant from Grönwall on
a ball containing both trajectories; for Burgers this needs an a priori
$L^\infty$ bound on $\bar u$, which holds pathwise for the smoothed driver).
Stated as assumption (B2'), matching the paper's existing conditionality.

### Lemma A3 (driver smoothing vanishes in law — the new key lemma)

*Setting.* $u^\star - \bar u = \int_0^\cdot S_{\cdot - s}\,d(L - \tilde L)(s)$
(the Gaussian part is treated identically and is easier). Per mode $k$, with
$g_k(\tau) = e^{-\lambda_k \tau}\mathbf 1_{\tau > 0}$ and the spline kernel
$K_h$ (order $\ge 1$, mass 1, support $\le c h$):
$$\hat u^\star_k - \hat{\bar u}_k = \int_0^T R_k(t, s)\, dL_k(s), \qquad
R_k(t,s) := g_k(t-s) - (g_k * K_h)(t-s).$$

*Kernel estimate.* Splitting $g_k$ into the jump at $\tau = 0$ and the smooth
tail:
$$\|R_k(t, \cdot)\|_{L^2(0,T)}^2 \;\le\; \underbrace{c_0\, h}_{\text{jump at } 0}
\;+\; \underbrace{c_1\, h^2 \|g_k'\|_{L^2}^2}_{\text{smooth part}}
\;\le\; c_0 h + c_1 h^2 \lambda_k/2
\;\le\; C^2 h \quad \text{for } h \le \lambda_K^{-1}.$$
Similarly $\|R_k(t,\cdot)\|_{L^\alpha(0,T)}^\alpha \le C^\alpha h$ for
$\alpha \in (1,2)$ (the jump contributes $h \cdot O(1)$, the smooth part
$h^\alpha\|g_k'\|_{L^\alpha}^\alpha$, both $O(h)$ in the bandlimited regime).

**(VG):** Itô isometry for compensated Poisson integrals (Peszat–Zabczyk):
$$\mathbb E\|u^\star - \bar u\|_H^2
= \sum_{|k| \le K} \int_0^T\!\! \int_0^T |R_k(t,s)|^2\, m_2\, ds\, dt
\;\le\; m_2\, T\, J_K\, C^2\, h.$$
Hence
$$\big(\mathbb E\|u^\star - \bar u\|_H^s\big)^{1/s} \le \big(\mathbb E\|u^\star - \bar u\|_H^2\big)^{1/2} \le C\sqrt{m_2 T\, J_K\, h\,}.$$

**(αS):** The field $Z := u^\star - \bar u$ is jointly SαS in $H$
(linear functional of the stable random measure). For an $H$-valued SαS
element given by $Z = \int \mathcal R(s)\,dL(s)$ with deterministic
kernel sections $\mathcal R(s) := R(\cdot, s) \in L^2_t$, the scale
parameter is
$$\sigma_Z^\alpha \;=\; \sum_{|k|\le K}\int_0^T \big\|R_k(\cdot, s)\big\|_{L^2_t}^{\alpha}\, ds$$
*(note the composition: $H$-norm inside, $L^\alpha(ds)$ outside — the
random-measure variable is $s$)*, and by the LePage series representation
and the moment bounds for stable norms (Samorodnitsky–Taqqu Ch. 3;
Ledoux–Talagrand Ch. 5), $\big(\mathbb E\|Z\|_H^s\big)^{1/s} \le c_{\alpha,s}\,\sigma_Z$
for $s < \alpha$. The kernel section estimate (jump region $t \in (s, s+ch)$
contributes $h \cdot O(1)$; smooth tail contributes $h^2\lambda_k$) gives
$\|R_k(\cdot,s)\|_{L^2_t} \le C\sqrt h$ for $h \le \lambda_K^{-1}$, hence
$$\big(\mathbb E\|Z\|_H^s\big)^{1/s} \;\le\; c_{\alpha,s}\,
\big(T\, J_K\big)^{1/\alpha}\, C\,\sqrt h.$$

**Conclusion (both classes).** The law-level smoothing error vanishes at the
*same* temporal rate $h^{1/2}$, with class-dependent mode-count prefactor:
$$\big(\mathbb E\|u^\star - \bar u\|^s\big)^{1/s} \;\le\;
\begin{cases}
C_{\rm VG}\,J_K^{1/2}\, h^{1/2} & \text{(VG)},\\[2pt]
C_{\alpha}\,J_K^{1/\alpha}\, h^{1/2} & \text{(αS)}.
\end{cases}$$

> **Correction history.** v1 of this note claimed $h^{1/\alpha}$ for αS via
> the per-time-slice scale $\|R_k(t,\cdot)\|_{L^\alpha(ds)}$ — a wrong norm
> composition (it computes the marginal scale of $Z(t)$, not the scale of the
> $H$-valued element). Numerical verification (§5b) caught it: the robust
> median estimator scales as $h^{0.49}$, not $h^{0.67}$. The corrected rate
> is *slower* than the v1 claim for $\alpha < 2$; the theorem statement
> below uses the verified $h^{1/2}$.

*This is the resolution of (O2): the $\Theta(1)$ pathwise jump floor is an
artifact of measuring the **residual** against the smoothed driver; the
**laws** of the two mild solutions converge as $h \to 0$. The old
$4\sqrt J$ floor becomes $C\sqrt{J_K h}$ — trainable to zero by grid
refinement.*

### Lemma A4 (cylindrical lift of Bołbotowski–Bouchitté to $H$)

*For centred $\mu, \nu \in \mathcal P_2(H)$, $H$ separable Hilbert:*
$$W_2^2(\mu, \nu) \;\le\; 4\,\zeta_2(\mu, \nu).$$

**Proof.** Let $\pi_m : H \to \mathbb R^m$ be the projection onto the first $m$
coordinates of a fixed ONB. (i) For $f \in \Lambda_2(\mathbb R^m)$,
$f \circ \pi_m \in \Lambda_2(H)$: $(f\circ\pi_m)'(x) = \pi_m^* f'(\pi_m x)$ and
$\|\pi_m^*(f'(\pi_m x) - f'(\pi_m y))\| \le \|\pi_m x - \pi_m y\| \le \|x-y\|$.
Hence $\zeta_2(\pi_m\#\mu, \pi_m\#\nu) \le \zeta_2(\mu,\nu)$.
(ii) By BB25 Thm. 3.1 in $\mathbb R^m$ (constant dimension-free):
$W_2^2(\pi_m\#\mu, \pi_m\#\nu) \le 4\zeta_2(\pi_m\#\mu, \pi_m\#\nu) \le 4\zeta_2(\mu,\nu)$.
(iii) $W_2(\pi_m\#\mu, \pi_m\#\nu) \uparrow W_2(\mu,\nu)$ as $m \to \infty$
(standard: optimal couplings of projections are tight; any weak limit couples
$\mu,\nu$; monotone convergence on the cost). $\blacksquare$

### Theorem A (full-law convergence for Lévy-driven Bochner PINNs)

*Assume (A1)–(A4) of the paper, bandlimited drivers as in §1, the shared-driver
coupling (B1), residual stability (B2/B2'), and let:*
* *(VG):* $s = 2$;
* *(αS):* $\alpha \in (1,2)$, $s \in (1, \alpha)$.

*Set $\Delta_s := \big(\mathbb E\|u_\theta - u_{\rm FD}\|_H^s\big)^{1/s}$. Then*
$$\Delta_s \;\le\; \underbrace{C_{\rm stab}\,\delta_s^{1/s}}_{\text{training residual}}
\;+\; \underbrace{C_{\rm sm}\,J_K^{\gamma}\,h^{1/2}}_{\text{driver smoothing, } \gamma = \frac12 \text{ (VG)},\ \frac1\alpha \text{ (αS)}}
\;+\; \underbrace{\varepsilon_{\rm num}}_{\text{solver error, cited}}$$
*and consequently*
$$\boxed{\;\rho_s\big(u_\theta,\, u_{\rm FD}\big) \;\le\;
M_s^{\,s-1}\,\Delta_s \;+\; \Delta_s^{\,s} \;+\; \Delta_s,\qquad
M_s := \big(\mathbb E\|u_{\rm FD}\|_H^s\big)^{1/s}.\;}$$

*All constants explicit: $C_{\rm stab}$ from parabolic stability, $C_{\rm sm}$
from Lemma A3 (depends on $m_2, T$ or $\alpha$), $M_s$ bounded via the
Green-function and driver moments (LePage for αS). No Gaussianity. No
$4\sqrt J$. Finite for α-stable.*

**Proof.** $L^s(\mathbb P; H)$-norm triangle inequality on the chain
$u_\theta \to \bar u \to u^\star \to u_{\rm FD}$ (Lemmas A2, A3, citation), then
Lemma A1 with $X = u_\theta - \mathbb E u_\theta$, $Y = u_{\rm FD} - \mathbb E u_{\rm FD}$
(centring changes $\Delta_s$ by at most a factor 2 via
$\|\mathbb E(X-Y)\| \le \Delta_s$, absorbed in constants), plus the mean term
$\|\mathbb E u_\theta - \mathbb E u_{\rm FD}\| \le \Delta_s$. $\blacksquare$

### Corollaries

* **(C1, VG — Wasserstein recovered, floor removed.)** By Lemma A4:
  $$W_2(\mathcal L(u_\theta), \mathcal L(u_{\rm FD}))
  \;\le\; 2\sqrt{\zeta_2 + \text{mean}^2}
  \;\lesssim\; 2\, M_2^{1/2}\, \Delta_2^{1/2} \quad (\Delta_2 \text{ small}).$$
  Comparison with Prop. `levy-ito`: the old bound had a *non-vanishing*
  $4\sqrt J$; the new bound vanishes as $(\delta_2^{1/2} + \sqrt{J_K h})^{1/2}$
  — a *soft-rate* ($\Delta^{1/2}$) but *floor-free* certificate. Both can be
  stated; the min is the operative bound.
* **(C2, αS — first finite bound.)** $\rho_s$ finite and $\to 0$ for
  $s < \alpha$; on pointwise marginals $u(x,t) \in \mathbb R$, Rio (T2) gives
  $W_s(\mathcal L(u_\theta(x,t)), \mathcal L(u_{\rm FD}(x,t))) \le (c_s\,\zeta_s^{\rm marg})^{1/s}$,
  with $\zeta_s^{\rm marg} \le \zeta_s$ by the projection argument of Lemma A4(i)
  applied to the evaluation functional (after mollification — evaluation is not
  bounded on $H$; use local averages $\langle u, \varphi_{x,t}\rangle$, which is
  what the experiments measure anyway).
* **(C3, Gaussian consistency.)** As jump intensity $\to 0$ (VG $\nu \to 0$),
  Theorem A's bound converges to a $\zeta_2$ version of Prop. 3.1's pathwise
  leg; the spectral leg (Prop. 3.1) is unchanged and complementary.

---

## 4. Rate table (operating point: $d=2$, bandlimited $K$, time grid $h = 1/N_t$)

| driver | metric | training term | smoothing term | floor |
|---|---|---|---|---|
| Gaussian (old Prop. 3.1) | $W_2$ | $C_1\sqrt\varepsilon + C_3\sqrt\delta$ | — | none |
| VG (old Prop. `levy-ito`) | $W_2$ | $C_1\sqrt\varepsilon + C_3\sqrt\delta$ | — | $4\sqrt J$ ❌ |
| VG (**new**, Thm. A + C1) | $W_2$ | $\lesssim M_2^{1/2}(C_{\rm stab}\delta_2^{1/2})^{1/2}$ | $\lesssim M_2^{1/2}(J_K h)^{1/4}$ | **none** ✅ |
| VG (**new**, Thm. A) | $\zeta_2$ | $M_2\, C_{\rm stab}\,\delta_2^{1/2}$ | $M_2\,C_{\rm sm}(J_K h)^{1/2}$ | **none** ✅ |
| αS, $s<\alpha$ (**new**) | $\zeta_s$ | $M_s^{s-1} C_{\rm stab}\,\delta_s^{1/s}$ | $M_s^{s-1} C_\alpha J_K^{1/\alpha} h^{1/2}$ | **none** ✅ |

Notes:
* The $J_K$-dependence persists but now *multiplies a vanishing factor* —
  the honest replacement of the $4\sqrt J$ floor.
* $h \le \lambda_K^{-1}$ required (resolve the fastest mode); same regime as
  the FD solver's stability condition, so no new constraint in practice.

---

## 5b. Numerical verification of Lemma A3 (2026-06-09)

`scripts/verify_lemma_A3.py` — single mode, exact stochastic convolution vs
linear-spline-smoothed driver, $T = 1$, fine grid $2^{16}$,
$\lambda \in \{4, 25\}$, $h \in [2.4\cdot 10^{-4}, 6.3\cdot 10^{-2}]$,
fit restricted to the resolved regime $\lambda h \le 1$.

| class | estimator | predicted slope | measured ($\lambda{=}4$) | measured ($\lambda{=}25$) |
|---|---|---|---|---|
| VG ($\sigma{=}1,\nu{=}0.5$, 256 reps) | $(\mathbb E\|Z\|^2)^{1/2}$ | $+0.5$ | $+0.492$ | $+0.504$ |
| αS ($\alpha{=}1.5$, 1024 reps) | median $\|Z\|$ | $+0.5$ (corrected) | $+0.495$ | $+0.483$ |
| αS | $(\mathbb E\|Z\|^{1.2})^{1/1.2}$ | $+0.5$ | $+0.655$* | $+0.508$ |

\* The $L^{1.2}$-moment estimator has infinite variance at $\alpha = 1.5$
(needs $2s < \alpha$): the $+0.655$ fit is dominated by a single outlier
replication at the coarsest $h$. The median is the trustworthy estimator
and agrees with the corrected $h^{1/2}$ at both $\lambda$.

**Constant check (VG).** The ratio $\mathbb E\|Z\|^2 / (m_2 T h)$ is flat
in $h$ at $\approx 0.165 \approx 1/6$ — consistent with the
linear-interpolation $L^2$ constant; conjecture
$C_{\rm VG}^2 = m_2 T/6$ sharp for linear splines (derivation: pleasant
exercise, not needed for the theorem).

**Verification verdict.** Lemma A3 (VG) confirmed including the constant.
Lemma A3 (αS) as stated in v1 was *falsified* in the exponent and has been
corrected above ($h^{1/\alpha} \to h^{1/2}$); the corrected statement is
confirmed by the median estimator at both $\lambda$.

---

## 5. Council critique log (resolved before writing)

* **Mathematician.** (i) $\zeta_s = \infty$ without exact mean matching since
  $\Lambda_s$ contains all linear functionals → resolved by the centred metric
  $\rho_s$ with explicit mean term. (ii) Stable $s$-moments in $H$: cannot
  Fubini $\mathbb E(\int|Z|^2)^{s/2}$ directly at $s<2$ → resolved via LePage
  series (Lemma A3, αS case). (iii) Evaluation functionals unbounded on $H$ →
  marginal corollary stated for mollified evaluations only.
* **Numerical Analyst.** (i) Heat-kernel time singularity makes
  $\|g_k'\|_{L^2}$ blow up as $\lambda_k \to \infty$; bandlimiting caps it —
  the assumption is *essential*, scope honestly matches (A3). (ii) The kernel
  jump at $\tau = 0$ limits the smoothing rate to $h^{1/2}$ (VG) regardless of
  spline order — don't promise $h^1$. (iii) $\delta_s^{1/s}$ for $s<2$ is a
  *better* exponent than $\sqrt\delta$ at small $\delta$, but $\delta_s$ itself
  is a weaker norm — no free lunch, the table states both factors.
* **ML Engineer.** (i) Every quantity in Theorem A is measurable at training
  time. (ii) **Practical prescription that falls out of the proof:** for
  α-stable drivers, train the PDE residual in $L^s$, $s < \alpha$ — the MSE
  residual has infinite variance under stable forcing and SGD on it is
  ill-posed. The metric choice dictates the loss exponent. This should be a
  highlighted remark in the paper. (iii) Theorem A alone does not explain
  variance-collapse prevention (that is the spectral leg's job); A and the
  spectral leg remain complementary, as in the Gaussian proof.

---

## 6. What this changes in the paper

1. **Replace** Prop. `levy-ito` by Theorem A + Corollary C1; move the
  $4\sqrt J$ version to a remark ("strong-coupling bound; superseded in law").
2. **Delete** the "α-stable out of reach" sentence in `subsec:levy-ext`;
  replace by Corollary C2 ($s < \alpha$).
3. **Rewrite** the Remark after Lemma S1.6: the spline floor is pathwise-only;
  in law it vanishes at rate $(J_K h)^\gamma$ (Lemma A3).
4. **Add** the loss-exponent prescription (Council/ML-Engineer item ii) to the
  discussion: $L^s$ residual training for heavy tails.
5. New supplementary section: proofs of A1–A4 (A1, A3, A4 complete above;
  A2 linear complete, semilinear conditional as before).

## 7. Open leg (deferred): distribution-only certificate

Without the shared-driver coupling (ensemble setting), second-order losses
cannot pin the law (phase/polyspectra freedom — information-theoretic, not
technical). The natural closure: a characteristic-functional probe loss
$\mathcal L_{\rm cf} = \sup_{\varphi \in \mathcal P}|\phi_{u_\theta}(\varphi) - \phi_{u_{\rm FD}}(\varphi)|$
on a finite probe family $\mathcal P \subset H$, transferred to $\zeta_s$ by an
Esseen-type smoothing inequality. The 1-d prototype closes (Esseen smoothing
with weight $|\xi|^{-s-1}$ on a window $|\xi| \le \Lambda$, tail controlled by
$s$-moments); the *probe-family design* in $H$ — which $\varphi$, what weight,
what $\Lambda(N)$ — is the genuine research question. Candidate: probes =
the bandlimited Fourier modes themselves with stable-scale weights. Deferred
to v2.

---

*v1.1 (2026-06-09). A1, A3, A4 proven in full; A2 rigorous (linear) /
conditional (semilinear), matching the paper's existing honesty level.
Lemma A3 numerically verified (§5b): VG confirmed with sharp-looking
constant $m_2T/6$; αS exponent corrected $h^{1/\alpha} \to h^{1/2}$ after
the verification falsified the v1 norm-composition. Next: LaTeX-ify into
supplementary S2; main-text edits per §6.*
