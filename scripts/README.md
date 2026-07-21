# Paper 1 reproducibility scripts

Consolidated from `bochner_pinn/src/` and `01_paper/`.

```
scripts/
├── generate_polished_figures.py   # produces figures/ from raw experimental output
├── _make_paper_figures.py         # legacy figure-generation script from 01_paper/figures/
├── verification/                  # constants + empirical W2 calculations
│   ├── p1_constants.py            # computes C_1, C_2, C_3 constants in Prop 3.1
│   ├── p1_constants.npz
│   ├── p1_empirical_w2.py         # empirical W2 measurement script
│   └── p1_empirical_w2.npz
├── analysis/                      # post-training diagnostics
├── phase2/                        # main 2D experimental code (Heston-VG, hierarchical SPDE)
├── finance/                       # SPY calibration code (§9 of main.tex)
└── one_d/                         # 1D Bochner PINN experiments (space-weather K_p in §9)
```

## Source data not in this directory

The 3D Bochner PINN runs referenced by Figure 20 (`fig20_grid3d_conv.pdf`) live in
`/home/hubi/spde/spde-inne/pulled_from_runpod/spde-inne/{08_burgers_3d, 09_fpl_3d,
10_heston_vg_3d, 11_rda_3d}/results/` (multi-seed `.pt` model checkpoints +
`.log` training traces). These were trained on rented GPU and pulled back; they
should be moved to a clean data directory before submission.

The 2D experimental data referenced by Figures 2-18 was regenerated from the
phase2 code in this directory; raw outputs are in
`bochner_pinn/src/phase2/results_*` (and parallel copies in
`spectral_pinn_for_spde/02_rough_vol/results/bochner_*`).
