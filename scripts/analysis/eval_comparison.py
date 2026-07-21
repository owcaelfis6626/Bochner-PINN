"""
Compare 2D FNO vs 16-seed PINN ensemble on held-out test seeds.
Evaluates N_TEST seeds per PINN model, computes ensemble stats, prints table.
"""
import sys, os, json, time
import numpy as np
import torch
from pathlib import Path

# Paths
SPDE_2D = Path(__file__).parent / "spde_2d"
sys.path.insert(0, str(SPDE_2D))

from config_pinn_2d import ConfigPINN2D
from models_2d import HierarchicalPINNSystem2D
from trainer_2d import _spatial_psd_xy, _temporal_psd, _build_realisation

# ── Config ──────────────────────────────────────────────────────────────────
FNO_METRICS  = Path(__file__).parent / "results_fno_2d" / "fno2d_metrics.json"
PINN_DIR     = Path(__file__).parent / "results_pinn_2d"
N_TEST       = 4          # test seeds per PINN model (fast; ±std stabilises at ~4)
TEST_SEEDS   = [2001, 2002, 2003, 2004]   # never overlap train (1-16) or val (50k+)
CHUNK        = 65536      # PINN forward-pass chunk size (CPU-safe)


# ── Spectral metrics (matching FNO eval) ────────────────────────────────────
def spectral_metrics(pred: torch.Tensor, true: torch.Tensor) -> dict:
    """pred/true: (Nx, Ny, Nt) float32."""
    def safe_log(x): return torch.log10(x.clamp(min=1e-12))

    sp_pred = _spatial_psd_xy(pred.unsqueeze(0))
    sp_true = _spatial_psd_xy(true.unsqueeze(0))
    sp_err  = (safe_log(sp_pred) - safe_log(sp_true)).abs().mean().item()

    tp_pred = _temporal_psd(pred.unsqueeze(0))
    tp_true = _temporal_psd(true.unsqueeze(0))
    tp_err  = (safe_log(tp_pred) - safe_log(tp_true)).abs().mean().item()

    var_ratio = (pred.var() / true.var().clamp(min=1e-12)).item()

    return {"sp_err": sp_err, "tp_err": tp_err, "var_ratio": var_ratio}


# ── Evaluate one PINN checkpoint on one test seed ───────────────────────────
def eval_pinn_seed(ckpt_path: Path, cfg: ConfigPINN2D,
                   test_seed: int, device: torch.device) -> dict:
    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)
    system = HierarchicalPINNSystem2D(cfg).to(device)
    system.load_state_dict(ckpt["model"])
    system.eval()

    _, (x_grid, y_grid, t_grid), (z_true, y_true, p_true) = \
        _build_realisation(cfg, test_seed, use_jumps=True, device=device)

    Nx, Ny, Nt = z_true.shape
    # Build flat (Nx*Ny*Nt, 1) coordinate tensors
    xi = torch.tensor(x_grid, dtype=torch.float32, device=device)
    yi = torch.tensor(y_grid, dtype=torch.float32, device=device)
    ti = torch.tensor(t_grid, dtype=torch.float32, device=device)
    XX, YY, TT = torch.meshgrid(xi, yi, ti, indexing="ij")   # (Nx,Ny,Nt) each
    xf = XX.reshape(-1, 1); yf = YY.reshape(-1, 1); tf = TT.reshape(-1, 1)

    # Chunked forward (avoids OOM on CPU)
    z_chunks, y_chunks, p_chunks = [], [], []
    with torch.no_grad():
        for i in range(0, xf.size(0), CHUNK):
            zp, yp, pp = system.forward_all(xf[i:i+CHUNK],
                                            yf[i:i+CHUNK],
                                            tf[i:i+CHUNK])
            z_chunks.append(zp); y_chunks.append(yp); p_chunks.append(pp)

    z_pred = torch.cat(z_chunks).reshape(Nx, Ny, Nt)
    y_pred = torch.cat(y_chunks).reshape(Nx, Ny, Nt)
    p_pred = torch.cat(p_chunks).reshape(Nx, Ny, Nt)

    return {
        "z": spectral_metrics(z_pred, z_true),
        "y": spectral_metrics(y_pred, y_true),
        "p": spectral_metrics(p_pred, p_true),
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    cfg    = ConfigPINN2D()
    device = torch.device("cpu")   # local, no GPU
    print(f"Evaluating PINN ensemble on {N_TEST} test seeds: {TEST_SEEDS}")
    print(f"Grid: {cfg.Nx}×{cfg.Ny}×{cfg.Nt}  Chunk: {CHUNK}\n")

    ckpt_files = sorted(PINN_DIR.glob("pinn2d_seed*.pt"),
                        key=lambda p: int(p.stem.replace("pinn2d_seed", "")))

    all_metrics = {ch: {"sp_err": [], "tp_err": [], "var_ratio": []}
                   for ch in ("z", "y", "p")}

    for ci, ckpt in enumerate(ckpt_files):
        seed_id = int(ckpt.stem.replace("pinn2d_seed", ""))
        print(f"  [{ci+1:2d}/16] seed {seed_id:2d}:", end="", flush=True)
        t0 = time.time()
        seed_avg = {ch: {"sp_err": [], "tp_err": [], "var_ratio": []}
                    for ch in ("z", "y", "p")}
        for ts in TEST_SEEDS:
            m = eval_pinn_seed(ckpt, cfg, ts, device)
            for ch in ("z", "y", "p"):
                for k in ("sp_err", "tp_err", "var_ratio"):
                    seed_avg[ch][k].append(m[ch][k])
        for ch in ("z", "y", "p"):
            for k in ("sp_err", "tp_err", "var_ratio"):
                val = float(np.mean(seed_avg[ch][k]))
                all_metrics[ch][k].append(val)
        print(f"  z_sp={np.mean(seed_avg['z']['sp_err']):.3e}  "
              f"y_sp={np.mean(seed_avg['y']['sp_err']):.3e}  "
              f"p_sp={np.mean(seed_avg['p']['sp_err']):.3e}  "
              f"[{time.time()-t0:.0f}s]")

    # ── Ensemble summary ────────────────────────────────────────────────────
    ens = {}
    for ch in ("z", "y", "p"):
        ens[ch] = {
            "sp_err_mean":    float(np.mean(all_metrics[ch]["sp_err"])),
            "sp_err_std":     float(np.std(all_metrics[ch]["sp_err"])),
            "tp_err_mean":    float(np.mean(all_metrics[ch]["tp_err"])),
            "tp_err_std":     float(np.std(all_metrics[ch]["tp_err"])),
            "var_ratio_mean": float(np.mean(all_metrics[ch]["var_ratio"])),
            "var_ratio_std":  float(np.std(all_metrics[ch]["var_ratio"])),
        }

    # Save
    out = PINN_DIR / "pinn_ensemble_eval.json"
    with open(out, "w") as f:
        json.dump({"pinn_ensemble": ens, "n_pinn_seeds": len(ckpt_files),
                   "n_test_seeds": N_TEST, "test_seeds": TEST_SEEDS}, f, indent=2)
    print(f"\nSaved -> {out}\n")

    # ── Comparison table ─────────────────────────────────────────────────────
    with open(FNO_METRICS) as f:
        fno = json.load(f)

    hdr = f"{'':6s}  {'sp_err':>22s}  {'tp_err':>22s}  {'var_ratio':>18s}"
    sep = "─" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)
    for ch in ("z", "y", "p"):
        e  = ens[ch]
        fo = fno[ch]
        pinn_sp  = f"{e['sp_err_mean']:.3e} ± {e['sp_err_std']:.1e}"
        pinn_tp  = f"{e['tp_err_mean']:.3e} ± {e['tp_err_std']:.1e}"
        pinn_vr  = f"{e['var_ratio_mean']:.3f} ± {e['var_ratio_std']:.3f}"
        fno_sp   = f"{fo['sp_err']:.3e}"
        fno_tp   = f"{fo['tp_err']:.3e}"
        fno_vr   = f"{fo['var_ratio']:.3f}"
        print(f"\n[{ch}]")
        print(f"  FNO  (ep47)   sp={fno_sp:>10s}   tp={fno_tp:>10s}   var={fno_vr}")
        print(f"  PINN (n=16)   sp={pinn_sp:>22s}   tp={pinn_tp:>22s}   var={pinn_vr}")
    print(sep)


if __name__ == "__main__":
    main()
