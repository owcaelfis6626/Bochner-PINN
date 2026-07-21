"""
Supervised training loop for the FNO operator.

Trains for `epochs` passes through the train split, evaluates on the
validation split each epoch, and saves the best (lowest val total loss)
checkpoint to `output_dir/fno_best.pt`.
"""

import os
import time
import json
import torch
from torch.utils.data import TensorDataset, DataLoader

from fno_model import FNO2d
from fno_losses import fno_loss
from fno_dataset import build_split, normalize_per_channel
from config_fno import FNOConfig


def train(fno_cfg: FNOConfig, device: torch.device, log_path: str = None):
    os.makedirs(fno_cfg.output_dir, exist_ok=True)

    # ── Data ────────────────────────────────────────────────────────
    X_tr, Y_tr = build_split(fno_cfg, fno_cfg.n_train,
                             fno_cfg.base_seed_train, "train")
    X_va, Y_va = build_split(fno_cfg, fno_cfg.n_val,
                             fno_cfg.base_seed_val, "val")

    # Normalise inputs (per-channel) using train stats
    X_tr_n, x_stats = normalize_per_channel(X_tr)
    X_va_n, _ = normalize_per_channel(X_va, x_stats)

    # Output stats kept too — useful at eval time, but we train on raw outputs
    _, y_stats = normalize_per_channel(Y_tr)

    # ── Pre-move entire dataset to GPU ──────────────────────────────────
    # X_tr+Y_tr ≈ 3.7 GB, val ≈ 463 MB — well within 32 GB VRAM.
    # Eliminates all CPU→GPU transfers in the hot loop, turning a CPU-bound
    # bottleneck into a pure GPU-bound training loop.
    print(f"  [data] moving datasets to {device}...", flush=True)
    X_tr_n = X_tr_n.to(device)
    Y_tr    = Y_tr.to(device)
    X_va_n  = X_va_n.to(device)
    Y_va    = Y_va.to(device)
    print("  [data] done.", flush=True)

    train_ds = TensorDataset(X_tr_n, Y_tr)
    val_ds   = TensorDataset(X_va_n, Y_va)
    # pin_memory=False: data is already on GPU, no CPU pinning needed.
    # num_workers=0: DataLoader just slices GPU tensors — no subprocess overhead.
    train_loader = DataLoader(train_ds, batch_size=fno_cfg.batch_size,
                              shuffle=True, num_workers=0, pin_memory=False,
                              drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=fno_cfg.batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)

    # ── Model ───────────────────────────────────────────────────────
    model = FNO2d(
        in_channels=fno_cfg.in_channels,
        out_channels=fno_cfg.out_channels,
        modes_x=fno_cfg.modes_x,
        modes_t=fno_cfg.modes_t,
        width=fno_cfg.width,
        n_layers=fno_cfg.n_layers,
        activation=fno_cfg.activation,
    ).to(device)
    print(f"  [model] params: {model.num_params():,}")

    # Save the un-compiled model for state_dict() (compiled wrapper has
    # extra prefixes that break checkpoint load).
    raw_model = model

    # torch.compile fuses ops into bigger CUDA graphs, eliminating most
    # of the Python/host overhead per step. Critical for small-batch
    # workloads on big GPUs (the 5090 was at <5% util without this).
    use_compile = bool(getattr(fno_cfg, "use_compile", True))
    if use_compile and device.type == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead",
                                  dynamic=False, fullgraph=False)
            print("  [model] torch.compile enabled (reduce-overhead)")
        except Exception as e:
            print(f"  [model] torch.compile failed ({e}); running eager")

    # bf16 mixed precision: 5090 (Blackwell) tensor cores love bf16,
    # halves memory bandwidth and roughly doubles throughput on the
    # spectral conv complex matmuls.
    use_amp = bool(getattr(fno_cfg, "use_amp", True))
    amp_dtype = torch.bfloat16
    print(f"  [train] AMP={'bf16' if use_amp else 'off'}  "
          f"compile={'on' if use_compile else 'off'}")

    opt = torch.optim.AdamW(model.parameters(),
                            lr=fno_cfg.lr,
                            weight_decay=fno_cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=fno_cfg.epochs, eta_min=fno_cfg.lr_min,
    )

    best_val = float("inf")
    history = []
    t0 = time.time()

    comp_keys = ("mse", "mean", "var", "spec_x", "spec_t")

    def _zero_accums():
        return (torch.zeros((), device=device),
                {k: torch.zeros((), device=device) for k in comp_keys})

    for ep in range(1, fno_cfg.epochs + 1):
        model.train()
        tr_total_t, tr_comp_t = _zero_accums()
        n_batches = 0

        for xb, yb in train_loader:
            # xb, yb already on GPU — no transfer needed
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    pred = model(xb)
                    loss, comp = fno_loss(
                        pred, yb,
                        w_mse=fno_cfg.w_mse,
                        w_spec_x=fno_cfg.w_spec_x,
                        w_spec_t=fno_cfg.w_spec_t,
                        w_mean=fno_cfg.w_mean,
                        w_var=fno_cfg.w_var,
                    )
            else:
                pred = model(xb)
                loss, comp = fno_loss(
                    pred, yb,
                    w_mse=fno_cfg.w_mse,
                    w_spec_x=fno_cfg.w_spec_x,
                    w_spec_t=fno_cfg.w_spec_t,
                    w_mean=fno_cfg.w_mean,
                    w_var=fno_cfg.w_var,
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), fno_cfg.grad_clip)
            opt.step()

            # Tensor accumulation: NO per-batch GPU->CPU sync.
            tr_total_t = tr_total_t + loss.detach()
            for k in comp_keys:
                tr_comp_t[k] = tr_comp_t[k] + comp[k]
            n_batches += 1

        sched.step()
        # One sync per epoch, not per batch.
        tr_total = (tr_total_t / max(n_batches, 1)).item()
        tr_components = {k: (tr_comp_t[k] / max(n_batches, 1)).item()
                         for k in comp_keys}

        # Validation
        model.eval()
        va_total_t, va_comp_t = _zero_accums()
        n_val_b = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                # xb, yb already on GPU
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        pred = model(xb)
                        loss, comp = fno_loss(
                            pred, yb,
                            w_mse=fno_cfg.w_mse,
                            w_spec_x=fno_cfg.w_spec_x,
                            w_spec_t=fno_cfg.w_spec_t,
                            w_mean=fno_cfg.w_mean,
                            w_var=fno_cfg.w_var,
                        )
                else:
                    pred = model(xb)
                    loss, comp = fno_loss(
                        pred, yb,
                        w_mse=fno_cfg.w_mse,
                        w_spec_x=fno_cfg.w_spec_x,
                        w_spec_t=fno_cfg.w_spec_t,
                        w_mean=fno_cfg.w_mean,
                        w_var=fno_cfg.w_var,
                    )
                va_total_t = va_total_t + loss.detach()
                for k in comp_keys:
                    va_comp_t[k] = va_comp_t[k] + comp[k]
                n_val_b += 1
        va_total = (va_total_t / max(n_val_b, 1)).item()
        va_components = {k: (va_comp_t[k] / max(n_val_b, 1)).item()
                         for k in comp_keys}

        elapsed = time.time() - t0
        history.append({
            "epoch": ep,
            "train_total": tr_total,
            "val_total": va_total,
            "train": tr_components,
            "val": va_components,
            "lr": opt.param_groups[0]["lr"],
            "elapsed": elapsed,
        })

        if ep % fno_cfg.log_every == 0 or ep == 1:
            print(f"  [ep {ep:4d}/{fno_cfg.epochs}]  "
                  f"train={tr_total:.4e}  val={va_total:.4e}  "
                  f"mse={va_components['mse']:.3e}  "
                  f"sx={va_components['spec_x']:.3e}  "
                  f"st={va_components['spec_t']:.3e}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}  "
                  f"t={elapsed/60:.1f}m")

        # Save best checkpoint (from raw_model, not the compiled wrapper —
        # OptimizedModule prefixes break load_state_dict on a fresh FNO2d).
        if va_total < best_val:
            best_val = va_total
            ckpt_path = os.path.join(fno_cfg.output_dir, "fno_best.pt")
            torch.save({
                "model": raw_model.state_dict(),
                "fno_cfg": fno_cfg.__dict__,
                "x_stats": {k: v.cpu() for k, v in x_stats.items()},
                "y_stats": {k: v.cpu() for k, v in y_stats.items()},
                "epoch": ep,
                "val_total": va_total,
            }, ckpt_path)

    if log_path:
        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)

    print(f"  [done] best val = {best_val:.4e}  total {(time.time()-t0)/60:.1f} min")
    return raw_model, x_stats, y_stats, history
