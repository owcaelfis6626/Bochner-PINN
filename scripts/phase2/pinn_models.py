"""
PINN architectures for the additional 2D SPDE problems.

* HestonPINNSystem  : two networks (S, v).
* ACPINNSystem      : single network (u).

Both share the Fourier-feature embedding + residual MLP backbone used in
the main spde_2d package, parameterised through the corresponding config
classes.
"""

import math
import torch
import torch.nn as nn


class FourierFeatureEmbedding(nn.Module):
    def __init__(self, in_dim: int = 3, num_features: int = 48,
                 scale: float = 2.0):
        super().__init__()
        B = torch.randn(in_dim, num_features) * scale
        self.register_buffer("B", B)
        self.out_dim = 2 * num_features

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * v @ self.B
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.Tanh()

    def forward(self, x):
        h = self.act(x); h = self.fc1(h)
        h = self.act(h); h = self.fc2(h)
        return x + h


class _ScalarPINN(nn.Module):
    """Generic (x, y, t) -> scalar PINN."""

    def __init__(self, hidden_layers: int, hidden_dim: int,
                 num_fourier_features: int, fourier_scale: float):
        super().__init__()
        self.embed = FourierFeatureEmbedding(
            in_dim=3,
            num_features=num_fourier_features,
            scale=fourier_scale,
        )
        layers = [nn.Linear(self.embed.out_dim, hidden_dim)]
        for _ in range(hidden_layers):
            layers.append(ResidualBlock(hidden_dim))
        layers += [nn.Tanh(), nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, y, t):
        inp = torch.cat([x, y, t], dim=-1)
        return self.net(self.embed(inp))


# ─────────────────────────── Heston ───────────────────────────────────

class HestonPINNSystem(nn.Module):
    """Two coupled scalar PINNs:  S(x,y,t)  and  v(x,y,t)."""

    def __init__(self, cfg):
        super().__init__()
        self.net_S = _ScalarPINN(
            cfg.hidden_layers, cfg.hidden_dim,
            cfg.num_fourier_features, cfg.fourier_scale,
        )
        self.net_v = _ScalarPINN(
            cfg.hidden_layers, cfg.hidden_dim,
            cfg.num_fourier_features, cfg.fourier_scale,
        )

    def forward_all(self, x, y, t):
        return self.net_S(x, y, t), self.net_v(x, y, t)


# ─────────────────────────── Allen-Cahn ──────────────────────────────

class ACPINNSystem(nn.Module):
    """Single scalar PINN  u(x,y,t)  — wrapped in a system class so the
    trainer can use the same .forward_all interface as Heston."""

    def __init__(self, cfg):
        super().__init__()
        self.net_u = _ScalarPINN(
            cfg.hidden_layers, cfg.hidden_dim,
            cfg.num_fourier_features, cfg.fourier_scale,
        )

    def forward_all(self, x, y, t):
        return (self.net_u(x, y, t),)


# Aliases for problems that share the single-scalar architecture.
class BurgersPINNSystem(ACPINNSystem):
    """Same architecture as ACPINNSystem — single scalar field u."""
    pass


class RDAPINNSystem(ACPINNSystem):
    """Reaction-Advection PINN — single scalar field u."""
    pass


class FracDiffPINNSystem(ACPINNSystem):
    """Fractional Diffusion PINN — single scalar field u."""
    pass
