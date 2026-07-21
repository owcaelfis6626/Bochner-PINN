"""
2D Fourier Neural Operator for the SPDE hierarchy.

Reference: Li et al., "Fourier Neural Operator for Parametric Partial
Differential Equations", ICLR 2021.

A SpectralConv2d layer:
    1. take rfft2 of the input feature map (B, C, Nx, Nt)
    2. truncate to (modes_x, modes_t) low-frequency coefficients
    3. apply a learned per-mode complex linear map  C -> C
    4. zero-pad the rest and irfft2 back to (B, C, Nx, Nt)

The full FNO stacks several SpectralConv2d layers, each followed by a
1x1 conv (the "W" branch) and a non-linearity, plus a lifting layer at
the front and a projection MLP at the end.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _act(name: str):
    return {"relu": F.relu, "gelu": F.gelu, "silu": F.silu}[name.lower()]


class SpectralConv2d(nn.Module):
    """2D spectral convolution: truncated FFT -> per-mode linear -> iFFT."""

    def __init__(self, in_channels: int, out_channels: int,
                 modes_x: int, modes_t: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x   # along x (axis -2)
        self.modes_t = modes_t   # along t (axis -1)  -- note rfft halves this axis

        # Two complex weight blocks (positive and negative x-frequencies).
        # Stored as real tensors with last dim 2 (real, imag) for autograd.
        scale = 1.0 / (in_channels * out_channels)
        self.w1 = nn.Parameter(scale * torch.randn(
            in_channels, out_channels, modes_x, modes_t, 2))
        self.w2 = nn.Parameter(scale * torch.randn(
            in_channels, out_channels, modes_x, modes_t, 2))

    @staticmethod
    def _cmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Complex multiply via real-valued batched matmul (cuBLAS SGEMM).

        The original torch.einsum on complex tensors bypassed cuBLAS and
        fell into a ~100x slower CPU/fallback path. Four real matmuls with
        the @ operator route through cuBLAS SGEMM and actually use the GPU.

        a : (B, Cin, X, T)  complex
        b : (Cin, Cout, X, T, 2)  real  (last dim = [re, im])
        returns (B, Cout, X, T)  complex
        """
        ar = a.real   # (B, Cin, X, T)
        ai = a.imag
        br = b[..., 0]   # (Cin, Cout, X, T)
        bi = b[..., 1]

        B, Cin, X, T = ar.shape
        Cout = br.shape[1]

        # Collapse (X, T) into one batch dim → 3-D tensors for cuBLAS.
        # (X, T, B, Cin)  →  (X*T, B, Cin)
        ar3 = ar.permute(2, 3, 0, 1).reshape(X * T, B, Cin)
        ai3 = ai.permute(2, 3, 0, 1).reshape(X * T, B, Cin)
        # (X, T, Cin, Cout)  →  (X*T, Cin, Cout)
        br3 = br.permute(2, 3, 0, 1).reshape(X * T, Cin, Cout)
        bi3 = bi.permute(2, 3, 0, 1).reshape(X * T, Cin, Cout)

        # Batched SGEMM: (XT, B, Cin) @ (XT, Cin, Cout) → (XT, B, Cout)
        # (a_r + i·a_i)(b_r + i·b_i) = (a_r·b_r - a_i·b_i) + i(a_r·b_i + a_i·b_r)
        out_r = ar3 @ br3 - ai3 @ bi3   # (XT, B, Cout)
        out_i = ar3 @ bi3 + ai3 @ br3

        # Unpack back to (B, Cout, X, T)
        out_r = out_r.reshape(X, T, B, Cout).permute(2, 3, 0, 1)
        out_i = out_i.reshape(X, T, B, Cout).permute(2, 3, 0, 1)
        return torch.complex(out_r, out_i)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, Nx, Nt)
        B, C, Nx, Nt = x.shape
        # rfft2 over (Nx, Nt) -> (B, C, Nx, Nt//2 + 1) complex
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))

        out_ft = torch.zeros(
            B, self.out_channels, Nx, Nt // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )

        mx = min(self.modes_x, Nx // 2)
        mt = min(self.modes_t, Nt // 2 + 1)

        # Low x-frequencies (positive)
        out_ft[:, :, :mx, :mt] = self._cmul(
            x_ft[:, :, :mx, :mt], self.w1[:, :, :mx, :mt]
        )
        # High x-frequencies (negative, wrapped)
        out_ft[:, :, -mx:, :mt] = self._cmul(
            x_ft[:, :, -mx:, :mt], self.w2[:, :, :mx, :mt]
        )

        x = torch.fft.irfft2(out_ft, s=(Nx, Nt), dim=(-2, -1))
        return x


class FNOBlock(nn.Module):
    """One layer:  y = act( SpectralConv(x) + W(x) )   (residual implicit)."""

    def __init__(self, width: int, modes_x: int, modes_t: int, activation: str):
        super().__init__()
        self.spec = SpectralConv2d(width, width, modes_x, modes_t)
        self.w = nn.Conv2d(width, width, kernel_size=1)
        self.act = _act(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.spec(x) + self.w(x))


class FNO2d(nn.Module):
    """Full 2D FNO: lifting -> N FNOBlocks -> projection."""

    def __init__(self, in_channels: int, out_channels: int,
                 modes_x: int, modes_t: int, width: int,
                 n_layers: int, activation: str = "gelu"):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList([
            FNOBlock(width, modes_x, modes_t, activation)
            for _ in range(n_layers)
        ])
        self.proj1 = nn.Conv2d(width, 128, kernel_size=1)
        self.proj2 = nn.Conv2d(128, out_channels, kernel_size=1)
        self.act = _act(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, Nx, Nt)
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.act(self.proj1(x))
        x = self.proj2(x)
        return x

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
