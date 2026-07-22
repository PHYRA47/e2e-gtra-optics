"""A small U-Net image restoration model (IRM).

This is a lightweight stand-in for the paper's NAFNet-based restorer (Supp.
S1.6): a 2-level encoder/decoder with skip connections, residual output
(network predicts the correction to the degraded image). It is deliberately
small so the toy end-to-end demo trains in seconds on CPU. Swap it for any
``BaseRestoration`` -- NAFNet, Restormer, a Wiener+U-Net sandwich, etc.

The IRM is a ``torch.nn.Module`` and its parameters are optimized by Adam in the
joint loop's algorithm step, while the lens parameters are frozen; the lens step
does the reverse (Supp. S3.3.1).

Wiener stub: ``WienerDeconv`` documents the learnable-SNR Wiener front-end that
the paper sandwiches between two NAFNets; provided as an extension hook.
"""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
from .base import BaseRestoration


def _conv(ci, co):
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, padding=1), nn.GELU(),
        nn.Conv2d(co, co, 3, padding=1), nn.GELU(),
    )


class TinyUNet(BaseRestoration, nn.Module):
    """2-level residual U-Net. conditioned=False (blind restoration)."""
    conditioned = False

    def __init__(self, channels: int = 1, width: int = 16):
        nn.Module.__init__(self)
        BaseRestoration.__init__(self)
        self.enc1 = _conv(channels, width)
        self.down = nn.MaxPool2d(2)
        self.enc2 = _conv(width, width * 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = _conv(width * 2 + width, width)
        self.head = nn.Conv2d(width, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        e1 = self.enc1(x)
        e2 = self.enc2(self.down(e1))
        d1 = self.dec1(torch.cat([self.up(e2), e1], dim=1))
        return x + self.head(d1)                        # residual restoration

    def restore(self, degraded: torch.Tensor,
                psf: Optional[torch.Tensor] = None) -> torch.Tensor:
        single = degraded.dim() == 3
        x = degraded.unsqueeze(0) if single else degraded
        y = self.forward(x).clamp(0.0, 1.0)
        return y.squeeze(0) if single else y


class WienerDeconv(nn.Module):
    """Learnable-SNR Wiener deconvolution front-end (Supp. S1.6). STUB.

    Y = F^{-1}[ conj(H) / (|H|^2 + 1/SNR) * F(X) ], with log-SNR a learnable
    scalar (init log(1000)). The paper sandwiches this between two NAFNets.
    Provided as a documented extension hook; not used in the v1 toy.
    """
    def __init__(self, snr_init: float = 1000.0):
        super().__init__()
        self.log_snr = nn.Parameter(torch.log(torch.tensor(float(snr_init))))

    def forward(self, x, psf):  # pragma: no cover - extension hook
        raise NotImplementedError(
            "Wiener deconvolution is a v1 extension hook (Supp. S1.6)."
        )

