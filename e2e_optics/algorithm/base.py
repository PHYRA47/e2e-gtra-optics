"""Algorithm stage -- abstract interface for image restoration models (IRMs).

CONTRACT
--------
``restore(I', psf=None) -> I''`` where
  * ``I'``   : degraded capture, Tensor (C, H, W) or (B, C, H, W), in [0, 1]
  * ``psf``  : optional conditioning PSF (C, kH, kW); conditioned IRMs use it
  * ``I''``  : restored image, same shape/convention as I'

Image-driven design (Cote et al. 2026, sec. 4) is the special case where the
IRM is the identity -- the optics alone must produce a good image. That is the
Fig. 4 toy problem, provided by ``IdentityRestoration``.

The IRM is optimized with Adam (backward-mode AD), independently of the lens,
in the alternating joint loop.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import torch


class BaseRestoration(ABC, torch.nn.Module):
    #: whether ``restore`` consumes the ``psf`` argument (conditioned IRM)
    conditioned: bool = False

    @abstractmethod
    def restore(self, degraded: torch.Tensor,
                psf: Optional[torch.Tensor] = None) -> torch.Tensor:
        ...

    def forward(self, degraded: torch.Tensor,
                psf: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.restore(degraded, psf)
