"""Bridge stage -- abstract interface.

The bridge is the "middle part": it turns a SpotDiagram into a simulated
capture I' that the algorithm stage can restore. Two responsibilities:

  1. Forward imaging model (differentiable end-to-end so backward-mode AD can
     reach dL/d(spot diagram)):  spot diagram -> PSF -> convolve scene -> noise.
  2. Host the GTRA lift (``bridge.gtra``), which converts the algorithm's
     scalar loss into per-ray optics residuals.

CONTRACT
--------
``simulate(spot, scene) -> I'`` where
  * ``scene``  : Tensor (C, H, W), intensity in [0, 1]
  * ``I'``     : Tensor (C, H, W), same convention
The PSF used internally must be energy-normalized (sums to 1 per channel).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import torch
from ..optics.base import SpotDiagram


class BaseBridge(ABC, torch.nn.Module):
    @abstractmethod
    def psf(self, spot: SpotDiagram) -> torch.Tensor:
        """Estimate the PSF from a spot diagram.

        Returns a tensor of shape (C, kH, kW), energy-normalized per channel.
        """
        ...

    @abstractmethod
    def simulate(self, spot: SpotDiagram, scene: torch.Tensor,
                 add_noise: bool = True) -> torch.Tensor:
        """Simulate the aberrated capture I' of ``scene`` through the optics."""
        ...
