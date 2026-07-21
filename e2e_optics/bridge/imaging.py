"""Imaging simulation: spot diagram -> PSF -> blurred, noisy capture I'.

v1 uses a SHIFT-INVARIANT convolution over an image patch (one PSF per patch),
which the paper notes is a valid approximation over regions small enough that
the PSF is locally constant (sec. 4.2). Additive white Gaussian noise follows
Supp. S1.5 / sec. 4.2 (sigma = 0.5% of the max pixel value by default).

Extension hooks (documented, not implemented in v1):
  * PSF grid + spatially-varying overlap-add convolution with a Hann window
    (Supp. S1.5) -- for full-frame simulation with field-varying PSF.
  * Diffraction-compensated PSF (bridge.kde_psf.airy_field).
  * Spectral -> RGB integration via a sensor response matrix (multi-wavelength).
"""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F

from .base import BaseBridge
from .kde_psf import kde_psf
from ..optics.base import SpotDiagram


class ConvolutionImaging(BaseBridge):
    def __init__(self,
                 grid_size: int = 33,
                 pixel_pitch_mm: float = 0.005,
                 sigma_bins: float = 2.0,
                 noise_std_frac: float = 0.005,
                 field_index: int = 0,
                 wavelength_index: int = 0,
                 seed: Optional[int] = None):
        """
        grid_size       : PSF support (odd)
        pixel_pitch_mm  : PSF bin size = sensor pixel pitch (mm)
        noise_std_frac  : Gaussian noise std as a fraction of max pixel (0.005)
        field_index     : which field's PSF to use for a shift-invariant patch
        wavelength_index: which wavelength (monochromatic toy uses 0)
        """
        super().__init__()
        self.grid_size = int(grid_size)
        self.pixel_pitch_mm = float(pixel_pitch_mm)
        self.sigma_bins = float(sigma_bins)
        self.noise_std_frac = float(noise_std_frac)
        self.field_index = int(field_index)
        self.wavelength_index = int(wavelength_index)
        self._gen = None
        if seed is not None:
            self._gen = torch.Generator().manual_seed(seed)

    # --- BaseBridge -------------------------------------------------------
    def psf(self, spot: SpotDiagram) -> torch.Tensor:
        """Return one (1, G, G) shift-invariant PSF (single channel v1)."""
        p = kde_psf(spot, self.grid_size, self.pixel_pitch_mm,
                    self.sigma_bins, per_field=True)          # (F,W,G,G)
        k = p[self.field_index, self.wavelength_index]        # (G,G)
        return k.unsqueeze(0)                                 # (1,G,G)

    def convolve(self, scene: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:
        """Shift-invariant convolution. scene (C,H,W), psf (C,G,G) -> (C,H,W)."""
        C, H, W = scene.shape
        g = psf.shape[-1]
        pad = g // 2
        x = scene.unsqueeze(0)                                # (1,C,H,W)
        # depthwise convolution: one kernel per channel
        weight = psf.flip(-1, -2).unsqueeze(1)                # (C,1,G,G) correlation->conv
        x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        y = F.conv2d(x, weight, groups=C)
        return y.squeeze(0)

    def simulate(self, spot: SpotDiagram, scene: torch.Tensor,
                 add_noise: bool = True) -> torch.Tensor:
        C = scene.shape[0]
        k = self.psf(spot)                                    # (1,G,G)
        if C > 1:
            k = k.expand(C, -1, -1)
        blurred = self.convolve(scene, k)
        if add_noise and self.noise_std_frac > 0:
            std = self.noise_std_frac * scene.max().clamp_min(1e-6)
            if self._gen is not None:
                n = torch.randn(blurred.shape, generator=self._gen,
                                dtype=blurred.dtype) * std
            else:
                n = torch.randn_like(blurred) * std
            blurred = blurred + n
        return blurred.clamp(0.0, 1.0)

