"""Differentiable geometric PSF via kernel density estimation (Supp. S1.4.2).

Binning rays into a histogram is non-differentiable (a ray's bin membership
jumps discontinuously as parameters move). Instead each ray deposits its energy
with a small SEPARABLE TRIANGULAR kernel of support ``sigma`` bins. Two
properties matter:

  * differentiable: the deposited weights are piecewise-linear in the ray's
    sub-pixel position, so d(PSF)/d(ray position) exists everywhere;
  * energy-conserving: each ray contributes the same TOTAL energy regardless of
    its sub-bin offset (the triangular weights over the touched bins always sum
    to 1), so the PSF integrates to the ray count -> normalize to 1.

This is exactly the estimator the paper validates in Supp. S1.4.2.

Diffraction compensation (Supp. S1.4.3 / Eq. 12) is provided as a documented
stub ``airy_field`` -- v1 uses the geometric PSF only.
"""
from __future__ import annotations
from typing import Optional
import math

import torch

from ..optics.base import SpotDiagram


def _triangular_deposit(coord_bins: torch.Tensor, n_bins: int,
                        sigma: float) -> torch.Tensor:
    """1-D separable triangular splat weights.

    coord_bins : (...,) ray positions in *bin units* (0 = grid center bin edge).
    Returns weights (..., n_bins) summing to 1 for a ray whose kernel support
    lies wholly INSIDE the grid, and to the on-grid *fraction* otherwise.

    A triangular kernel of half-width ``sigma`` centered at each ray, evaluated
    at integer bin centers. The normalizer is the sum over the kernel's FULL
    support -- i.e. what the weights would sum to on an unbounded grid -- not the
    sum over the bins that happen to be on-grid.

    That distinction is the whole point. Normalizing by the on-grid sum forces
    every ray to deposit unit energy even when it lands outside the frame, so a
    ray 1-2 bins past the edge dumps its full energy into the border bin. On the
    starting design of the toy (98-99% of rays off a 25x25 / 11.3 um grid) that
    put 25-67% of the "PSF" energy in a one-bin frame around the border -- a pure
    artifact that looked like diffraction rings. Normalizing by the full-support
    sum instead lets off-grid energy genuinely leave the grid; ``kde_psf`` then
    renormalizes the whole PSF to unit energy, spreading the deficit over the
    real PSF shape rather than the frame (Supp. S1.4 note).

    Energy per ray is still exactly conserved for any ray whose support is
    interior, which is the property the paper's KDE relies on.
    """
    centers = torch.arange(n_bins, dtype=coord_bins.dtype, device=coord_bins.device)
    d = (coord_bins.unsqueeze(-1) - centers.reshape(*([1] * coord_bins.dim()), n_bins)).abs()
    w = torch.clamp(1.0 - d / sigma, min=0.0)              # triangular, on-grid

    # full-support normalizer: same kernel summed over the integer bin centers
    # around the ray, ignoring the grid bounds. Exact for any sigma (for integer
    # sigma it equals sigma identically; for e.g. sigma=2.5 it oscillates 2.4-2.6,
    # so it must be computed rather than assumed constant).
    K = int(math.ceil(sigma))
    frac = coord_bins - torch.floor(coord_bins)                      # in [0,1)
    off = torch.arange(-K, K + 1, dtype=coord_bins.dtype, device=coord_bins.device)
    dn = (frac.unsqueeze(-1) - off.reshape(*([1] * frac.dim()), off.numel())).abs()
    norm = torch.clamp(1.0 - dn / sigma, min=0.0).sum(-1, keepdim=True)

    return w / norm.clamp_min(1e-12)


def kde_psf(spot: SpotDiagram,
            grid_size: int = 33,
            pixel_pitch_mm: float = 0.005,
            sigma_bins: float = 2.0,
            per_field: bool = True) -> torch.Tensor:
    """Estimate geometric PSF(s) from a spot diagram by triangular-kernel KDE.

    Parameters
    ----------
    spot : SpotDiagram
    grid_size : int
        PSF kernel is grid_size x grid_size bins (odd -> centered).
    pixel_pitch_mm : float
        Physical size of one PSF bin (mm). Rays farther than
        grid_size/2 * pitch from the field centroid fall off-grid (their lost
        energy is redistributed by renormalization -- Supp. S1.4 note).
    sigma_bins : float
        Triangular kernel half-support in bins (paper uses 2).
    per_field : bool
        If True, return one PSF per (field, wavelength): shape (F, W, G, G).
        If False, collapse fields -> (W, G, G).

    Returns
    -------
    psf : Tensor
        Energy-normalized (sums to 1 over the GxG grid) per returned PSF.
    """
    xy = spot.xy                                            # (F,W,P,2) mm
    F, W, P, _ = xy.shape
    dtype = xy.dtype
    cen = spot.centroids().reshape(F, 1, 1, 2)              # center each PSF on centroid
    rel = (xy - cen) / pixel_pitch_mm                       # bins, centered at 0
    half = (grid_size - 1) / 2.0
    coord = rel + half                                      # 0..grid_size-1
    valid = spot.valid.to(dtype)                            # (F,W,P)

    wx = _triangular_deposit(coord[..., 0], grid_size, sigma_bins)   # (F,W,P,G)
    wy = _triangular_deposit(coord[..., 1], grid_size, sigma_bins)   # (F,W,P,G)
    # outer product per ray -> (F,W,P,G,G); weight by validity; sum over rays P
    w = valid.unsqueeze(-1).unsqueeze(-1) * (wy.unsqueeze(-1) * wx.unsqueeze(-2))
    psf = w.sum(dim=2)                                      # (F,W,G,G)
    psf = psf / psf.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-12)
    if not per_field:
        psf = psf.mean(dim=0)                               # (W,G,G)
    return psf


def airy_field(grid_size: int, pixel_pitch_mm: float, wavelength_um: float,
               na: float, dtype=torch.float64) -> torch.Tensor:
    """Airy amplitude pattern U_Airy(r) ∝ 2 J1(k NA r)/(k NA r)  (Eq. 12/S25).

    STUB / extension hook (Supp. S1.4.3). Diffraction compensation convolves the
    geometric-PSF amplitude (sqrt of intensity, flat phase) with this Airy field
    and squares the result, preventing the optimizer from chasing physically
    impossible sub-diffraction-limited spots. First dark ring at r1≈0.61 λ/NA;
    the kernel should span >= 6 r1. Not used in the v1 all-geometric slice.
    """
    raise NotImplementedError(
        "Diffraction-compensated PSF is a v1 extension hook (Supp. S1.4.3)."
    )



def offgrid_fraction(spot: SpotDiagram, grid_size: int,
                     pixel_pitch_mm: float) -> torch.Tensor:
    """Fraction of valid rays per field landing outside the PSF grid: (F,).

    A geometric PSF is only meaningful when the grid actually contains the spot.
    The KDE renormalizes each PSF to unit energy, so a grid that is too small
    still returns a plausible-looking image -- it is just the *clipped* spot,
    with the truncated energy redistributed over whatever remained on-grid.

    Rule of thumb: keep this below a few percent. On the toy's STARTING design
    (spot radius ~1.5 mm) a 25x25 grid at 11.3 um pitch (half-width 136 um)
    clips 99% of the rays; the same grid is appropriate once the design has
    converged to a ~20 um spot. Size the grid to the design, or report this
    number alongside the PSF.
    """
    cen = spot.centroids().reshape(-1, 1, 1, 2)
    rel = (spot.xy - cen).abs() / pixel_pitch_mm
    off = (rel > (grid_size - 1) / 2.0).any(-1) & spot.valid          # (F,W,P)
    n = spot.valid.sum(dim=(1, 2)).clamp_min(1)
    return off.sum(dim=(1, 2)).to(spot.xy.dtype) / n
