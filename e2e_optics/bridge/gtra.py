"""The GTRA lift -- the "middle part" that makes end-to-end LM possible.

This is the single most important idea in Cote et al. (2026): a transformation
that converts a *scalar* task loss into a *vector* of per-ray residuals with the
same value and gradient at the current iterate. That restores the residual
structure Levenberg-Marquardt needs (M = 2FWP >> N) while remaining faithful to
the downstream task objective.

Two functions, both pure (no state):

  tra_residuals(spot)                    -> conventional TRA residual  (Eq. 3)
  gtra_residuals(eps, L, grad_L, ...)    -> GTRA residual              (Eq. 6/S68)

GTRA generalizes TRA: it reduces to TRA exactly when w = 1/(FWP) and
eps' = centroid (see the paper, "Relation to conventional TRA").

--------------------------------------------------------------------------------
The lift (Eq. 5 / S66):

    L_TD(theta) ~= 1/2 * w * || eps(theta) - eps' ||^2
    w      = ||grad_L||^2 / (2 L)                         (scalar step size)
    eps'   = eps0 - 2 L grad_L / ||grad_L||^2             (per-ray target)
    l_GTRA = sqrt(w) * (eps(theta) - eps')                (Eq. 6 / S68)

Here:
  eps0   = current spot diagram (flat vector, R^{2FWP})
  L      = current scalar task loss (a Python float / 0-d tensor)
  grad_L = dL/d(eps), obtained by BACKWARD-mode AD through the *whole* pipeline
           (imaging sim + restoration + loss), computed ONCE per LM iteration.

The LM engine then needs J = d l_GTRA / d theta. Because w and eps' are held
CONSTANT during the LM solve, J = sqrt(w) * d eps / d theta -- the Jacobian of
the RAY TRACER ONLY, taken with FORWARD-mode AD. Forward-mode is never applied
through the expensive image sim / network. That asymmetry is the whole point.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch

from ..optics.base import SpotDiagram


def tra_residuals(spot: SpotDiagram,
                  weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Conventional transverse ray aberration residual (Eq. 3).

    l_TRA = (1/sqrt(FWP)) * (eps - eps_bar), where eps_bar is the per-field
    centroid broadcast over wavelength & pupil. ||l_TRA|| for one field is its
    effective spot radius. Returned flat, ordering (F, W, P, 2).
    """
    xy = spot.xy                                   # (F,W,P,2)
    F, W, P, _ = xy.shape
    cen = spot.centroids(weights)                  # (F,2)
    resid = xy - cen.reshape(F, 1, 1, 2)           # (F,W,P,2)
    resid = resid * spot.valid.unsqueeze(-1).to(xy.dtype)
    scale = 1.0 / (F * W * P) ** 0.5
    return (scale * resid).reshape(-1)


def gtra_weight_and_target(eps0: torch.Tensor, L: torch.Tensor,
                           grad_L: torch.Tensor,
                           eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the scalar weight w and per-ray target eps' from (eps0, L, grad_L).

    w    = ||grad_L||^2 / (2 L)
    eps' = eps0 - 2 L grad_L / ||grad_L||^2
    All inputs are detached (these are constants for the LM solve).
    """
    eps0 = eps0.detach().reshape(-1)
    grad_L = grad_L.detach().reshape(-1)
    L = torch.as_tensor(L).detach().reshape(())
    g2 = grad_L.dot(grad_L).clamp_min(eps)
    w = g2 / (2.0 * L.clamp_min(eps))
    eps_prime = eps0 - 2.0 * L * grad_L / g2
    return w, eps_prime


def clip_control_values(eps_prime: torch.Tensor, grid_half_extent: torch.Tensor,
                        eps0: torch.Tensor) -> torch.Tensor:
    """Bound control values to the virtual PSF grid (Supp. S3.1).

    If a target eps' would fall outside the finite PSF grid we clip it back onto
    the grid boundary along the ray's displacement direction, preserving the
    optimization direction. ``grid_half_extent`` is (2,) = (x_max, y_max) in mm
    (per ray-coordinate); ``eps0`` is the current landing point.

    v1 uses a simple per-coordinate clamp, which is the leading-order version of
    the paper's rescaling; the exact energy-preserving rescale is a documented
    refinement.
    """
    xy0 = eps0.reshape(-1, 2)
    xyp = eps_prime.reshape(-1, 2)
    lo = -grid_half_extent.reshape(1, 2)
    hi = grid_half_extent.reshape(1, 2)
    xyp_clipped = torch.maximum(torch.minimum(xyp, hi), lo)
    return xyp_clipped.reshape(-1)


def gtra_residuals(eps: torch.Tensor,
                   L: torch.Tensor,
                   grad_L: torch.Tensor,
                   eps0: Optional[torch.Tensor] = None,
                   grid_half_extent: Optional[torch.Tensor] = None,
                   return_wt: bool = False):
    """Generalized transverse ray aberration residual (Eq. 6 / S68).

    Parameters
    ----------
    eps : Tensor (2FWP,)
        The spot diagram as a function of theta -- this is the ONLY argument that
        carries gradient w.r.t. theta (forward-mode AD flows through here).
    L : scalar
        Current task loss value (constant).
    grad_L : Tensor (2FWP,)
        dL/d(eps) from backward-mode AD (constant).
    eps0 : Tensor (2FWP,), optional
        Spot diagram at the current iterate; defaults to eps.detach().
    grid_half_extent : Tensor (2,), optional
        Half-size of the PSF grid in mm for control-value clipping.

    Returns
    -------
    residual : Tensor (2FWP,)      = sqrt(w) * (eps - eps')
    (w, eps') : optional, if return_wt
    """
    if eps0 is None:
        eps0 = eps.detach()
    w, eps_prime = gtra_weight_and_target(eps0, L, grad_L)
    if grid_half_extent is not None:
        eps_prime = clip_control_values(eps_prime, grid_half_extent, eps0)
    residual = torch.sqrt(w) * (eps - eps_prime)
    if return_wt:
        return residual, (w, eps_prime)
    return residual

