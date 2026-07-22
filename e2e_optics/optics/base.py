"""Optics stage -- abstract interface and the SpotDiagram container.

CONTRACT
--------
An optics model maps its internal parameter vector ``theta`` to a
``SpotDiagram``: the (x, y) landing coordinates of a bundle of traced rays at
the image plane, organized by field / wavelength / pupil position.

The spot diagram ``epsilon`` is the intermediate representation on which the
GTRA lift operates (Cote et al. 2026, Fig. 3). Everything downstream (PSF,
imaging, loss) depends on theta *only through* the spot diagram -- this is the
assumption that makes GTRA valid.

Units & conventions (stated once, honored everywhere):
  * lengths in millimetres (mm)
  * optical axis is +z; image plane is the last surface
  * fields are specified as angles (degrees) of the chief ray in object space
  * ``SpotDiagram.xy`` has shape (n_field, n_wavelength, n_pupil, 2), last dim (x, y)
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class SpotDiagram:
    """Ray landing coordinates at the image plane.

    Attributes
    ----------
    xy : Tensor, shape (F, W, P, 2)
        (x, y) image-plane coordinates in mm, for F fields, W wavelengths,
        P pupil samples.
    valid : Tensor(bool), shape (F, W, P)
        False where a ray failed (TIR, missed surface, non-converged march).
        Failed rays must be excluded from centroid / RMS / PSF computations.
    fields_deg : Tensor, shape (F,)
        Field angles in degrees (metadata, for plotting and PSF-grid placement).
    wavelengths_um : Tensor, shape (W,)
        Wavelengths in micrometres (metadata).
    """
    xy: torch.Tensor
    valid: torch.Tensor
    fields_deg: Optional[torch.Tensor] = None
    wavelengths_um: Optional[torch.Tensor] = None

    @property
    def shape(self):
        return self.xy.shape

    @property
    def n_field(self):  return self.xy.shape[0]
    @property
    def n_wave(self):   return self.xy.shape[1]
    @property
    def n_pupil(self):  return self.xy.shape[2]

    def flat(self) -> torch.Tensor:
        """Flatten to the ``epsilon in R^{2 F W P}`` vector used by GTRA/TRA.

        Ordering is (F, W, P, 2) row-major -> matches ``gtra_residuals``.
        """
        return self.xy.reshape(-1)

    def centroids(self, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Per-field spot centroid (mean over wavelength & pupil of valid rays).

        Returns shape (F, 2). This is ``epsilon_bar`` in the TRA definition (Eq. 3).
        """
        xy = self.xy
        m = self.valid.unsqueeze(-1).to(xy.dtype)                # (F,W,P,1)
        if weights is not None:
            m = m * weights.reshape(1, -1, 1, 1)
        num = (xy * m).sum(dim=(1, 2))                           # (F,2)
        den = m.sum(dim=(1, 2)).clamp_min(1e-12)
        return num / den

    def rms_radius(self, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Field-wise RMS spot radius in mm (Eq. S23). Returns shape (F,)."""
        c = self.centroids(weights).unsqueeze(1).unsqueeze(1)    # (F,1,1,2)
        d2 = ((self.xy - c) ** 2).sum(-1)                        # (F,W,P)
        m = self.valid.to(self.xy.dtype)
        if weights is not None:
            m = m * weights.reshape(1, -1, 1)
        msr = (d2 * m).sum(dim=(1, 2)) / m.sum(dim=(1, 2)).clamp_min(1e-12)
        return torch.sqrt(msr)

    def effective_spot_radius(self, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Scalar effective spot radius = mean over fields of the RMS radius."""
        return self.rms_radius(weights).mean()


class BaseOptics(ABC, torch.nn.Module):
    """Abstract optics stage.

    Concrete subclasses register their optimizable variables (curvatures,
    spacings, aspheric coefficients, glass, ...) as attributes and implement
    two things:

      * ``forward()`` -> SpotDiagram         (the ray trace)
      * ``get_theta`` / ``set_theta``        (flat parameter-vector view for LM)

    The flat ``theta`` view is what the Levenberg-Marquardt engine reads and
    writes; keeping it separate from ``nn.Parameter`` state lets the same model
    be driven by either LM (via set_theta) or Adam (via autograd on Parameters).
    """

    @abstractmethod
    def forward(self) -> SpotDiagram:
        """Trace rays and return the spot diagram for the current parameters."""
        ...

    @abstractmethod
    def get_theta(self) -> torch.Tensor:
        """Return the current optimizable variables as a flat 1-D tensor."""

    @abstractmethod
    def set_theta(self, theta: torch.Tensor) -> None:
        """Overwrite the optimizable variables from a flat 1-D tensor."""

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Number of optimizable variables N (length of theta)."""
