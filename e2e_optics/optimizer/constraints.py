"""Soft geometric constraints as LM residuals (Supp. S2.2.2).

The paper does not clamp or clip the design. It appends *residuals* to the same
least-squares vector the TRA/GTRA terms live in, built from the ramp function

    ramp(x) = max(x, 0),

so a constraint contributes nothing while it is satisfied and grows smoothly once
it is violated (Supp. S2.2.2). Because they are residuals, the Levenberg-Marquardt
solver handles them with exactly the same machinery as the image-quality terms --
no separate penalty schedule, no projection step, and the Jacobian rows for
inactive constraints are simply zero.

Every quantity here comes from the "rays as probes" trace
(``lens._trace_packed(..., probes=True)``): the ray-marching distances and angles
that the spot-diagram trace already computes are reused as constraint operands,
rather than re-deriving lens geometry analytically.

Implemented
-----------
``ray_path_residuals``     l_RP, Eq. S41-S42 -- axial ray-marching distance per
                           spacing within [tz_min, tz_max]. Prevents surface
                           overlap and backtracking rays, enforces minimum
                           element thickness and image clearance.
``ray_angle_residuals``    l_RA, Eq. S43-S45 -- angles of incidence and
                           refraction within +-theta_max. Prevents missed
                           surfaces (zeta_I < 0) and total internal reflection
                           (zeta_R < 0).
``surface_normal_residuals`` l_SN, Eq. S46-S47 -- angle between a refractive
                           interface normal and the optical axis within
                           +-theta_max, for manufacturability.
``geometric_residuals``    all three concatenated, ready to append to the TRA or
                           GTRA residual vector.

Defaults follow the paper's own manufacturability specifications (Table S8):
element thickness >= 0.25 mm, air gap >= 0.1 mm, incidence/refraction <= 60 deg,
surface normals <= 30 deg.

Not implemented here: the imaging constraints of Supp. S2.2.3 (distortion l_D,
relative illumination). They need a paraxial reference image height rather than
ray probes; see the extension hooks at the end of this module.
"""
from __future__ import annotations
from typing import Optional, Sequence
import math
import torch

__all__ = ["ramp", "ray_path_residuals", "ray_angle_residuals",
           "surface_normal_residuals", "geometric_residuals",
           "GeometricConstraints", "spacing_kinds_from_optics",
           "distortion_residuals"]


def ramp(x: torch.Tensor) -> torch.Tensor:
    """ramp(x) = max(x, 0) (Supp. S2.2.2).

    Zero and zero-gradient while the constraint holds, so satisfied constraints
    contribute nothing to the residual vector or to the Jacobian.
    """
    return torch.clamp(x, min=0.0)


def _scaled(parts: Sequence[torch.Tensor], n_rays: int) -> torch.Tensor:
    """Concatenate penalty blocks with the 1/sqrt(n_r) scaling of Eq. S42.

    The scaling makes the residual magnitude independent of how many rays are
    traced, so constraint weight does not silently change when the pupil sampling
    is refined.
    """
    if not parts:
        return torch.zeros(0)
    return torch.cat([p.reshape(-1) for p in parts]) / math.sqrt(max(n_rays, 1))


def ray_path_residuals(probes: dict,
                       tz_min: Sequence[float] | float = 0.0,
                       tz_max: Sequence[float] | float = float("inf"),
                       ) -> torch.Tensor:
    """l_RP -- axial ray-marching distance bounds (Eq. S41-S42).

    Parameters
    ----------
    probes : dict from ``lens._trace_packed(..., probes=True)``; uses ``tz``
        of shape (n_spacings, F, W, P).
    tz_min, tz_max : per-spacing lower/upper bounds in mm, or a scalar applied to
        every spacing. A lower bound of at least 0 everywhere prevents
        backtracking rays; inside a refractive element the bounds act as minimum
        and maximum thickness; on the final spacing the lower bound is the image
        clearance (flange distance).

    Returns a 1-D residual tensor, zero where every ray complies.
    """
    tz = probes["tz"]                                   # (K,F,W,P)
    K = tz.shape[0]
    n_rays = int(tz.shape[1] * tz.shape[2] * tz.shape[3])
    lo = _as_per_spacing(tz_min, K, tz)
    hi = _as_per_spacing(tz_max, K, tz)
    # Eq. S41: dz = ramp(max(tz - tz_max, tz_min - tz))
    dz = ramp(torch.maximum(tz - hi, lo - tz))
    return _scaled([dz], n_rays)


def _as_per_spacing(v, K: int, like: torch.Tensor) -> torch.Tensor:
    if isinstance(v, (int, float)):
        out = torch.full((K,), float(v), dtype=like.dtype)
    else:
        v = list(v)
        if len(v) != K:
            raise ValueError(f"expected {K} per-spacing bounds, got {len(v)}")
        out = torch.tensor([float(x) for x in v], dtype=like.dtype)
    return out.reshape(K, 1, 1, 1)


def ray_angle_residuals(probes: dict, theta_max_deg: float = 60.0) -> torch.Tensor:
    """l_RA -- angles of incidence and refraction within +-theta_max (Eq. S43-S45).

    Uses zeta_I = cos^2(theta) and zeta_R = cos^2(theta'), where a negative value
    means a missed surface (incidence) or total internal reflection (refraction).
    Both penalties are ``ramp(cos^2(theta_max) - zeta)``, so they activate before
    the ray actually fails -- which is the point: a hard ray failure has no
    usable gradient, while this residual pushes the design away from failure
    smoothly.
    """
    ci, cr = probes["ci"], probes["cr"]                 # (S,F,W,P) each
    n_rays = int(ci.shape[1] * ci.shape[2] * ci.shape[3])
    c2 = math.cos(math.radians(float(theta_max_deg))) ** 2
    return _scaled([ramp(c2 - ci), ramp(c2 - cr)], n_rays)


def surface_normal_residuals(probes: dict, theta_max_deg: float = 30.0) -> torch.Tensor:
    """l_SN -- interface normal to optical axis within +-theta_max (Eq. S46-S47).

    zeta_SN = cos^2(theta_SN) with theta_SN the angle between the surface normal
    at the ray's intersection point and the optical axis. Penalizing steep normals
    keeps surfaces manufacturable (the paper's specification is <= 30 deg).
    """
    sn = probes["sn"]
    n_rays = int(sn.shape[1] * sn.shape[2] * sn.shape[3])
    c2 = math.cos(math.radians(float(theta_max_deg))) ** 2
    return _scaled([ramp(c2 - sn)], n_rays)


def geometric_residuals(probes: dict,
                        tz_min: Sequence[float] | float = 0.0,
                        tz_max: Sequence[float] | float = float("inf"),
                        theta_max_deg: float = 60.0,
                        normal_max_deg: float = 30.0,
                        weights: Optional[dict] = None) -> torch.Tensor:
    """l_RP + l_RA + l_SN concatenated (Supp. S2.2.2).

    Append the result to the image-quality residual vector before handing it to
    LM::

        def residuals(theta):
            xy, _, pk = lens._trace_packed(pack(theta), probes=True)
            return torch.cat([tra_residuals(xy),
                              geometric_residuals(pk, tz_min=..., tz_max=...)])

    ``weights`` optionally scales each block, e.g. ``{"rp": 10.0}``. The paper
    does not weight the blocks against each other -- the 1/sqrt(n_r) scaling of
    Eq. S42 is the only normalization -- so the default is unweighted.
    """
    w = {"rp": 1.0, "ra": 1.0, "sn": 1.0}
    if weights:
        w.update(weights)
    blocks = [
        w["rp"] * ray_path_residuals(probes, tz_min, tz_max),
        w["ra"] * ray_angle_residuals(probes, theta_max_deg),
        w["sn"] * surface_normal_residuals(probes, normal_max_deg),
    ]
    return torch.cat(blocks)


def spacing_kinds_from_optics(optics) -> list:
    """Label each axial spacing in the probe ``tz`` array as glass / air / image.

    The probe array has one more entry than there are surfaces, because the trace
    starts at the entrance-pupil plane: ``tz[0]`` is the pupil-to-first-surface
    hop, ``tz[k]`` for k>=1 is the spacing that surface k-1 opens, and the last
    entry is the rear-element-to-sensor clearance. Getting that offset wrong
    silently applies a glass thickness bound to an air gap, so derive it from the
    optics rather than typing it out.
    """
    kinds = ["air"]                                  # pupil -> first surface
    n = optics.n_surfaces
    for si in range(n - 1):
        # the spacing FOLLOWING surface si carries index of refraction n_after[si]
        kinds.append("glass" if float(optics.n_after[si]) > 1.0 else "air")
    kinds.append("image")                            # rear element -> sensor
    return kinds


class GeometricConstraints:
    """Bundles the constraint bounds so a residual closure stays readable.

    ``spacing_kinds`` labels each axial spacing so the paper's per-kind
    manufacturability specifications (Table S8) can be applied without
    hand-typing a bound array: ``"glass"`` -> thickness >= ``min_glass``,
    ``"air"`` -> gap >= ``min_air``, ``"image"`` -> clearance >= ``min_image``.

    Example
    -------
    >>> gc = GeometricConstraints.from_optics(lens)   # labels read off the lens
    >>> resid_geo = gc(lens.ray_probes())
    """

    def __init__(self, spacing_kinds: Sequence[str],
                 min_glass: float = 0.25, max_glass: float = float("inf"),
                 min_air: float = 0.1, min_image: float = 0.1,
                 theta_max_deg: float = 60.0, normal_max_deg: float = 30.0,
                 weights: Optional[dict] = None):
        self.spacing_kinds = list(spacing_kinds)
        self.theta_max_deg = float(theta_max_deg)
        self.normal_max_deg = float(normal_max_deg)
        self.weights = dict(weights) if weights else None
        lo, hi = [], []
        for kind in self.spacing_kinds:
            if kind == "glass":
                lo.append(min_glass); hi.append(max_glass)
            elif kind == "air":
                lo.append(min_air); hi.append(float("inf"))
            elif kind == "image":
                lo.append(min_image); hi.append(float("inf"))
            else:
                raise ValueError(f"unknown spacing kind {kind!r}; "
                                 "expected 'glass', 'air' or 'image'")
        self.tz_min, self.tz_max = lo, hi

    @classmethod
    def from_optics(cls, optics, **kw):
        """Build with the spacing kinds read off the optics (recommended).

        >>> gc = GeometricConstraints.from_optics(lens)      # paper defaults
        >>> gc = GeometricConstraints.from_optics(lens, min_glass=0.4)
        """
        return cls(spacing_kinds_from_optics(optics), **kw)

    def __call__(self, probes: dict) -> torch.Tensor:
        return geometric_residuals(probes, self.tz_min, self.tz_max,
                                   self.theta_max_deg, self.normal_max_deg,
                                   self.weights)

    def report(self, probes: dict):
        """Per-block violation summary: count of active terms and worst value."""
        out = {}
        for name, r in (("ray_path", ray_path_residuals(probes, self.tz_min, self.tz_max)),
                        ("ray_angle", ray_angle_residuals(probes, self.theta_max_deg)),
                        ("surface_normal", surface_normal_residuals(probes, self.normal_max_deg))):
            act = int((r > 0).sum())
            out[name] = {"n_terms": int(r.numel()), "n_active": act,
                         "worst": float(r.max()) if r.numel() else 0.0}
        return out


# ---------------------------------------------------------------------------
# Extension hooks: imaging constraints (Supp. S2.2.3)
# ---------------------------------------------------------------------------
def distortion_residuals(*args, **kwargs):
    """l_D -- distortion within +-D_max (Eq. S48-S49).

    STUB (extension hook). Needs the paraxial image height y'_h per field as a
    reference, which is a first-order quantity rather than a ray probe, so it
    does not belong to the S2.2.2 probe family implemented above. Left
    unimplemented rather than approximated: a wrong distortion reference would
    silently bias every optimization that switched it on.
    """
    raise NotImplementedError(
        "Distortion residuals (Supp. S2.2.3, Eq. S48-S49) are a documented "
        "extension hook. They require a paraxial image-height reference y'_h "
        "per field, not a ray probe. Implement the paraxial trace first, then "
        "D_h = (y_h - y'_h) / y'_h and dD = ramp(|D_h| - D_max)."
    )
