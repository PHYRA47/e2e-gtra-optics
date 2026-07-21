"""Differentiable, rotationally-symmetric sequential ray tracer.

Traces collimated ray bundles (object at infinity) from the entrance pupil,
through a stack of even-aspheric refractive surfaces, to a flat image plane,
producing the SpotDiagram consumed by the bridge stage.

Faithful to Cote et al. (2026), Supp. S1:
  * even-aspheric sag (Eq. 9)
  * Newton ray-marching to each surface (Eq. S8-S10)
  * vector Snell refraction with the cos^2(theta') form (Eq. S12-S13)
  * Hartmann dispersion model (Eq. 8 / S4)

Design for autodiff
--------------------
The whole trace is exposed as a *pure function of a flat parameter vector*
``spot_from_theta(theta) -> epsilon`` so that the LM engine can take its
Jacobian with forward-mode AD (torch.func.jacfwd). Trainable variables are a
masked subset of [curvatures, conics, aspheric coeffs, thicknesses]; fixed
entries are carried as a buffer and re-inserted inside the trace.

Numerical care taken:
  * the sag derivative is computed in u = r^2, so nothing divides by r at r->0;
  * Newton marching uses a fixed iteration count (differentiable);
  * refraction clamps the TIR discriminant and reports a validity mask.

Vignetting / ray-aiming and diffractive (DOE) surfaces are stubbed with clear
hooks -- see ``diffract`` and ``RotationallySymmetricLens.aim_rays``.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Sequence
import math
import torch

from .base import BaseOptics, SpotDiagram


# ----------------------------------------------------------------------------
# Dispersion (Eq. 8 / S4)
# ----------------------------------------------------------------------------
def hartmann_index(nd: float, vd: float, wavelength_um: float,
                   dPgF: float = 0.0) -> float:
    """Refractive index at ``wavelength_um`` via the Hartmann model n=A+C/(l-B).

    A, B, C derived from (n_d, v_d, dP_g,F) exactly as in Supp. Eqs. S5-S7.
    Fraunhofer lines (um): C 0.6563, d 0.5876, F 0.4861, g 0.4358.
    """
    lC, ld, lF, lg = 0.6563, 0.5876, 0.4861, 0.4358
    PgF = 0.6438 - 0.001682 * vd + dPgF                       # Eq. S3
    B = (-lC * lF + lC * lg + PgF * (lC * lg - lF * lg)) / \
        (-lF + lg + PgF * (lC - lF))                          # Eq. S5
    C = (B - lC) * (B - lF) * (nd - 1.0) / (vd * (lC - lF))   # Eq. S6
    A = (B * nd + C - ld * nd) / (B - ld)                     # Eq. S7
    return A + C / (wavelength_um - B)


# ----------------------------------------------------------------------------
# Even-aspheric sag and its derivative (Eq. 9)
# ----------------------------------------------------------------------------
def asphere_sag(u: torch.Tensor, c: torch.Tensor, k: torch.Tensor,
                asph: torch.Tensor) -> torch.Tensor:
    """Even-aspheric sag z(r) as a function of u = r^2.

    z = c*u / (1 + sqrt(1 - (1+k) c^2 u)) + sum_j a_j u^j   (j = 2..A+1)
    ``asph`` are the coefficients [a2, a3, ...] for powers u^2, u^3, ...
    (i.e. r^4, r^6, ... -- the even asphere convention, degrees 4, 6, 8, ...).
    """
    disc = torch.clamp(1.0 - (1.0 + k) * c * c * u, min=1e-9)
    base = c * u / (1.0 + torch.sqrt(disc))
    poly = torch.zeros_like(u)
    for j, a in enumerate(asph):            # a_j multiplies u^(j+2) = r^(2j+4)
        poly = poly + a * u.pow(j + 2)
    return base + poly


def asphere_dsag_du(u: torch.Tensor, c: torch.Tensor, k: torch.Tensor,
                    asph: torch.Tensor) -> torch.Tensor:
    """d(sag)/du. Then d(sag)/dx = dsag_du * 2x, d(sag)/dy = dsag_du * 2y."""
    disc = torch.clamp(1.0 - (1.0 + k) * c * c * u, min=1e-9)
    s = torch.sqrt(disc)
    D = 1.0 + s
    # d/du [ c u / D ] with dD/du = -(1+k)c^2 / (2 s)
    dbase = (c * D + c * u * (1.0 + k) * c * c / (2.0 * s)) / (D * D)
    dpoly = torch.zeros_like(u)
    for j, a in enumerate(asph):
        dpoly = dpoly + (j + 2) * a * u.pow(j + 1)
    return dbase + dpoly


# ----------------------------------------------------------------------------
# Refraction (Eq. S11-S13) -- robust vector form
# ----------------------------------------------------------------------------
def refract(d: torch.Tensor, n: torch.Tensor, mu: torch.Tensor):
    """Refract unit directions ``d`` at unit normals ``n`` with ratio mu=n1/n2.

    Returns (d_out, ok) where ok is False for total internal reflection.
    Normals are flipped per-ray so the incidence cosine is positive.
    Shapes: d,n : (..., 3); mu : scalar or (...,).
    """
    cos_i = -(d * n).sum(-1, keepdim=True)
    n = torch.where(cos_i < 0, -n, n)        # ensure normal opposes incidence
    cos_i = cos_i.abs()
    mu_ = mu.reshape(*([1] * (d.dim() - 1)), 1) if torch.is_tensor(mu) else mu
    sin2_t = (mu_ * mu_) * (1.0 - cos_i * cos_i)      # Eq. S13 rearranged
    ok = (sin2_t <= 1.0).squeeze(-1)
    cos_t = torch.sqrt(torch.clamp(1.0 - sin2_t, min=0.0))
    d_out = mu_ * d + (mu_ * cos_i - cos_t) * n       # Eq. S12
    return d_out, ok


def diffract(*args, **kwargs):
    """Grating-modified refraction for diffractive/metasurface interfaces.

    STUB (extension hook). Implements the generalized law of refraction
    (Eq. S15-S18): n' sin(theta') - n sin(theta) = (lambda/2pi) dphi/dr, with
    the phase gradient dphi/dr obtained by AD from a continuous phase profile
    phi(r) = sum b_j r^(2j). Left unimplemented in v1 (all-refractive slice).
    """
    raise NotImplementedError(
        "Diffractive surfaces are a documented v1 extension hook -- see "
        "docs/ANALYSIS_AND_DESIGN.md sec. 3.1 and Supp. S1.2.3."
    )


# ----------------------------------------------------------------------------
# Surface description
# ----------------------------------------------------------------------------
@dataclass
class Surface:
    """One rotationally-symmetric refractive surface.

    curvature  : 1/radius in 1/mm (0 = flat)
    conic      : conic constant k
    asph       : aspheric coefficients [a2, a3, ...] (powers r^4, r^6, ...)
    thickness  : axial distance (mm) from this surface's vertex to the next
                 (or to the image plane, for the last surface)
    n_after    : refractive index of the medium AFTER this surface
                 (index BEFORE the first surface is air = 1.0)
    is_stop    : marks the aperture stop (metadata)
    semi_aperture : clear semi-diameter (mm) for layout plots / vignetting
    """
    curvature: float = 0.0
    conic: float = 0.0
    asph: Sequence[float] = field(default_factory=list)
    thickness: float = 1.0
    n_after: float = 1.0
    is_stop: bool = False
    semi_aperture: float = 5.0


# ----------------------------------------------------------------------------
# Pupil sampling: jittered concentric rings (Supp. S1.3.1, Fig. S1)
# ----------------------------------------------------------------------------
def concentric_pupil(n_rings: int, jitter: bool = True,
                     generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Sample the unit disk on concentric rings; ring i holds 6*i points.

    Returns (P, 2) normalized pupil coordinates in [-1, 1]^2 (inside the disk).
    Full-disk sampling (not half) so off-axis meridional asymmetry is captured.
    """
    pts = [torch.zeros(1, 2)]
    for i in range(1, n_rings + 1):
        radius = i / n_rings
        m = 6 * i
        ang = torch.arange(m, dtype=torch.float32) / m * 2 * math.pi
        if jitter:
            j = torch.rand(m, generator=generator) if generator is not None else torch.rand(m)
            ang = ang + (j - 0.5) * (2 * math.pi / m)
        r = radius * torch.ones(m)
        pts.append(torch.stack([r * torch.cos(ang), r * torch.sin(ang)], -1))
    return torch.cat(pts, 0)


# ----------------------------------------------------------------------------
# The lens
# ----------------------------------------------------------------------------
class RotationallySymmetricLens(BaseOptics):
    """Sequential rotationally-symmetric aspheric lens (all-refractive)."""

    def __init__(self,
                 surfaces: List[Surface],
                 epd: float,
                 fields_deg: Sequence[float],
                 wavelengths_um: Sequence[float] = (0.5876,),
                 n_pupil_rings: int = 12,
                 variables: Optional[Sequence[str]] = None,
                 newton_iters: int = 12,
                 pupil_jitter: bool = True,
                 dtype: torch.dtype = torch.float64,
                 seed: int = 0):
        """
        surfaces      : list of Surface (last thickness = distance to image plane)
        epd           : entrance pupil diameter (mm); pupil placed at surface 0
        fields_deg    : field angles (deg); object at infinity
        wavelengths_um: traced wavelengths (um)
        variables     : which quantities are optimizable, any of
                        {"curvature", "conic", "asph", "thickness"}. Curvatures,
                        conics and aspherics of ALL surfaces and thicknesses of
                        all-but-the-image gap are made variable per selection.
                        Default: curvature + asph + thickness (matches the toy).
        newton_iters  : Newton iterations for aspheric ray-marching
        dtype         : float64 recommended for LM conditioning
        """
        super().__init__()
        self.dtype = dtype
        self.newton_iters = int(newton_iters)
        self.fields_deg = torch.tensor(list(fields_deg), dtype=dtype)
        self.wavelengths_um = torch.tensor(list(wavelengths_um), dtype=dtype)
        self.epd = float(epd)
        S = len(surfaces)
        self.n_surfaces = S
        self.max_asph = max((len(s.asph) for s in surfaces), default=0)

        # --- packed parameter tensors (fixed = buffers; values live here) ----
        curv = torch.tensor([s.curvature for s in surfaces], dtype=dtype)
        conic = torch.tensor([s.conic for s in surfaces], dtype=dtype)
        asph = torch.zeros(S, self.max_asph, dtype=dtype)
        for i, s in enumerate(surfaces):
            if len(s.asph):
                asph[i, :len(s.asph)] = torch.tensor(s.asph, dtype=dtype)
        thick = torch.tensor([s.thickness for s in surfaces], dtype=dtype)
        n_after = torch.tensor([s.n_after for s in surfaces], dtype=dtype)

        self.register_buffer("curv", curv)
        self.register_buffer("conic", conic)
        self.register_buffer("asph", asph)
        self.register_buffer("thick", thick)
        self.register_buffer("n_after", n_after)
        self.semi_aperture = [s.semi_aperture for s in surfaces]
        self.is_stop = [s.is_stop for s in surfaces]

        # --- build the variable mask over [curv | conic | asph | thick] ------
        if variables is None:
            variables = ("curvature", "asph", "thickness")
        self.variables = tuple(variables)
        masks = []
        masks.append(torch.full((S,), "curvature" in variables, dtype=torch.bool))
        masks.append(torch.full((S,), "conic" in variables, dtype=torch.bool))
        am = torch.zeros(S, self.max_asph, dtype=torch.bool)
        if "asph" in variables:
            for i, s in enumerate(surfaces):
                am[i, :len(s.asph)] = True         # only real coeffs are variable
        masks.append(am.reshape(-1))
        tm = torch.full((S,), "thickness" in variables, dtype=torch.bool)
        tm[-1] = False                              # image distance held (focus fixed here)
        masks.append(tm)
        self.register_buffer("var_mask", torch.cat(masks))

        # pupil sampling (constant across theta)
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("pupil", concentric_pupil(n_pupil_rings, pupil_jitter, g).to(dtype))

    # ---- flat packed <-> structured -------------------------------------
    def _pack(self) -> torch.Tensor:
        return torch.cat([self.curv, self.conic, self.asph.reshape(-1), self.thick])

    def _unpack(self, packed: torch.Tensor):
        S, A = self.n_surfaces, self.max_asph
        i = 0
        curv = packed[i:i + S]; i += S
        conic = packed[i:i + S]; i += S
        asph = packed[i:i + S * A].reshape(S, A); i += S * A
        thick = packed[i:i + S]
        return curv, conic, asph, thick

    # ---- BaseOptics interface -------------------------------------------
    @property
    def n_params(self) -> int:
        return int(self.var_mask.sum().item())

    def get_theta(self) -> torch.Tensor:
        return self._pack()[self.var_mask].clone()

    def set_theta(self, theta: torch.Tensor) -> None:
        packed = self._pack().clone()
        packed[self.var_mask] = theta.to(packed.dtype)
        curv, conic, asph, thick = self._unpack(packed)
        with torch.no_grad():
            self.curv.copy_(curv); self.conic.copy_(conic)
            self.asph.copy_(asph); self.thick.copy_(thick)

    # ---- the trace -------------------------------------------------------
    def _trace_packed(self, packed: torch.Tensor):
        """Core trace as a pure function of the packed parameter vector.

        Returns xy (F, W, P, 2) and valid (F, W, P).
        """
        curv, conic, asph, thick = self._unpack(packed)
        dtype = packed.dtype
        pupil = self.pupil.to(dtype)                       # (P,2) in [-1,1]
        F = self.fields_deg.numel(); W = self.wavelengths_um.numel(); P = pupil.shape[0]

        # vertex z of each surface (stop/pupil at z=0 == surface 0 vertex)
        z_vert = torch.cat([torch.zeros(1, dtype=dtype), torch.cumsum(thick, 0)])  # (S+1,)
        z_img = z_vert[-1]

        # --- ray initialization (collimated bundles) ---
        rp = 0.5 * self.epd                                # pupil radius mm
        px = (pupil[:, 0] * rp)                            # (P,)
        py = (pupil[:, 1] * rp)
        # positions (F,W,P,3) at z=0
        pos = torch.zeros(F, W, P, 3, dtype=dtype)
        pos[..., 0] = px.reshape(1, 1, P)
        pos[..., 1] = py.reshape(1, 1, P)
        # directions from field angle (object at infinity): d=(0,sin u,cos u)
        u = torch.deg2rad(self.fields_deg).to(dtype)       # (F,)
        d = torch.zeros(F, W, P, 3, dtype=dtype)
        d[..., 1] = torch.sin(u).reshape(F, 1, 1)
        d[..., 2] = torch.cos(u).reshape(F, 1, 1)

        valid = torch.ones(F, W, P, dtype=torch.bool)

        n_before = torch.ones(W, dtype=dtype)              # air before first surface
        for si in range(self.n_surfaces):
            c = curv[si]; k = conic[si]; a = asph[si]
            zc = z_vert[si]
            # ---- Newton march to surface (solve z0+t dz = zc + sag(rho^2)) ----
            x0, y0, z0 = pos[..., 0], pos[..., 1], pos[..., 2]
            dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]
            t = (zc - z0) / dz                             # plane-at-vertex guess
            for _ in range(self.newton_iters):
                x = x0 + t * dx; y = y0 + t * dy; z = z0 + t * dz
                uu = x * x + y * y
                sag = asphere_sag(uu, c, k, a)
                f = z - zc - sag
                dsag_du = asphere_dsag_du(uu, c, k, a)
                drho2_dt = 2.0 * ((x0 + t * dx) * dx + (y0 + t * dy) * dy)
                fp = dz - dsag_du * drho2_dt
                t = t - f / fp
            x = x0 + t * dx; y = y0 + t * dy; z = z0 + t * dz
            pos = torch.stack([x, y, z], -1)
            valid = valid & torch.isfinite(t) & (t > -1e3)

            # ---- surface normal n ∝ (-2x dsag_du, -2y dsag_du, 1) ----
            uu = x * x + y * y
            dsag_du = asphere_dsag_du(uu, c, k, a)
            nx = -2.0 * x * dsag_du; ny = -2.0 * y * dsag_du
            nz = torch.ones_like(nx)
            nrm = torch.stack([nx, ny, nz], -1)
            nrm = nrm / nrm.norm(dim=-1, keepdim=True).clamp_min(1e-12)

            # ---- refraction: mu = n_before / n_after (per wavelength) ----
            n_aft = self.n_after[si] * torch.ones(W, dtype=dtype) if self.n_after.dim() == 1 else self.n_after[si]
            # dispersion extension point: replace n_aft with hartmann_index(...) per wavelength
            mu_w = (n_before / n_aft).reshape(1, W, 1)      # (1,W,1)
            d, ok = refract(d, nrm, mu_w)
            valid = valid & ok
            n_before = n_aft

        # ---- march to image plane (flat) ----
        x0, y0, z0 = pos[..., 0], pos[..., 1], pos[..., 2]
        dz = d[..., 2]
        t = (z_img - z0) / dz
        xi = x0 + t * d[..., 0]; yi = y0 + t * d[..., 1]
        xy = torch.stack([xi, yi], -1)                      # (F,W,P,2)
        # Flag rays that land absurdly far from the paraxial image (near-grazing
        # edge rays that refract but miss any sensible image region). They are
        # kept finite for the Jacobian but excluded from centroid/RMS/PSF via the
        # validity mask. Threshold = a few image heights.
        max_field_mm = float(torch.tan(torch.deg2rad(self.fields_deg.abs().max()))
                             ) * abs(float(z_img.detach())) + 0.5 * self.epd
        far = xy.norm(dim=-1) > (5.0 * max_field_mm + 10.0)
        valid = valid & (~far)
        # sanitize failed rays so the LM Jacobian stays finite; mask handles PSF
        xy = torch.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)
        return xy, valid

    def spot_from_theta(self, theta: torch.Tensor) -> torch.Tensor:
        """Pure function theta -> flat epsilon (2*F*W*P,). Used by LM's jacfwd."""
        packed = self._pack().clone()
        packed = packed.to(theta.dtype)
        packed[self.var_mask] = theta
        xy, _ = self._trace_packed(packed)
        return xy.reshape(-1)

    def forward(self) -> SpotDiagram:
        xy, valid = self._trace_packed(self._pack())
        return SpotDiagram(xy=xy, valid=valid,
                           fields_deg=self.fields_deg, wavelengths_um=self.wavelengths_um)

    def spot_object_from_flat(self, eps_flat: torch.Tensor) -> SpotDiagram:
        """Wrap a flat epsilon (2*F*W*P,) as a SpotDiagram for backward-mode AD.

        Used by the joint loop: it needs a leaf ``eps`` it can differentiate the
        task loss against (dL/d eps). This reshapes ``eps_flat`` to (F,W,P,2) and
        pairs it with the CURRENT validity mask / fields / wavelengths so the
        bridge can build a PSF from it. ``eps_flat`` carries the autograd graph;
        the mask is a constant.
        """
        F = self.fields_deg.numel()
        W = self.wavelengths_um.numel()
        P = self.pupil.shape[0]
        xy = eps_flat.reshape(F, W, P, 2)
        with torch.no_grad():
            _, valid = self._trace_packed(self._pack())
        return SpotDiagram(xy=xy, valid=valid,
                           fields_deg=self.fields_deg, wavelengths_um=self.wavelengths_um)

    # ---- extension hook --------------------------------------------------
    def aim_rays(self, *a, **k):
        """Vignetting factor estimation + iterative ray aiming (Supp. S1.3.2).

        STUB. v1 assumes the stop is at the front (pupil == surface 0) with no
        vignetting, which is exact for the front-stop toy lens. For internal
        stops / mechanical vignetting, implement the three vignetting factors
        and the linearized aiming of Eqs. S19-S22 here.
        """
        raise NotImplementedError("Ray aiming is a v1 extension hook (Supp. S1.3.2).")

    # ---- layout geometry for plotting -----------------------------------
    def surface_profile(self, si: int, n: int = 200):
        """(r, z) polyline of surface ``si`` for layout plots, in mm."""
        sa = self.semi_aperture[si]
        r = torch.linspace(-sa, sa, n, dtype=self.dtype)
        u = r * r
        z = asphere_sag(u, self.curv[si], self.conic[si], self.asph[si])
        z_vert = torch.cat([torch.zeros(1, dtype=self.dtype), torch.cumsum(self.thick, 0)])
        return r.detach(), (z + z_vert[si]).detach()
