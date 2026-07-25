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
    Shapes: d,n : (..., 3); mu : scalar or a tensor broadcastable to the
    batch dims of ``d`` (e.g. ``(1, W, 1)`` for a per-wavelength ratio). The
    trailing vector axis is added here, so the wavelength axis is preserved
    rather than collapsed to a scalar.
    """
    cos_i = -(d * n).sum(-1, keepdim=True)
    n = torch.where(cos_i < 0, -n, n)        # ensure normal opposes incidence
    cos_i = cos_i.abs()
    mu_ = mu[..., None] if torch.is_tensor(mu) else mu
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
    n_after    : refractive index of the medium AFTER this surface, at the
                 d-line (n_d). Index BEFORE the first surface is air = 1.0.
    is_stop    : marks the aperture stop (metadata)
    semi_aperture : clear semi-diameter (mm). Leave as None (default) to DERIVE
                 it from the traced ray footprint -- the paper specifies aperture
                 by EPD / f-number and never gives element semi-diameters as
                 inputs (Supp. S2.1, Table S8). Set a float only to impose a
                 mechanical override (a real barrel or a stop smaller than the
                 light), which `aperture_report()` will then flag if the traced
                 rays exceed it.
    abbe       : Abbe number v_d of the medium after this surface. Only used
                 when chromatic dispersion is enabled. 0.0 (default) means "no
                 dispersion data" -> that surface stays at its scalar n_after
                 for every wavelength (e.g. air, or a monochromatic glass).
    dpgf       : partial-dispersion deviation dP_g,F (default 0.0 = normal
                 glass line); refines the Hartmann fit for special glasses.
    """
    curvature: float = 0.0
    conic: float = 0.0
    asph: Sequence[float] = field(default_factory=list)
    thickness: float = 1.0
    n_after: float = 1.0
    is_stop: bool = False
    semi_aperture: Optional[float] = None
    abbe: float = 0.0
    dpgf: float = 0.0


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
                 dispersion: bool = True,
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
        dispersion    : if True (default), glass surfaces carrying Abbe data are
                        traced with a per-wavelength Hartmann index, so multiple
                        wavelengths show chromatic aberration. If False, every
                        wavelength uses the scalar d-line n_after (monochromatic
                        -- reproduces the pre-dispersion behavior bit-for-bit).
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
        # Aperture is specified by the EPD (above); element semi-diameters are
        # DERIVED from the ray footprint unless explicitly overridden. Keep the
        # declared values (possibly all None) so `effective_semi_apertures()` can
        # honour real mechanical limits while `aperture_report()` flags overflow.
        self._semi_aperture_declared = (
            None if all(s.semi_aperture is None for s in surfaces)
            else [s.semi_aperture for s in surfaces])
        self._sa_cache = None
        self.is_stop = [s.is_stop for s in surfaces]

        # --- per-surface x per-wavelength index table (S, W) -----------------
        # Glass indices are FIXED (not optimization variables), so this table is
        # precomputed once and never enters the autodiff Jacobian. For a glass
        # surface with Abbe data and dispersion enabled, each wavelength gets its
        # own Hartmann index n(lambda); otherwise the scalar n_after is broadcast
        # across all wavelengths (the monochromatic path).
        self.dispersion = bool(dispersion)
        self.abbe = [float(s.abbe) for s in surfaces]
        self.dpgf = [float(s.dpgf) for s in surfaces]
        W = self.wavelengths_um.numel()
        n_tab = n_after.reshape(S, 1).repeat(1, W).clone()          # (S, W)
        if self.dispersion:
            for si, s in enumerate(surfaces):
                if s.n_after > 1.0 and s.abbe > 0.0:                 # a glass with data
                    for wi, wl in enumerate(self.wavelengths_um.tolist()):
                        n_tab[si, wi] = hartmann_index(s.n_after, s.abbe,
                                                       float(wl), s.dpgf)
        self.register_buffer("n_after_w", n_tab)

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
    def _trace_packed(self, packed: torch.Tensor, probes: bool = False):
        """Core trace as a pure function of the packed parameter vector.

        Returns xy (F, W, P, 2) and valid (F, W, P).

        With ``probes=True`` a third value is returned: a dict of intermediate
        per-ray geometry, following the paper's "rays as probes" idea (Supp.
        S2.2.2) where quantities collected during the spot-diagram trace are
        reused to derive element apertures and geometric constraints:

          ``r``   (S, F, W, P)   radial height |(x,y)| at each surface
          ``tz``  (S, F, W, P)   axial marching distance into each spacing
          ``ci``  (S, F, W, P)   cos^2 of the angle of incidence
          ``cr``  (S, F, W, P)   cos^2 of the angle of refraction

        Negative ``ci``/``cr`` flag a missed surface and total internal
        reflection respectively. Everything stays attached to the autograd graph,
        so residuals built from these probes are differentiable w.r.t. theta.
        Default ``probes=False`` leaves the traced values and their gradients
        bit-identical to the plain trace.
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
        pr_r, pr_tz, pr_ci, pr_cr, pr_reach = [], [], [], [], []

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
            if probes:
                # radial height at this surface and the axial distance travelled
                # through the spacing that ENDS here (Supp. S2.2.2 uses t_z).
                pr_r.append(torch.sqrt((x * x + y * y).clamp_min(0.0) + 1e-30))
                pr_tz.append(z - z0)
                # conic reachability: the sag square-root argument. Negative =>
                # the surface does not extend to this radius, i.e. the ray misses.
                pr_reach.append(1.0 - (1.0 + k) * c * c * (x * x + y * y))
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
            # n_after_w is (S, W): the precomputed per-wavelength index. With
            # dispersion off (or air / no-Abbe surfaces) every wavelength shares
            # the scalar n_after, so this reduces to the monochromatic trace.
            n_aft = self.n_after_w[si].to(dtype)            # (W,)
            mu_w = (n_before / n_aft).reshape(1, W, 1)      # (1,W,1)
            if probes:
                # zeta_I = cos^2(theta), zeta_R = cos^2(theta') of Supp. S2.2.2.
                # zeta_R < 0 is exactly the total-internal-reflection signal;
                # both are smooth in theta, unlike the boolean masks.
                ci2 = ((d * nrm).sum(-1)) ** 2
                sin2_t = (mu_w.squeeze(-1) ** 2) * (1.0 - ci2)
                pr_ci.append(ci2)
                pr_cr.append(1.0 - sin2_t)
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
        if probes:
            # the last spacing (rear element -> image plane) is the image
            # clearance that Supp. S2.2.2 constrains alongside the glass/air gaps
            pr_tz.append(z_img - z0)
            pk = {"r": torch.stack(pr_r), "tz": torch.stack(pr_tz),
                  "ci": torch.stack(pr_ci), "cr": torch.stack(pr_cr),
                  "reach": torch.stack(pr_reach), "valid": valid}
            return xy, valid, pk
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

    # ---- apertures: specified by EPD, DERIVED per element ------------------
    @property
    def semi_aperture(self):
        """Per-surface clear semi-diameter (mm) actually in force.

        Derived from the ray footprint by default, so it tracks the design as the
        optimizer moves it; a declared value overrides its surface. Cached per
        parameter state so repeated layout calls do not re-trace.
        """
        key = None
        if self._semi_aperture_declared is None or any(
                v is None for v in self._semi_aperture_declared):
            key = float(self._pack().sum())          # cheap design fingerprint
            if self._sa_cache is not None and self._sa_cache[0] == key:
                return self._sa_cache[1]
        vals = [float(v) for v in self.effective_semi_apertures()]
        if key is not None:
            self._sa_cache = (key, vals)
        return vals

    def ray_probes(self):
        """Per-ray geometric probes for the current design (see _trace_packed)."""
        _, _, pk = self._trace_packed(self._pack(), probes=True)
        return pk

    def clear_semi_apertures(self, margin: float = 1.05) -> torch.Tensor:
        """Clear semi-diameter (mm) of each element, DERIVED from the ray trace.

        The paper specifies the aperture of a system by its entrance pupil
        diameter or f-number (Supp. S2.1, Table S8) -- element semi-diameters are
        never given as inputs; they follow from the light the system actually
        carries. So we take the largest ray radius reaching each surface over the
        full field x wavelength x pupil grid and scale it by ``margin``.

        This matters during optimization: with thickness (or curvature) free, the
        off-axis beam walks up and down the rear elements, so any hand-declared
        semi-diameter goes stale and the layout draws rays passing outside the
        glass they refract at. A derived aperture cannot go stale.

        Returns a detached (S,) tensor -- geometry for drawing and reporting, not
        an optimization variable.
        """
        r = self.ray_probes()["r"]                      # (S,F,W,P)
        v = self.ray_probes()["valid"]                  # (F,W,P)
        r = torch.where(v.unsqueeze(0), r, torch.zeros_like(r))
        return (r.reshape(self.n_surfaces, -1).max(dim=1).values * float(margin)).detach()

    def effective_semi_apertures(self, margin: float = 1.05) -> torch.Tensor:
        """Semi-apertures used for drawing: the declared override where the user
        gave one, else the derived value."""
        der = self.clear_semi_apertures(margin)
        if self._semi_aperture_declared is None:
            return der
        out = der.clone()
        for si, v in enumerate(self._semi_aperture_declared):
            if v is not None:
                out[si] = float(v)
        return out

    def aperture_report(self, margin: float = 1.05):
        """Per-surface table of declared vs required clear semi-diameter.

        Returns a list of dicts with keys ``surface``, ``declared`` (None if the
        aperture is derived), ``required`` (max traced ray radius, mm),
        ``recommended`` (required x margin) and ``overflow`` (mm by which the
        traced light exceeds the declared value; > 0 means rays pass outside the
        element -- the defect this replaces).
        """
        req = (self.clear_semi_apertures(1.0)).tolist()
        rows = []
        for si in range(self.n_surfaces):
            dec = (None if self._semi_aperture_declared is None
                   else self._semi_aperture_declared[si])
            rows.append({"surface": si,
                         "declared": None if dec is None else float(dec),
                         "required": float(req[si]),
                         "recommended": float(req[si]) * float(margin),
                         "overflow": (0.0 if dec is None
                                      else max(0.0, float(req[si]) - float(dec)))})
        return rows

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

    def ray_paths(self, field_index: int = 0, wavelength_index: int = 0,
                  n_fan: int = 9, pupil_radius: Optional[float] = None):
        """Meridional ray-fan polylines for a layout plot, in mm.

        Traces ``n_fan`` rays evenly across the pupil (y only, the meridional
        plane) for one field and wavelength, recording each ray's (z, y) vertex
        at every surface and at the image plane.

        ``pupil_radius`` sets the half-height of the launched fan (mm). Defaults
        to the entrance-pupil radius ``0.5 * epd``; pass a smaller value (e.g.
        the limiting clear semi-aperture) to keep the fan inside the glass for a
        clean layout drawing.

        ``wavelength_index`` selects which traced wavelength's index to use, so
        chromatic fans bend by the correct per-colour index.

        Returns
        -------
        zs : (n_fan, S+2) tensor -- z of each ray at the pupil, at its true
             intersection on every surface, and at the image plane. Each ray
             carries its OWN z per surface (a marginal ray meets a curved
             surface at a different z than the chief ray), so plotting (zs, ys)
             row-by-row puts every bend exactly on the drawn surface curve.
        ys : (n_fan, S+2) tensor -- y of each ray at those same nodes.

        Purely for visualization (detached); does not touch the autodiff path.
        """
        with torch.no_grad():
            dtype = self.dtype
            S = self.n_surfaces
            wi = int(wavelength_index)
            z_vert = torch.cat([torch.zeros(1, dtype=dtype),
                                torch.cumsum(self.thick, 0)])       # (S+1,)
            z_img = z_vert[-1]
            rp = 0.5 * self.epd if pupil_radius is None else float(pupil_radius)
            # meridional fan across the pupil (y in [-rp, rp], x = 0)
            py = torch.linspace(-rp, rp, n_fan, dtype=dtype)
            pos = torch.zeros(n_fan, 3, dtype=dtype)
            pos[:, 1] = py
            u = torch.deg2rad(self.fields_deg[field_index]).to(dtype)
            d = torch.zeros(n_fan, 3, dtype=dtype)
            d[:, 1] = torch.sin(u); d[:, 2] = torch.cos(u)

            nodes_z = [torch.zeros(n_fan, dtype=dtype)]   # pupil plane, per ray
            nodes_y = [pos[:, 1].clone()]

            n_before = torch.ones(1, dtype=dtype)
            for si in range(S):
                c = self.curv[si]; k = self.conic[si]; a = self.asph[si]
                zc = z_vert[si]
                x0, y0, z0 = pos[:, 0], pos[:, 1], pos[:, 2]
                dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
                t = (zc - z0) / dz
                for _ in range(self.newton_iters):
                    x = x0 + t * dx; y = y0 + t * dy; z = z0 + t * dz
                    uu = x * x + y * y
                    f = z - zc - asphere_sag(uu, c, k, a)
                    dsag = asphere_dsag_du(uu, c, k, a)
                    fp = dz - dsag * 2.0 * ((x0 + t * dx) * dx + (y0 + t * dy) * dy)
                    t = t - f / fp
                x = x0 + t * dx; y = y0 + t * dy; z = z0 + t * dz
                pos = torch.stack([x, y, z], -1)
                nodes_z.append(z.clone())        # per-ray true intersection z
                nodes_y.append(y.clone())
                uu = x * x + y * y
                dsag = asphere_dsag_du(uu, c, k, a)
                nrm = torch.stack([-2.0 * x * dsag, -2.0 * y * dsag,
                                   torch.ones_like(x)], -1)
                nrm = nrm / nrm.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                mu = (n_before / self.n_after_w[si, wi]).reshape(1)
                d, _ = refract(d, nrm, mu)
                n_before = self.n_after_w[si, wi].reshape(1)
            # march to image plane
            t = (z_img - pos[:, 2]) / d[:, 2]
            y_img = pos[:, 1] + t * d[:, 1]
            nodes_z.append(z_img.expand(n_fan).clone())
            nodes_y.append(y_img)

            zs = torch.stack(nodes_z, dim=1)                         # (n_fan, S+2)
            ys = torch.stack(nodes_y, dim=1)                         # (n_fan, S+2)
        return zs.detach(), ys.detach()
