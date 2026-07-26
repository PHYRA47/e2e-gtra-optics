"""Diagnostic and comparison plots for the e2e_optics pipeline.

This is the reusable visualization layer -- every figure produced during the
build lives here as a function you can call on any optics / bridge / result, so
the plots are part of the code base rather than throwaway script cells.

Two families:

*Single-state diagnostics* (one optics):
    plot_layout(optics)          -- ray-fan layout + surface profiles
    plot_spot_diagram(optics)         -- per-field spot scatter
    plot_psf(optics, bridge)          -- geometric PSF grids
    plot_convergence(history)       -- LM / task-loss curves

*Before / after comparison* (paper Fig. 4 style):
    compare_designs(optics, theta_a, theta_b, ...)   -- side-by-side layout+spots
    plot_restoration(scene, blurred, restored)    -- capture vs restored triptych

*Progressive / evolution view* (question 3):
    OptimizationRecorder    -- callback that snapshots theta each LM iteration
    plot_spot_evolution(optics, recorder)           -- spot vs iteration grid
    animate_spot_evolution(optics, recorder, path)  -- GIF of the spot shrinking

All functions return a matplotlib Figure so you can further tweak or save it.
Nothing here touches autograd -- everything is detached and CPU-friendly.
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence, List
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

from .optics.raytrace import asphere_sag

# consistent, colour-blind-safe field palette used across every figure
FIELD_COLORS = ['#2166ac', '#b2182b', '#1a9850', '#762a83', '#e08214',
                '#01665e', '#8c510a', '#54278f']


# --------------------------------------------------------------------------- #
#  customizable colour / font scheme
# --------------------------------------------------------------------------- #
@dataclass
class VizStyle:
    """One place to control the look of every e2e_optics figure.

    Change a field to restyle *all* plots consistently. Either mutate the
    module-level ``DEFAULT_STYLE`` in place::

        from e2e_optics import viz
        viz.DEFAULT_STYLE.glass_facecolor = '#cfe8ff'
        viz.DEFAULT_STYLE.field_colors = ['crimson', 'navy']

    or build a one-off and pass it through the ``style=`` argument of any
    plotting function::

        dark = viz.VizStyle(window_facecolor='#111', figure_facecolor='#111',
                            font_color='w', glass_facecolor='#22364a')
        viz.plot_layout(optics, style=dark)

    Groups map onto the four things you asked to control:
      * **surfaces**  -> glass_* / surface_edge_*
      * **field lines** -> field_colors / ray_*
      * **window / background** -> window_facecolor / figure_facecolor / grid_*
      * **font** -> font_color / font_family / *_size
    """
    # --- surfaces (the glass elements) ---
    glass_facecolor: str = '#bfe0f5'      # fill of a closed optics element
    glass_alpha: float = 0.0              # 0 = transparent (outline only)
    surface_edge_color: str = '#14405c'   # outline of the glass
    surface_edge_lw: float = 1.4
    # --- field lines (traced rays) ---
    field_colors: Sequence[str] = field(default_factory=lambda: list(FIELD_COLORS))
    # per-wavelength colours used when a layout is drawn chromatically; matched
    # to the F/d/C order (short->long => blue->green->red, paper convention)
    wavelength_colors: Sequence[str] = field(
        default_factory=lambda: ['#2166ac', '#1a9850', '#b2182b'])
    ray_lw: float = 0.9
    ray_alpha: float = 0.9
    # --- entrance pupil ---
    pupil_color: str = '#888888'
    # --- window / background ---
    window_facecolor: str = '#ffffff'     # axes (plot-area) background
    figure_facecolor: str = '#ffffff'     # figure (outer) background
    grid_color: str = '#b0b0b0'
    grid_alpha: float = 0.25
    image_plane_color: str = '#666666'
    # --- font ---
    font_color: str = '#1a1a1a'
    font_family: str = 'DejaVu Sans'
    title_size: float = 9.0
    label_size: float = 8.0
    tick_size: float = 7.0

    def color(self, i: int) -> str:
        """Field colour for field index ``i`` (wraps if fewer colours given)."""
        return self.field_colors[i % len(self.field_colors)]

    def wcolor(self, i: int) -> str:
        """Wavelength colour for wavelength index ``i`` (wraps)."""
        return self.wavelength_colors[i % len(self.wavelength_colors)]


# the scheme used when a function is called without an explicit style=
DEFAULT_STYLE = VizStyle()


def _S(style: Optional[VizStyle]) -> VizStyle:
    return DEFAULT_STYLE if style is None else style


def _fig(nrows=1, ncols=1, w=3.4, h=3.0, style: Optional[VizStyle] = None):
    st = _S(style)
    fig, ax = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows))
    fig.patch.set_facecolor(st.figure_facecolor)
    for a in np.atleast_1d(np.array(ax, dtype=object)).ravel():
        _style_axes(a, st)
    return fig, ax


def _style_axes(ax, st: VizStyle):
    """Apply window background, tick/label/spine colours and fonts to one axes."""
    ax.set_facecolor(st.window_facecolor)
    ax.tick_params(colors=st.font_color, labelsize=st.tick_size)
    for spine in ax.spines.values():
        spine.set_color(st.font_color)
    ax.xaxis.label.set_color(st.font_color)
    ax.yaxis.label.set_color(st.font_color)
    ax.xaxis.label.set_fontsize(st.label_size)
    ax.yaxis.label.set_fontsize(st.label_size)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontfamily(st.font_family)
    ax.title.set_color(st.font_color)
    ax.title.set_fontfamily(st.font_family)
    ax.title.set_fontsize(st.title_size)


def _field_labels(optics):
    return [f"{float(a):.0f}\u00b0" for a in optics.fields_deg]


# --------------------------------------------------------------------------- #
#  closed glass-element geometry  (draw a real optics body, not bare curves)
# --------------------------------------------------------------------------- #
def _sag_profile(optics, si: int, R: float, n: int = 160):
    """(z, r) of surface ``si`` sampled to outer radius ``R``.

    Beyond the surface's own clear semi-aperture the sag is held flat at its
    edge value, giving a ground annular edge instead of an extrapolated (and
    possibly divergent) aspheric tail.
    """
    sa = float(optics.semi_aperture[si])
    r = torch.linspace(-R, R, n, dtype=optics.dtype)
    r_clamped = torch.clamp(r, -sa, sa)
    u = r_clamped * r_clamped
    z = asphere_sag(u, optics.curv[si], optics.conic[si], optics.asph[si])
    z_vert = torch.cat([torch.zeros(1, dtype=optics.dtype),
                        torch.cumsum(optics.thick, 0)])
    return (z + z_vert[si]).detach().numpy(), r.detach().numpy()


def _element_spans(optics):
    """List of (i_front, i_back) surface-index pairs bounding each glass element.

    A gap after surface i is glass when ``n_after[i] > 1``; that gap is bounded
    by surfaces i (front) and i+1 (back).
    """
    n_after = optics.n_after.detach().numpy().ravel() if optics.n_after.dim() == 1 \
        else optics.n_after.detach().numpy()[:, 0]
    spans = []
    for i in range(optics.n_surfaces - 1):
        if float(n_after[i]) > 1.0 + 1e-6:
            spans.append((i, i + 1))
    return spans


def _draw_elements(ax, optics, st: VizStyle):
    """Draw each glass element as a filled, closed body with vertical rim edges.

    Returns the outer radius used (max semi-aperture) so the caller can size
    the ray fan and axes to match.
    """
    from matplotlib.patches import Polygon
    spans = _element_spans(optics)
    drawn = set()
    for (i_f, i_b) in spans:
        R = max(float(optics.semi_aperture[i_f]), float(optics.semi_aperture[i_b]))
        zf, rf = _sag_profile(optics, i_f, R)
        zb, rb = _sag_profile(optics, i_b, R)
        # closed polygon: front surface bottom->top, back surface top->bottom
        xs = np.concatenate([zf, zb[::-1]])
        ys = np.concatenate([rf, rb[::-1]])
        # apply alpha to the FILL only (RGBA facecolor) so the rim edge stays
        # fully opaque even when glass_alpha=0 (transparent = outline-only body)
        from matplotlib.colors import to_rgba
        face = to_rgba(st.glass_facecolor, st.glass_alpha)
        poly = Polygon(np.column_stack([xs, ys]), closed=True,
                       facecolor=face, edgecolor=st.surface_edge_color,
                       lw=st.surface_edge_lw, joinstyle='round', zorder=3)
        ax.add_patch(poly)
        drawn.update((i_f, i_b))
    # any surfaces not part of a glass element (e.g. a bare stop) -> thin line
    for si in range(optics.n_surfaces):
        if si in drawn:
            continue
        z, r = _sag_profile(optics, si, float(optics.semi_aperture[si]))
        ax.plot(z, r, color=st.surface_edge_color, lw=st.surface_edge_lw,
                zorder=3)
    return max(float(s) for s in optics.semi_aperture)


def _limiting_radius(optics) -> float:
    """Half-height for the layout ray fan, in mm.

    The fan must never be wider than the light the system actually collects, so
    the entrance-pupil radius (epd/2) is a hard cap. Within that, use the
    aperture-stop clear radius if a stop is flagged, else the smallest clear
    semi-aperture so the fan stays inside every element.

    The cap matters now that semi-apertures are DERIVED from the ray footprint
    (docs/semi_apertures.md): the derived minimum can exceed epd/2 -- on the toy
    it is 3.30 mm against a 2.50 mm pupil radius -- and launching the fan there
    draws rays the stop would have blocked. Those rays miss the rear element at
    wide field and shoot off at absurd angles, which is what put a 69 mm ray in
    the layout figure.
    """
    cap = 0.5 * float(optics.epd)
    r = cap
    for si, is_stop in enumerate(optics.is_stop):
        if is_stop:
            r = float(optics.semi_aperture[si])
            break
    else:
        r = min(float(s) for s in optics.semi_aperture)
    return min(r, cap)


def _entrance_pupil_z(optics) -> float:
    """z of the entrance pupil plane (mm).

    For the object-at-infinity collimated model the bundles are launched at the
    first surface vertex; with a front stop the entrance pupil coincides with
    it, at z = 0 (the tangent plane to the first surface). If the stop is an
    interior surface, use that vertex.
    """
    z_vert = torch.cat([torch.zeros(1, dtype=optics.dtype),
                        torch.cumsum(optics.thick, 0)])
    for si, is_stop in enumerate(optics.is_stop):
        if is_stop:
            return float(z_vert[si])
    return 0.0


def _draw_cross_section(ax, optics, st: VizStyle, rays: str = "chief_marginal",
                        n_fan: int = 9, fields=None, chromatic: bool = False,
                        show_image_plane: bool = True, show_pupil: bool = True):
    """Shared layout renderer: closed elements + traced rays.

    ``rays``:
      * ``"chief_marginal"`` -- the classic chief + two marginal rays per field
        (3 rays), the standard layout convention;
      * ``"fan"`` -- a denser ``n_fan``-ray meridional fan.

    ``chromatic``: when True, trace each wavelength and colour rays by
    wavelength (paper convention); otherwise trace the d-line and colour by
    field. Rays are launched across the limiting clear aperture so they stay
    inside the glass, and each ray is drawn on its OWN per-surface intersection
    z, so bends sit exactly on the surface curves.
    """
    R_out = _draw_elements(ax, optics, st)
    rp = _limiting_radius(optics)
    z_img = float(torch.cumsum(optics.thick, 0)[-1])
    nf = 3 if rays == "chief_marginal" else int(n_fan)
    if show_image_plane:
        ax.axvline(z_img, color=st.image_plane_color, ls='--', lw=1.0, zorder=1)
    # entrance pupil plane (a short vertical marker at the stop vertex)
    if show_pupil:
        z_ep = _entrance_pupil_z(optics)
        ax.plot([z_ep, z_ep], [-rp, rp], color=st.pupil_color, lw=1.2,
                ls=(0, (4, 2)), zorder=1)
        for s in (-1, 1):
            ax.plot([z_ep], [s * rp], marker='_', color=st.pupil_color,
                    ms=7, zorder=1)
    if fields is None:
        fields = range(optics.fields_deg.numel())
    # dispersion is meaningful only with >1 wavelength
    W = optics.wavelengths_um.numel()
    w_indices = range(W) if (chromatic and W > 1) else [0]
    for fi in fields:
        for wi in w_indices:
            col = st.wcolor(wi) if (chromatic and W > 1) else st.color(fi)
            zs, ys = optics.ray_paths(field_index=fi, wavelength_index=wi,
                                    n_fan=nf, pupil_radius=rp)
            for j in range(ys.shape[0]):
                ax.plot(zs[j].numpy(), ys[j].numpy(), color=col, lw=st.ray_lw,
                        alpha=st.ray_alpha, zorder=2)
    ax.set_aspect('equal', adjustable='datalim')
    return z_img, R_out


# --------------------------------------------------------------------------- #
#  single-state diagnostics
# --------------------------------------------------------------------------- #
def _layout_legend(ax, optics, st, chromatic):
    """Legend keyed by wavelength (chromatic) or field (monochromatic)."""
    if chromatic and optics.wavelengths_um.numel() > 1:
        labels = [f"{float(w)*1000:.0f} nm" for w in optics.wavelengths_um]
        handles = [plt.Line2D([], [], color=st.wcolor(i), label=l)
                   for i, l in enumerate(labels)]
        title = "wavelength"
    else:
        handles = [plt.Line2D([], [], color=st.color(i), label=l)
                   for i, l in enumerate(_field_labels(optics))]
        title = "field"
    leg = ax.legend(handles=handles, fontsize=st.tick_size, title=title,
                    loc='upper left', framealpha=0.9)
    leg.get_frame().set_facecolor(st.window_facecolor)
    leg.get_frame().set_edgecolor(st.font_color)
    leg.get_title().set_color(st.font_color)
    for txt in leg.get_texts():
        txt.set_color(st.font_color)


def _focus_inset(ax, optics, st, field_index, rays, n_fan):
    """Magnified inset on the focus region so per-wavelength (chromatic) ray
    separation is visible even when it is sub-pixel on the full-scale axes.

    Traces every wavelength's ray fan, finds the focus window (tight box around
    where the marginal rays converge just ahead of the image plane), and redraws
    the same rays there. Returns the inset axes (or None if <2 wavelengths).
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    W = optics.wavelengths_um.numel()
    if W < 2:
        return None
    z_img = float(torch.cumsum(optics.thick, 0)[-1])
    rp = _limiting_radius(optics)
    nf = 3 if rays == "chief_marginal" else int(n_fan)
    # collect each ray's final straight segment (exit of last surface -> image)
    # per wavelength; between the last surface and the image plane rays are
    # straight, so we can resample them at any z to find best focus.
    segs = []  # (wi, z_exit(nf,), y_exit(nf,), slope(nf,))
    for wi in range(W):
        zs, ys = optics.ray_paths(field_index=field_index, wavelength_index=wi,
                                n_fan=nf, pupil_radius=rp)
        z0 = zs[:, -2].numpy(); z1 = zs[:, -1].numpy()
        y0 = ys[:, -2].numpy(); y1 = ys[:, -1].numpy()
        dz = np.where(np.abs(z1 - z0) < 1e-9, 1e-9, z1 - z0)
        segs.append((wi, z0, y0, (y1 - y0) / dz))

    def ray_y(z):  # (W, nf) ray heights at plane z
        return np.stack([y0 + m * (z - ze) for (_, ze, y0, m) in segs])

    # best focus: z minimising pooled ray-height spread near the image region
    z_exit_max = max(float(s[1].max()) for s in segs)
    zg = np.linspace(z_exit_max + 1e-3, z_img * 1.02, 400)
    spread = np.array([ray_y(z).std() for z in zg])
    z_focus = float(zg[int(spread.argmin())])
    # window: tight in z around focus (covers axial color); y around the focal
    # spot, floored so a diffraction-limited toy never collapses to a line
    yf = ray_y(z_focus)
    y_c = float(yf.mean())
    # axial-color extent: per-wavelength focus spread
    zfoc_w = []
    for wi in range(W):
        sp_w = np.array([np.ptp(y0 + m * (z - ze))
                         for z in zg
                         for (w2, ze, y0, m) in [segs[wi]]])
        zfoc_w.append(float(zg[int(sp_w.argmin())]))
    z_ax = max(np.ptp(zfoc_w), 0.05)
    half_z = max(2.5 * z_ax, 0.25)
    half_y = max(float(np.abs(yf - y_c).max()) * 2.0, half_z * 0.12, 0.01)
    axin = inset_axes(ax, width="40%", height="40%", loc='lower right',
                      borderpad=1.0)
    axin.axvline(z_img, color=st.image_plane_color, ls='--', lw=0.8, zorder=1)
    zdraw = np.linspace(z_focus - half_z, z_img, 40)
    for (wi, ze, y0, m) in segs:
        col = st.wcolor(wi)
        yy = y0[:, None] + m[:, None] * (zdraw[None, :] - ze[:, None])
        for j in range(yy.shape[0]):
            axin.plot(zdraw, yy[j], color=col, lw=st.ray_lw + 0.3,
                      alpha=0.95, zorder=2)
    axin.set_xlim(z_focus - half_z, z_focus + half_z)
    axin.set_ylim(y_c - half_y, y_c + half_y)
    axin.set_title("focus (zoom)", fontsize=st.tick_size, color=st.font_color,
                   fontfamily=st.font_family)
    axin.tick_params(labelsize=st.tick_size * 0.8, colors=st.font_color)
    axin.set_facecolor(st.window_facecolor)
    for sp_ in axin.spines.values():
        sp_.set_edgecolor(st.font_color)
    mark_inset(ax, axin, loc1=2, loc2=4, fc="none",
               ec=st.font_color, lw=0.7, alpha=0.4)
    return axin


def _clamp_layout_view(ax, R_out, st, keep_factor: float = 1.6):
    """Clamp a cross-section's y-view to the element extent, annotating clipping.

    A badly-corrected design can send an off-axis ray far off the axis. Those
    are real traced rays, but autoscaling to them shrinks the glass to a few
    pixels and the figure stops showing the optics. Keep the glass plus a
    margin, and state in the axes how far the rays actually go so the clipping
    is never mistaken for the true ray extent.

    This is a guard, not a routine path: with the fan capped at the entrance
    pupil (``_limiting_radius``) and missed elements truncated by
    ``ray_paths``, neither toy triggers it. It fired constantly before those
    two fixes, on rays that were themselves artefacts.
    """
    y_lo, y_hi = ax.get_ylim()
    reach = max(abs(y_lo), abs(y_hi))
    keep = float(R_out) * keep_factor
    if reach <= keep:
        return False
    # 'datalim' (set by _draw_cross_section) lets matplotlib silently discard
    # these limits to satisfy the equal aspect -- the annotation would then
    # print a half-width the panel does not actually show. 'box' honours the
    # limits and reshapes the axes instead.
    ax.set_aspect('equal', adjustable='box')
    ax.set_ylim(-keep, keep)
    ax.text(0.99, 0.02,
            f"view clipped to \u00b1{keep:.1f} mm; rays reach \u00b1{reach:.0f} mm",
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=st.tick_size - 1, color="0.45", fontfamily=st.font_family)
    return True


def plot_layout(optics, field_index: Optional[int] = None,
                     rays: str = "chief_marginal", n_fan: int = 9,
                     chromatic: bool = False, title: str = "Optical layout",
                     focus_inset: Optional[bool] = None,
                     y_view: str = "elements",
                     style: Optional[VizStyle] = None):
    """Cross-section drawing: closed glass elements + traced rays.

    Each refractive element is drawn as a closed body (front and back surfaces
    joined at a ground rim). By default the fill is transparent (outline only);
    set ``style=VizStyle(glass_alpha=...)`` for a tinted fill.

    Rays: ``rays="chief_marginal"`` (default) draws the classic chief + two
    marginal rays per field; ``rays="fan"`` draws a denser ``n_fan`` fan. Rays
    launch from the entrance pupil across the limiting clear aperture, so they
    stay inside the glass and each bend sits on the true surface curve.

    ``y_view="elements"`` (default) clamps the vertical view to the glass extent
    so a wildly-diverging ray in an uncorrected design cannot squash the elements
    out of visibility; the clipping is annotated in the axes. Pass
    ``y_view="rays"`` to autoscale to every traced ray instead.

    ``chromatic=True`` traces every wavelength and colours rays by wavelength
    (paper convention, blue->green->red); otherwise the d-line is traced and
    rays are coloured by field. Pass ``style=VizStyle(...)`` to restyle
    surfaces / field lines / window / font.
    """
    st = _S(style)
    fig, ax = _fig(w=6.4, h=3.4, style=st)
    fields = [field_index] if field_index is not None else None
    z_img, R_out = _draw_cross_section(ax, optics, st, rays=rays, n_fan=n_fan,
                                       fields=fields, chromatic=chromatic)
    ax.text(z_img, R_out * 1.05, "image", ha='center', va='bottom',
            fontsize=st.tick_size, color=st.image_plane_color,
            fontfamily=st.font_family)
    ax.set_xlabel("z  (mm)"); ax.set_ylabel("y  (mm)")
    ax.set_title(title)
    ax.grid(True, color=st.grid_color, alpha=st.grid_alpha, zorder=0)

    if y_view == "elements":
        _clamp_layout_view(ax, R_out, st)
    multi_field = (field_index is None and optics.fields_deg.numel() > 1)
    if multi_field or (chromatic and optics.wavelengths_um.numel() > 1):
        _layout_legend(ax, optics, st, chromatic)
    # focus-region zoom: default on when chromatic (color split is often
    # sub-pixel on the full-scale axes) and a single field is shown
    want_inset = focus_inset
    if want_inset is None:
        want_inset = bool(chromatic and optics.wavelengths_um.numel() > 1
                          and field_index is not None)
    if want_inset:
        fi = field_index if field_index is not None else 0
        _focus_inset(ax, optics, st, fi, rays, n_fan)
    fig.tight_layout()
    return fig


def plot_spot_diagram(optics, wavelength_index: int = 0, title: str = "Spot diagram",
                      view_um: Optional[float] = None, style: Optional[VizStyle] = None):
    """Per-field spot scatter (microns, centroid-referenced)."""
    st = _S(style)
    sp = optics.forward()
    F = sp.xy.shape[0]
    cent = sp.centroids()
    fig, axs = _fig(1, F, w=2.6, h=2.8, style=st)
    if F == 1:
        axs = [axs]
    rms = sp.rms_radius()
    for fi in range(F):
        ax = axs[fi]
        m = sp.valid[fi, wavelength_index]
        pts = (sp.xy[fi, wavelength_index][m] - cent[fi]) * 1000.0   # um
        pts = pts.detach().numpy()
        ax.scatter(pts[:, 0], pts[:, 1], s=3, color=st.color(fi),
                   alpha=0.6, edgecolors='none')
        ax.set_title(f"{_field_labels(optics)[fi]}   RMS {float(rms[fi])*1000:.1f} \u00b5m")
        ax.set_aspect('equal')
        ax.axhline(0, color=st.grid_color, lw=0.5); ax.axvline(0, color=st.grid_color, lw=0.5)
        if view_um:
            ax.set_xlim(-view_um, view_um); ax.set_ylim(-view_um, view_um)
        ax.set_xlabel("x (\u00b5m)")
        if fi == 0:
            ax.set_ylabel("y (\u00b5m)")
    fig.suptitle(title, fontsize=st.title_size, color=st.font_color,
                 fontfamily=st.font_family)
    fig.tight_layout()
    return fig


def plot_psf(optics, bridge, title: str = "Geometric PSF (KDE)",
             cmap: str = 'inferno', style: Optional[VizStyle] = None):
    """Per-field PSF grid, built with the bridge's KDE settings.

    ``bridge.psf`` returns only the single configured field; here we render the
    PSF of *every* field, reusing the bridge's grid/pitch/sigma so what you see
    matches what the pipeline convolves with.
    """
    from .bridge.kde_psf import kde_psf, offgrid_fraction
    st = _S(style)
    sp = optics.forward()
    psf = kde_psf(sp, bridge.grid_size, bridge.pixel_pitch_mm,
                  bridge.sigma_bins, per_field=True)   # (F,W,G,G)
    # A too-small grid still returns a normalized, plausible-looking PSF -- it is
    # the CLIPPED spot. Report the clipping rather than hide it.
    off = offgrid_fraction(sp, bridge.grid_size, bridge.pixel_pitch_mm)
    half_um = (bridge.grid_size - 1) / 2.0 * bridge.pixel_pitch_mm * 1000.0
    F = psf.shape[0]
    fig, axs = _fig(1, F, w=2.4, h=2.6, style=st)
    if F == 1:
        axs = [axs]
    for fi in range(F):
        ax = axs[fi]
        p = psf[fi, 0].detach().numpy()
        ax.imshow(p, cmap=cmap)
        fo = float(off[fi]) * 100.0
        ax.set_title(_field_labels(optics)[fi] +
                     (f"  ({fo:.0f}% clipped)" if fo > 1.0 else ""),
                     color="#b2182b" if fo > 5.0 else st.font_color)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{title}\n"
                 f"grid {bridge.grid_size}\u00d7{bridge.grid_size} @ "
                 f"{bridge.pixel_pitch_mm*1000:.1f} \u00b5m/bin "
                 f"(\u00b1{half_um:.0f} \u00b5m)",
                 fontsize=st.title_size, color=st.font_color,
                 fontfamily=st.font_family)
    fig.tight_layout()
    return fig


def plot_convergence(history, ylabel: str = r"$\frac{1}{2}\|\ell\|^2$",
                     title: str = "Convergence", logy: bool = True,
                     color: Optional[str] = None, style: Optional[VizStyle] = None):
    """Loss-vs-iteration curve. ``history`` is a list/array of scalar losses."""
    st = _S(style)
    curve_color = st.color(0) if color is None else color
    h = [float(x) for x in history]
    fig, ax = _fig(w=4.2, h=3.0, style=st)
    (ax.semilogy if logy else ax.plot)(range(len(h)), h, color=curve_color, lw=1.5,
                                       marker='o', ms=2.5)
    ax.set_xlabel("iteration"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, which='both', color=st.grid_color, alpha=st.grid_alpha)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
#  before / after comparison  (paper Fig. 4 style)
# --------------------------------------------------------------------------- #
def compare_designs(optics, theta_before, theta_after,
                   labels=("start", "optimized"), rays: str = "chief_marginal",
                   n_fan: int = 9, chromatic: bool = False,
                   view_um: Optional[float] = None, style: Optional[VizStyle] = None):
    """2x2: layout (before / after) over spot diagrams (before / after).

    Uses the same closed-element / entrance-pupil ray renderer as
    ``plot_layout`` (chief+marginal rays by default; ``chromatic=True``
    for per-wavelength colouring). Restores ``optics`` to its incoming theta on
    exit, so it is side-effect free.
    """
    st = _S(style)
    theta_saved = optics.get_theta().clone()
    fig, axs = plt.subplots(2, 2, figsize=(9.0, 6.4))
    fig.patch.set_facecolor(st.figure_facecolor)
    for a in axs.ravel():
        _style_axes(a, st)
    for col, (th, lab) in enumerate(zip((theta_before, theta_after), labels)):
        optics.set_theta(th)
        # --- layout (top row) ---
        axL = axs[0, col]
        _, R_out = _draw_cross_section(axL, optics, st, rays=rays, n_fan=n_fan,
                                       chromatic=chromatic)
        # Without this the starting design's diverging 30-deg ray sets the scale
        # and both elements collapse to a few pixels -- the panel would show the
        # ray excursion instead of the optics being compared.
        _clamp_layout_view(axL, R_out, st)
        axL.set_title(f"{lab} \u2014 layout")
        axL.set_xlabel("z (mm)"); axL.set_ylabel("y (mm)")
        # --- worst-field spot (bottom row) ---
        axS = axs[1, col]
        sp = optics.forward(); cent = sp.centroids(); rms = sp.rms_radius()
        fi = int(torch.argmax(rms))
        m = sp.valid[fi, 0]
        pts = ((sp.xy[fi, 0][m] - cent[fi]) * 1000.0).detach().numpy()
        axS.scatter(pts[:, 0], pts[:, 1], s=4,
                    color=st.color(fi), alpha=0.6, edgecolors='none')
        axS.set_aspect('equal')
        # The panel draws the WORST field, so it must be labelled with that
        # field's own RMS. ESR is the mean over fields, so it necessarily sits
        # below the worst field's RMS -- printing it here would understate the
        # spot the reader is looking at (by ~1.6x on the starting toy).
        esr = float(sp.effective_spot_radius()) * 1000
        rms_fi = float(rms[fi]) * 1000
        axS.set_title(f"{lab} \u2014 worst field {_field_labels(optics)[fi]}"
                      f"  (RMS {rms_fi:.1f} \u00b5m)\n"
                      f"design-wide ESR {esr:.1f} \u00b5m (mean over fields)",
                      fontsize=st.title_size - 2)
        axS.set_xlabel("x (\u00b5m)"); axS.set_ylabel("y (\u00b5m)")
        # Default autoscales per panel and PRINTS the box half-width, because a
        # shared box cannot show a 420 um starting spot and a 20 um converged one
        # at once. Pass view_um for a genuinely shared box.
        v = view_um if view_um else float(np.abs(pts).max()) * 1.15
        axS.set_xlim(-v, v); axS.set_ylim(-v, v)
        axS.text(0.96, 0.035, f"\u00b1{v:.0f} \u00b5m", transform=axS.transAxes,
                 ha="right", va="bottom", fontsize=st.label_size - 2,
                 color="0.45", fontfamily=st.font_family)
    optics.set_theta(theta_saved)
    fig.tight_layout()
    return fig


def plot_restoration(scene, blurred, restored, cmap: str = 'gray',
                     style: Optional[VizStyle] = None):
    """Sharp / blurred / restored triptych with PSNR annotations."""
    st = _S(style)
    def _psnr(a, b):
        with torch.no_grad():
            return 10.0 * np.log10(1.0 / max(float(((a - b) ** 2).mean()), 1e-12))
    fig, axs = _fig(1, 3, w=2.5, h=2.7, style=st)
    imgs = [scene, blurred, restored]
    titles = ["sharp (target)",
              f"blurred\nPSNR {_psnr(blurred, scene):.1f} dB",
              f"restored\nPSNR {_psnr(restored, scene):.1f} dB"]
    for ax, im, t in zip(axs, imgs, titles):
        ax.imshow(im.squeeze().detach().numpy(), cmap=cmap, vmin=0, vmax=1)
        ax.set_title(t); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
#  progressive / evolution view  (question 3)
# --------------------------------------------------------------------------- #
class OptimizationRecorder:
    """Snapshots the parameter vector (and loss) each LM iteration.

    Pass an instance as the ``callback`` to ``LevenbergMarquardt.run`` (the LM
    engine calls ``callback(state)`` after every accepted step). Afterwards, the
    recorded thetas can be replayed through the optics to visualize how the spot /
    layout evolved -- without re-running the optimization.

    Usage
    -----
        rec = OptimizationRecorder(optics)
        lm.run(40, callback=rec)
        viz.plot_spot_evolution(optics, rec)
    """
    def __init__(self, optics, stride: int = 1):
        self.optics = optics
        self.stride = int(stride)
        self.thetas: List[torch.Tensor] = [optics.get_theta().clone()]
        self.losses: List[float] = []
        self._i = 0

    def __call__(self, state):
        # state is expected to expose .theta and .loss (LMState); fall back
        # gracefully if a bare tensor / float is passed.
        theta = getattr(state, "theta", state)
        loss = getattr(state, "loss", None)
        self._i += 1
        if self._i % self.stride == 0:
            self.thetas.append(theta.detach().clone())
            if loss is not None:
                self.losses.append(float(loss))

    @property
    def n_frames(self):
        return len(self.thetas)


def _spot_at(optics, theta, wavelength_index=0, field_index=None):
    """Spot points (um, centroid-referenced) at one design state.

    ``field_index=None`` picks the currently worst field, which is what a
    single-panel summary wants. Pass an explicit index to follow ONE fixed field
    across iterations: the worst field is not the same field throughout an
    optimization run, so a sequence of worst-field panels silently changes
    subject partway through and cannot be read as one field improving.

    Returns ``(pts, esr_um, field_index, rms_um)`` where ``esr_um`` is the
    whole-design effective spot radius and ``rms_um`` is the RMS radius of the
    field actually plotted.
    """
    saved = optics.get_theta().clone()
    optics.set_theta(theta)
    sp = optics.forward()
    esr = float(sp.effective_spot_radius()) * 1000
    rms_all = sp.rms_radius()
    fi = int(torch.argmax(rms_all)) if field_index is None else int(field_index)
    m = sp.valid[fi, wavelength_index]
    pts = ((sp.xy[fi, wavelength_index][m] - sp.centroids()[fi]) * 1000.0).detach().numpy()
    rms = float(rms_all[fi]) * 1000
    optics.set_theta(saved)
    return pts, esr, fi, rms


def plot_spot_evolution(optics, recorder, n_show: int = 6, view_um: Optional[float] = None,
                        fields: Optional[Sequence[int]] = None,
                        share_view: bool = False,
                        style: Optional[VizStyle] = None):
    """Spot-diagram evolution as a fields x iterations grid.

    One ROW per field angle, one COLUMN per sampled iteration, so each row is one
    fixed field improving and can be read straight across. Previously this drew a
    single row of *worst-field* spots, which changes which field it shows partway
    through a run (on the toy: field 2 at iteration 0, field 1 from iteration 3
    on), making the panels look like a discontinuous jump that is really a change
    of subject.

    Parameters
    ----------
    n_show     : number of iterations to sample across the recording
    fields     : field indices to show (default: all)
    view_um    : half-width of the plotted box in um, applied to every panel.
                 Default None autoscales per panel (see ``share_view``).
    share_view : if True, EVERY panel uses one box sized to the largest spot in
                 the grid. Honest but usually unreadable: a converging run shrinks
                 the spot 10-100x, so the converged panels collapse to a single
                 dot. Default False autoscales each panel and prints its box
                 half-width, so structure stays visible and the scale is stated.
    """
    st = _S(style)
    idx = np.linspace(0, recorder.n_frames - 1, min(n_show, recorder.n_frames))
    idx = sorted(set(int(round(i)) for i in idx))
    F = optics.fields_deg.numel()
    fld = list(range(F)) if fields is None else [int(f) for f in fields]

    # collect first so the view boxes can be derived from the real extents
    data = {}
    for fi in fld:
        for k in idx:
            data[(fi, k)] = _spot_at(optics, recorder.thetas[k], field_index=fi)

    # A converging run shrinks the spot by one to two orders of magnitude, so a
    # single box across a row renders the converged panels as one dot. Each panel
    # therefore gets its own box (spot STRUCTURE stays visible throughout) and the
    # box half-width is printed under the panel, so the shrinkage is read from the
    # numbers rather than being faked by a shared axis. Pass an explicit
    # ``view_um`` (or share_view=True) for a genuinely fixed box when the run
    # spans a narrow range.
    if view_um is not None:
        box = {key: float(view_um) for key in data}
    elif share_view:
        v = max(float(np.abs(d[0]).max()) for d in data.values()) * 1.15 + 1.0
        box = {key: v for key in data}
    else:
        box = {key: float(np.abs(d[0]).max()) * 1.25 + 0.5 for key, d in data.items()}

    fig, axs = _fig(len(fld), len(idx), w=1.95, h=2.35, style=st)
    axs = np.atleast_2d(axs).reshape(len(fld), len(idx))
    for row, fi in enumerate(fld):
        for col, k in enumerate(idx):
            ax = axs[row, col]
            pts, esr, _, rms = data[(fi, k)]
            ax.scatter(pts[:, 0], pts[:, 1], s=3, color=st.color(fi),
                       alpha=0.6, edgecolors='none')
            v = box[(fi, k)]
            ax.set_xlim(-v, v); ax.set_ylim(-v, v)
            ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("0.6")
            # per-panel RMS (this field, this iteration) inside the axes so it
            # cannot collide with the row below; box half-width underneath, since
            # the box is no longer constant
            # both annotations live INSIDE the axes: an outside label would
            # collide with the row above once every panel has its own box label
            ax.text(0.04, 0.965, f"RMS {rms:.1f} \u00b5m", transform=ax.transAxes,
                    ha="left", va="top", fontsize=st.label_size - 1,
                    color=st.color(fi), fontfamily=st.font_family)
            ax.text(0.96, 0.035, f"\u00b1{v:.0f} \u00b5m", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=st.label_size - 2,
                    color="0.45", fontfamily=st.font_family)
            if row == 0:
                ax.text(0.5, 1.22, f"iteration {k}", transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=st.label_size + 1,
                        color=st.font_color, fontfamily=st.font_family)
                ax.text(0.5, 1.135, f"ESR {esr:.0f} \u00b5m", transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=st.label_size - 1,
                        color="0.45", fontfamily=st.font_family)
            if col == 0:
                ax.text(-0.16, 0.5, f"{float(optics.fields_deg[fi]):.0f}\u00b0 field",
                        transform=ax.transAxes, ha="right", va="center",
                        rotation=90, fontsize=st.label_size + 1,
                        color=st.color(fi), fontfamily=st.font_family)

    fig.suptitle("Spot evolution during optimization \u2014 one row per field "
                 "(each panel autoscaled; box half-width printed)",
                 fontsize=st.title_size, color=st.font_color,
                 fontfamily=st.font_family)
    fig.tight_layout(rect=(0, 0.01, 1, 0.93 if len(fld) > 1 else 0.86))
    return fig


def plot_field_convergence(optics, recorder, fields: Optional[Sequence[int]] = None,
                           style: Optional[VizStyle] = None):
    """Per-field RMS spot radius vs LM iteration, log scale.

    The quantitative companion to :func:`plot_spot_evolution`. The design-wide
    effective spot radius is the MEAN of the per-field RMS radii
    (``rms_radius().mean()``), so it sits between the best and worst field and
    hides which field is actually limiting the design -- and which field that is
    changes during the run. Dotted lines mark those handovers; the crossover is
    visible as two curves swapping order.
    """
    st = _S(style)
    F = optics.fields_deg.numel()
    fld = list(range(F)) if fields is None else [int(f) for f in fields]
    saved = optics.get_theta().clone()
    iters = list(range(recorder.n_frames))
    rms = np.zeros((len(fld), len(iters)))
    esr = np.zeros(len(iters))
    worst = np.zeros(len(iters), dtype=int)
    for j, k in enumerate(iters):
        optics.set_theta(recorder.thetas[k])
        sp = optics.forward()
        r = sp.rms_radius()
        for i, fi in enumerate(fld):
            rms[i, j] = float(r[fi]) * 1000
        esr[j] = float(sp.effective_spot_radius()) * 1000
        worst[j] = int(torch.argmax(r))
    optics.set_theta(saved)

    fig, ax = _fig(w=6.4, h=4.0, style=st)
    ax.plot(iters, esr, color="0.55", lw=2.6, ls="--", zorder=1,
            label="ESR (design-wide = mean over fields)")
    for i, fi in enumerate(fld):
        ax.plot(iters, rms[i], color=st.color(fi), lw=1.9, zorder=2,
                label=f"{float(optics.fields_deg[fi]):.0f}\u00b0 field")
    # mark where the limiting field changes hands
    sw = [k for k in iters[1:] if worst[k] != worst[k - 1]]
    for k in sw:
        ax.axvline(k, color="#b2182b", lw=0.9, ls=":", zorder=0)
    if sw:
        lab = (f"limiting field changes hands (it {sw[0]}"
               + (f"\u2013{sw[-1]}, {len(sw)}\u00d7)" if len(sw) > 1 else ")"))
        ax.annotate(lab, xy=(sw[0], rms.max()), xycoords="data",
                    xytext=(8, -14), textcoords="offset points",
                    ha="left", va="top", fontsize=st.label_size - 1,
                    color="#b2182b", fontfamily=st.font_family)
    ax.set_yscale("log")
    ax.set_xlabel("LM iteration", fontsize=st.label_size, color=st.font_color)
    ax.set_ylabel("RMS spot radius (\u00b5m)", fontsize=st.label_size,
                  color=st.font_color)
    ax.set_title("Per-field convergence", fontsize=st.title_size,
                 color=st.font_color, fontfamily=st.font_family)
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(fontsize=st.label_size - 1, frameon=False)
    fig.tight_layout()
    return fig


def animate_spot_evolution(optics, recorder, path: str = "spot_evolution.gif",
                           fps: int = 6, view_um: Optional[float] = None,
                           fields: Optional[Sequence[int]] = None,
                           style: Optional[VizStyle] = None):
    """Write a GIF of the spot shrinking over iterations, one panel per field.

    Every panel shares one fixed view box derived from the whole recording, so
    the shrinkage is actually visible rather than rescaled away frame by frame.
    ``fields`` selects which field angles to animate (default: all).

    Requires pillow (already a dependency). Returns the output path.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    st = _S(style)
    F = optics.fields_deg.numel()
    fld = list(range(F)) if fields is None else [int(f) for f in fields]
    # One panel per field, all sharing a FIXED box taken from the largest spot in
    # the whole recording. A per-frame box would rescale away the very shrinkage
    # the animation exists to show, and a single worst-field panel would swap
    # which field it displays mid-run.
    if view_um is None:
        view_um = 1.0 + 1.15 * max(
            float(np.abs(_spot_at(optics, recorder.thetas[k], field_index=fi)[0]).max())
            for fi in fld for k in (0, recorder.n_frames - 1))
    fig, axs = _fig(1, len(fld), w=2.7, h=2.9, style=st)
    axs = np.atleast_1d(axs).ravel()
    scs, ttls = [], []
    for ax, fi in zip(axs, fld):
        sc = ax.scatter([], [], s=4, color=st.color(fi), alpha=0.6, edgecolors='none')
        ax.set_xlim(-view_um, view_um); ax.set_ylim(-view_um, view_um)
        ax.set_aspect('equal'); ax.set_xlabel("x (\u00b5m)")
        if fi == fld[0]:
            ax.set_ylabel("y (\u00b5m)")
        ttls.append(ax.set_title(""))
        ax.text(0.02, 0.97, f"{float(optics.fields_deg[fi]):.0f}\u00b0",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=st.label_size, color=st.color(fi),
                fontfamily=st.font_family)
        scs.append(sc)
    sup = fig.suptitle("", fontsize=st.title_size, color=st.font_color,
                       fontfamily=st.font_family)
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    def _update(k):
        for sc, ttl, fi in zip(scs, ttls, fld):
            pts, esr, _, rms = _spot_at(optics, recorder.thetas[k], field_index=fi)
            sc.set_offsets(pts if len(pts) else np.zeros((0, 2)))
            ttl.set_text(f"RMS {rms:.1f} \u00b5m")
        sup.set_text(f"iteration {k}    ESR {esr:.1f} \u00b5m")
        return (*scs, *ttls, sup)

    anim = FuncAnimation(fig, _update, frames=recorder.n_frames, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


# --- Backward-compatible aliases (see optics/__init__.py) -----------------
plot_lens_layout = plot_layout
compare_lenses = compare_designs
