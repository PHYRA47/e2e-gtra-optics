"""Diagnostic and comparison plots for the e2e_optics pipeline.

This is the reusable visualization layer -- every figure produced during the
build lives here as a function you can call on any lens / bridge / result, so
the plots are part of the code base rather than throwaway script cells.

Two families:

*Single-state diagnostics* (one lens):
    plot_lens_layout(lens)          -- ray-fan layout + surface profiles
    plot_spot_diagram(lens)         -- per-field spot scatter
    plot_psf(lens, bridge)          -- geometric PSF grids
    plot_convergence(history)       -- LM / task-loss curves

*Before / after comparison* (paper Fig. 4 style):
    compare_lenses(lens, theta_a, theta_b, ...)   -- side-by-side layout+spots
    plot_restoration(scene, blurred, restored)    -- capture vs restored triptych

*Progressive / evolution view* (question 3):
    OptimizationRecorder    -- callback that snapshots theta each LM iteration
    plot_spot_evolution(lens, recorder)           -- spot vs iteration grid
    animate_spot_evolution(lens, recorder, path)  -- GIF of the spot shrinking

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
        viz.plot_lens_layout(lens, style=dark)

    Groups map onto the four things you asked to control:
      * **surfaces**  -> glass_* / surface_edge_*
      * **field lines** -> field_colors / ray_*
      * **window / background** -> window_facecolor / figure_facecolor / grid_*
      * **font** -> font_color / font_family / *_size
    """
    # --- surfaces (the glass elements) ---
    glass_facecolor: str = '#bfe0f5'      # fill of a closed lens element
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


def _field_labels(lens):
    return [f"{float(a):.0f}\u00b0" for a in lens.fields_deg]


# --------------------------------------------------------------------------- #
#  closed glass-element geometry  (draw a real lens body, not bare curves)
# --------------------------------------------------------------------------- #
def _sag_profile(lens, si: int, R: float, n: int = 160):
    """(z, r) of surface ``si`` sampled to outer radius ``R``.

    Beyond the surface's own clear semi-aperture the sag is held flat at its
    edge value, giving a ground annular edge instead of an extrapolated (and
    possibly divergent) aspheric tail.
    """
    sa = float(lens.semi_aperture[si])
    r = torch.linspace(-R, R, n, dtype=lens.dtype)
    r_clamped = torch.clamp(r, -sa, sa)
    u = r_clamped * r_clamped
    z = asphere_sag(u, lens.curv[si], lens.conic[si], lens.asph[si])
    z_vert = torch.cat([torch.zeros(1, dtype=lens.dtype),
                        torch.cumsum(lens.thick, 0)])
    return (z + z_vert[si]).detach().numpy(), r.detach().numpy()


def _glass_spans(lens):
    """List of (i_front, i_back) surface-index pairs bounding each glass element.

    A gap after surface i is glass when ``n_after[i] > 1``; that gap is bounded
    by surfaces i (front) and i+1 (back).
    """
    n_after = lens.n_after.detach().numpy().ravel() if lens.n_after.dim() == 1 \
        else lens.n_after.detach().numpy()[:, 0]
    spans = []
    for i in range(lens.n_surfaces - 1):
        if float(n_after[i]) > 1.0 + 1e-6:
            spans.append((i, i + 1))
    return spans


def _draw_elements(ax, lens, st: VizStyle):
    """Draw each glass element as a filled, closed body with vertical rim edges.

    Returns the outer radius used (max semi-aperture) so the caller can size
    the ray fan and axes to match.
    """
    from matplotlib.patches import Polygon
    spans = _glass_spans(lens)
    drawn = set()
    for (i_f, i_b) in spans:
        R = max(float(lens.semi_aperture[i_f]), float(lens.semi_aperture[i_b]))
        zf, rf = _sag_profile(lens, i_f, R)
        zb, rb = _sag_profile(lens, i_b, R)
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
    for si in range(lens.n_surfaces):
        if si in drawn:
            continue
        z, r = _sag_profile(lens, si, float(lens.semi_aperture[si]))
        ax.plot(z, r, color=st.surface_edge_color, lw=st.surface_edge_lw,
                zorder=3)
    return max(float(s) for s in lens.semi_aperture)


def _limiting_radius(lens) -> float:
    """Half-height for the layout ray fan: the aperture-stop clear radius if a
    stop is flagged, else the smallest clear semi-aperture (so the fan stays
    inside every element)."""
    for si, is_stop in enumerate(lens.is_stop):
        if is_stop:
            return float(lens.semi_aperture[si])
    return min(float(s) for s in lens.semi_aperture)


def _entrance_pupil_z(lens) -> float:
    """z of the entrance pupil plane (mm).

    For the object-at-infinity collimated model the bundles are launched at the
    first surface vertex; with a front stop the entrance pupil coincides with
    it, at z = 0 (the tangent plane to the first surface). If the stop is an
    interior surface, use that vertex.
    """
    z_vert = torch.cat([torch.zeros(1, dtype=lens.dtype),
                        torch.cumsum(lens.thick, 0)])
    for si, is_stop in enumerate(lens.is_stop):
        if is_stop:
            return float(z_vert[si])
    return 0.0


def _draw_cross_section(ax, lens, st: VizStyle, rays: str = "chief_marginal",
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
    R_out = _draw_elements(ax, lens, st)
    rp = _limiting_radius(lens)
    z_img = float(torch.cumsum(lens.thick, 0)[-1])
    nf = 3 if rays == "chief_marginal" else int(n_fan)
    if show_image_plane:
        ax.axvline(z_img, color=st.image_plane_color, ls='--', lw=1.0, zorder=1)
    # entrance pupil plane (a short vertical marker at the stop vertex)
    if show_pupil:
        z_ep = _entrance_pupil_z(lens)
        ax.plot([z_ep, z_ep], [-rp, rp], color=st.pupil_color, lw=1.2,
                ls=(0, (4, 2)), zorder=1)
        for s in (-1, 1):
            ax.plot([z_ep], [s * rp], marker='_', color=st.pupil_color,
                    ms=7, zorder=1)
    if fields is None:
        fields = range(lens.fields_deg.numel())
    # dispersion is meaningful only with >1 wavelength
    W = lens.wavelengths_um.numel()
    w_indices = range(W) if (chromatic and W > 1) else [0]
    for fi in fields:
        for wi in w_indices:
            col = st.wcolor(wi) if (chromatic and W > 1) else st.color(fi)
            zs, ys = lens.ray_paths(field_index=fi, wavelength_index=wi,
                                    n_fan=nf, pupil_radius=rp)
            for j in range(ys.shape[0]):
                ax.plot(zs[j].numpy(), ys[j].numpy(), color=col, lw=st.ray_lw,
                        alpha=st.ray_alpha, zorder=2)
    ax.set_aspect('equal', adjustable='datalim')
    return z_img, R_out


# --------------------------------------------------------------------------- #
#  single-state diagnostics
# --------------------------------------------------------------------------- #
def _layout_legend(ax, lens, st, chromatic):
    """Legend keyed by wavelength (chromatic) or field (monochromatic)."""
    if chromatic and lens.wavelengths_um.numel() > 1:
        labels = [f"{float(w)*1000:.0f} nm" for w in lens.wavelengths_um]
        handles = [plt.Line2D([], [], color=st.wcolor(i), label=l)
                   for i, l in enumerate(labels)]
        title = "wavelength"
    else:
        handles = [plt.Line2D([], [], color=st.color(i), label=l)
                   for i, l in enumerate(_field_labels(lens))]
        title = "field"
    leg = ax.legend(handles=handles, fontsize=st.tick_size, title=title,
                    loc='upper left', framealpha=0.9)
    leg.get_frame().set_facecolor(st.window_facecolor)
    leg.get_frame().set_edgecolor(st.font_color)
    leg.get_title().set_color(st.font_color)
    for txt in leg.get_texts():
        txt.set_color(st.font_color)


def _focus_inset(ax, lens, st, field_index, rays, n_fan):
    """Magnified inset on the focus region so per-wavelength (chromatic) ray
    separation is visible even when it is sub-pixel on the full-scale axes.

    Traces every wavelength's ray fan, finds the focus window (tight box around
    where the marginal rays converge just ahead of the image plane), and redraws
    the same rays there. Returns the inset axes (or None if <2 wavelengths).
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    W = lens.wavelengths_um.numel()
    if W < 2:
        return None
    z_img = float(torch.cumsum(lens.thick, 0)[-1])
    rp = _limiting_radius(lens)
    nf = 3 if rays == "chief_marginal" else int(n_fan)
    # collect each ray's final straight segment (exit of last surface -> image)
    # per wavelength; between the last surface and the image plane rays are
    # straight, so we can resample them at any z to find best focus.
    segs = []  # (wi, z_exit(nf,), y_exit(nf,), slope(nf,))
    for wi in range(W):
        zs, ys = lens.ray_paths(field_index=field_index, wavelength_index=wi,
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


def plot_lens_layout(lens, field_index: Optional[int] = None,
                     rays: str = "chief_marginal", n_fan: int = 9,
                     chromatic: bool = False, title: str = "Lens layout",
                     focus_inset: Optional[bool] = None,
                     style: Optional[VizStyle] = None):
    """Cross-section drawing: closed glass elements + traced rays.

    Each refractive element is drawn as a closed body (front and back surfaces
    joined at a ground rim). By default the fill is transparent (outline only);
    set ``style=VizStyle(glass_alpha=...)`` for a tinted fill.

    Rays: ``rays="chief_marginal"`` (default) draws the classic chief + two
    marginal rays per field; ``rays="fan"`` draws a denser ``n_fan`` fan. Rays
    launch from the entrance pupil across the limiting clear aperture, so they
    stay inside the glass and each bend sits on the true surface curve.

    ``chromatic=True`` traces every wavelength and colours rays by wavelength
    (paper convention, blue->green->red); otherwise the d-line is traced and
    rays are coloured by field. Pass ``style=VizStyle(...)`` to restyle
    surfaces / field lines / window / font.
    """
    st = _S(style)
    fig, ax = _fig(w=6.4, h=3.4, style=st)
    fields = [field_index] if field_index is not None else None
    z_img, R_out = _draw_cross_section(ax, lens, st, rays=rays, n_fan=n_fan,
                                       fields=fields, chromatic=chromatic)
    ax.text(z_img, R_out * 1.05, "image", ha='center', va='bottom',
            fontsize=st.tick_size, color=st.image_plane_color,
            fontfamily=st.font_family)
    ax.set_xlabel("z  (mm)"); ax.set_ylabel("y  (mm)")
    ax.set_title(title)
    ax.grid(True, color=st.grid_color, alpha=st.grid_alpha, zorder=0)
    multi_field = (field_index is None and lens.fields_deg.numel() > 1)
    if multi_field or (chromatic and lens.wavelengths_um.numel() > 1):
        _layout_legend(ax, lens, st, chromatic)
    # focus-region zoom: default on when chromatic (color split is often
    # sub-pixel on the full-scale axes) and a single field is shown
    want_inset = focus_inset
    if want_inset is None:
        want_inset = bool(chromatic and lens.wavelengths_um.numel() > 1
                          and field_index is not None)
    if want_inset:
        fi = field_index if field_index is not None else 0
        _focus_inset(ax, lens, st, fi, rays, n_fan)
    fig.tight_layout()
    return fig


def plot_spot_diagram(lens, wavelength_index: int = 0, title: str = "Spot diagram",
                      view_um: Optional[float] = None, style: Optional[VizStyle] = None):
    """Per-field spot scatter (microns, centroid-referenced)."""
    st = _S(style)
    sp = lens.forward()
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
        ax.set_title(f"{_field_labels(lens)[fi]}   RMS {float(rms[fi])*1000:.1f} \u00b5m")
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


def plot_psf(lens, bridge, title: str = "Geometric PSF (KDE)",
             cmap: str = 'inferno', style: Optional[VizStyle] = None):
    """Per-field PSF grid, built with the bridge's KDE settings.

    ``bridge.psf`` returns only the single configured field; here we render the
    PSF of *every* field, reusing the bridge's grid/pitch/sigma so what you see
    matches what the pipeline convolves with.
    """
    from .bridge.kde_psf import kde_psf
    st = _S(style)
    sp = lens.forward()
    psf = kde_psf(sp, bridge.grid_size, bridge.pixel_pitch_mm,
                  bridge.sigma_bins, per_field=True)   # (F,W,G,G)
    F = psf.shape[0]
    fig, axs = _fig(1, F, w=2.4, h=2.6, style=st)
    if F == 1:
        axs = [axs]
    for fi in range(F):
        ax = axs[fi]
        p = psf[fi, 0].detach().numpy()
        ax.imshow(p, cmap=cmap)
        ax.set_title(_field_labels(lens)[fi])
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=st.title_size, color=st.font_color,
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
def compare_lenses(lens, theta_before, theta_after,
                   labels=("start", "optimized"), rays: str = "chief_marginal",
                   n_fan: int = 9, chromatic: bool = False,
                   view_um: Optional[float] = None, style: Optional[VizStyle] = None):
    """2x2: layout (before / after) over spot diagrams (before / after).

    Uses the same closed-element / entrance-pupil ray renderer as
    ``plot_lens_layout`` (chief+marginal rays by default; ``chromatic=True``
    for per-wavelength colouring). Restores ``lens`` to its incoming theta on
    exit, so it is side-effect free.
    """
    st = _S(style)
    theta_saved = lens.get_theta().clone()
    fig, axs = plt.subplots(2, 2, figsize=(9.0, 6.4))
    fig.patch.set_facecolor(st.figure_facecolor)
    for a in axs.ravel():
        _style_axes(a, st)
    for col, (th, lab) in enumerate(zip((theta_before, theta_after), labels)):
        lens.set_theta(th)
        # --- layout (top row) ---
        axL = axs[0, col]
        _draw_cross_section(axL, lens, st, rays=rays, n_fan=n_fan,
                            chromatic=chromatic)
        axL.set_title(f"{lab} \u2014 layout")
        axL.set_xlabel("z (mm)"); axL.set_ylabel("y (mm)")
        # --- worst-field spot (bottom row) ---
        axS = axs[1, col]
        sp = lens.forward(); cent = sp.centroids(); rms = sp.rms_radius()
        fi = int(torch.argmax(rms))
        m = sp.valid[fi, 0]
        pts = ((sp.xy[fi, 0][m] - cent[fi]) * 1000.0).detach().numpy()
        axS.scatter(pts[:, 0], pts[:, 1], s=4,
                    color=st.color(fi), alpha=0.6, edgecolors='none')
        axS.set_aspect('equal')
        esr = float(sp.effective_spot_radius()) * 1000
        axS.set_title(f"{lab} \u2014 spot @ {_field_labels(lens)[fi]}  (ESR {esr:.1f} \u00b5m)")
        axS.set_xlabel("x (\u00b5m)"); axS.set_ylabel("y (\u00b5m)")
        if view_um:
            axS.set_xlim(-view_um, view_um); axS.set_ylim(-view_um, view_um)
    lens.set_theta(theta_saved)
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
    recorded thetas can be replayed through the lens to visualize how the spot /
    layout evolved -- without re-running the optimization.

    Usage
    -----
        rec = OptimizationRecorder(lens)
        lm.run(40, callback=rec)
        viz.plot_spot_evolution(lens, rec)
    """
    def __init__(self, lens, stride: int = 1):
        self.lens = lens
        self.stride = int(stride)
        self.thetas: List[torch.Tensor] = [lens.get_theta().clone()]
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


def _spot_at(lens, theta, wavelength_index=0):
    saved = lens.get_theta().clone()
    lens.set_theta(theta)
    sp = lens.forward()
    esr = float(sp.effective_spot_radius()) * 1000
    fi = int(torch.argmax(sp.rms_radius()))
    m = sp.valid[fi, wavelength_index]
    pts = ((sp.xy[fi, wavelength_index][m] - sp.centroids()[fi]) * 1000.0).detach().numpy()
    lens.set_theta(saved)
    return pts, esr, fi


def plot_spot_evolution(lens, recorder, n_show: int = 6, view_um: Optional[float] = None,
                        style: Optional[VizStyle] = None):
    """Grid of worst-field spot diagrams sampled across the optimization."""
    st = _S(style)
    idx = np.linspace(0, recorder.n_frames - 1, min(n_show, recorder.n_frames))
    idx = sorted(set(int(round(i)) for i in idx))
    fig, axs = _fig(1, len(idx), w=2.2, h=2.5, style=st)
    if len(idx) == 1:
        axs = [axs]
    for ax, k in zip(axs, idx):
        pts, esr, fi = _spot_at(lens, recorder.thetas[k])
        ax.scatter(pts[:, 0], pts[:, 1], s=3, color=st.color(fi),
                   alpha=0.6, edgecolors='none')
        ax.set_aspect('equal')
        ax.set_title(f"it {k}\nESR {esr:.0f} \u00b5m")
        ax.set_xticks([]); ax.set_yticks([])
        if view_um:
            ax.set_xlim(-view_um, view_um); ax.set_ylim(-view_um, view_um)
    fig.suptitle("Spot evolution during optimization", fontsize=st.title_size,
                 color=st.font_color, fontfamily=st.font_family)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def animate_spot_evolution(lens, recorder, path: str = "spot_evolution.gif",
                           fps: int = 6, view_um: Optional[float] = None,
                           style: Optional[VizStyle] = None):
    """Write a GIF of the worst-field spot shrinking over iterations.

    Requires pillow (already a dependency). Returns the output path.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    st = _S(style)
    # fix a common view box from the first frame if none given
    if view_um is None:
        pts0, _, _ = _spot_at(lens, recorder.thetas[0])
        view_um = float(np.abs(pts0).max()) * 1.1 + 1.0
    fig, ax = _fig(w=3.4, h=3.4, style=st)
    sc = ax.scatter([], [], s=4, color=st.color(0), alpha=0.6, edgecolors='none')
    ax.set_xlim(-view_um, view_um); ax.set_ylim(-view_um, view_um)
    ax.set_aspect('equal'); ax.set_xlabel("x (\u00b5m)"); ax.set_ylabel("y (\u00b5m)")
    ttl = ax.set_title("")

    def _update(k):
        pts, esr, fi = _spot_at(lens, recorder.thetas[k])
        sc.set_offsets(pts)
        sc.set_color(st.color(fi))
        ttl.set_text(f"iteration {k}    ESR {esr:.1f} \u00b5m")
        return sc, ttl

    anim = FuncAnimation(fig, _update, frames=recorder.n_frames, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path
