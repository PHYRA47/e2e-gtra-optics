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
from typing import Optional, Sequence, List
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

# consistent, colour-blind-safe field palette used across every figure
FIELD_COLORS = ['#2166ac', '#b2182b', '#1a9850', '#762a83', '#e08214',
                '#01665e', '#8c510a', '#54278f']


def _fig(nrows=1, ncols=1, w=3.4, h=3.0):
    fig, ax = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows))
    return fig, ax


def _field_labels(lens):
    return [f"{float(a):.0f}\u00b0" for a in lens.fields_deg]


# --------------------------------------------------------------------------- #
#  single-state diagnostics
# --------------------------------------------------------------------------- #
def plot_lens_layout(lens, field_index: Optional[int] = None, n_fan: int = 9,
                     title: str = "Lens layout"):
    """Cross-section: surface profiles + traced meridional ray fans.

    If ``field_index`` is None, overlays one ray fan per field (each in its
    field colour); otherwise shows just that field.
    """
    fig, ax = _fig(w=6.4, h=3.4)
    # surfaces
    for si in range(lens.n_surfaces):
        r, z = lens.surface_profile(si)
        ax.plot(z.numpy(), r.numpy(), color='k', lw=1.3, zorder=3)
    # image plane
    z_img = float(torch.cumsum(lens.thick, 0)[-1])
    sa = max(lens.semi_aperture)
    ax.axvline(z_img, color='0.4', ls='--', lw=1.0, zorder=1)
    ax.text(z_img, sa * 1.05, "image", ha='center', va='bottom', fontsize=7,
            color='0.4')
    # rays
    fields = [field_index] if field_index is not None else range(lens.fields_deg.numel())
    for fi in fields:
        col = FIELD_COLORS[fi % len(FIELD_COLORS)]
        z_nodes, ys = lens.ray_paths(field_index=fi, n_fan=n_fan)
        z_np = z_nodes.numpy()
        for j in range(ys.shape[0]):
            ax.plot(z_np, ys[j].numpy(), color=col, lw=0.7, alpha=0.8, zorder=2)
    ax.set_xlabel("z  (mm)"); ax.set_ylabel("y  (mm)")
    ax.set_title(title); ax.set_aspect('equal', adjustable='datalim')
    if field_index is None and lens.fields_deg.numel() > 1:
        handles = [plt.Line2D([], [], color=FIELD_COLORS[i % len(FIELD_COLORS)],
                              label=l) for i, l in enumerate(_field_labels(lens))]
        ax.legend(handles=handles, fontsize=7, title="field", loc='upper left')
    fig.tight_layout()
    return fig


def plot_spot_diagram(lens, wavelength_index: int = 0, title: str = "Spot diagram",
                      view_um: Optional[float] = None):
    """Per-field spot scatter (microns, centroid-referenced)."""
    sp = lens.forward()
    F = sp.xy.shape[0]
    cent = sp.centroids()
    fig, axs = _fig(1, F, w=2.6, h=2.8)
    if F == 1:
        axs = [axs]
    rms = sp.rms_radius()
    for fi in range(F):
        ax = axs[fi]
        m = sp.valid[fi, wavelength_index]
        pts = (sp.xy[fi, wavelength_index][m] - cent[fi]) * 1000.0   # um
        pts = pts.detach().numpy()
        ax.scatter(pts[:, 0], pts[:, 1], s=3, color=FIELD_COLORS[fi % len(FIELD_COLORS)],
                   alpha=0.6, edgecolors='none')
        ax.set_title(f"{_field_labels(lens)[fi]}   RMS {float(rms[fi])*1000:.1f} \u00b5m",
                     fontsize=8)
        ax.set_aspect('equal'); ax.axhline(0, color='0.8', lw=0.5); ax.axvline(0, color='0.8', lw=0.5)
        if view_um:
            ax.set_xlim(-view_um, view_um); ax.set_ylim(-view_um, view_um)
        ax.set_xlabel("x (\u00b5m)", fontsize=7)
        if fi == 0:
            ax.set_ylabel("y (\u00b5m)", fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    return fig


def plot_psf(lens, bridge, title: str = "Geometric PSF (KDE)"):
    """Per-field PSF grid, built with the bridge's KDE settings.

    ``bridge.psf`` returns only the single configured field; here we render the
    PSF of *every* field, reusing the bridge's grid/pitch/sigma so what you see
    matches what the pipeline convolves with.
    """
    from .bridge.kde_psf import kde_psf
    sp = lens.forward()
    psf = kde_psf(sp, bridge.grid_size, bridge.pixel_pitch_mm,
                  bridge.sigma_bins, per_field=True)   # (F,W,G,G)
    F = psf.shape[0]
    fig, axs = _fig(1, F, w=2.4, h=2.6)
    if F == 1:
        axs = [axs]
    for fi in range(F):
        ax = axs[fi]
        p = psf[fi, 0].detach().numpy()
        ax.imshow(p, cmap='inferno')
        ax.set_title(_field_labels(lens)[fi], fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    return fig


def plot_convergence(history, ylabel: str = r"$\frac{1}{2}\|\ell\|^2$",
                     title: str = "Convergence", logy: bool = True,
                     color: str = '#2166ac'):
    """Loss-vs-iteration curve. ``history`` is a list/array of scalar losses."""
    h = [float(x) for x in history]
    fig, ax = _fig(w=4.2, h=3.0)
    (ax.semilogy if logy else ax.plot)(range(len(h)), h, color=color, lw=1.5,
                                       marker='o', ms=2.5)
    ax.set_xlabel("iteration"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, which='both', alpha=0.25)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
#  before / after comparison  (paper Fig. 4 style)
# --------------------------------------------------------------------------- #
def compare_lenses(lens, theta_before, theta_after,
                   labels=("start", "optimized"), n_fan: int = 9,
                   view_um: Optional[float] = None):
    """2x2: layout (before / after) over spot diagrams (before / after).

    Restores ``lens`` to its incoming theta on exit, so it is side-effect free.
    """
    theta_saved = lens.get_theta().clone()
    fig, axs = plt.subplots(2, 2, figsize=(9.0, 6.4))
    for col, (th, lab) in enumerate(zip((theta_before, theta_after), labels)):
        lens.set_theta(th)
        # --- layout (top row) ---
        axL = axs[0, col]
        for si in range(lens.n_surfaces):
            r, z = lens.surface_profile(si)
            axL.plot(z.numpy(), r.numpy(), color='k', lw=1.2)
        z_img = float(torch.cumsum(lens.thick, 0)[-1])
        axL.axvline(z_img, color='0.5', ls='--', lw=0.9)
        for fi in range(lens.fields_deg.numel()):
            zc, ys = lens.ray_paths(field_index=fi, n_fan=n_fan)
            for j in range(ys.shape[0]):
                axL.plot(zc.numpy(), ys[j].numpy(),
                         color=FIELD_COLORS[fi % len(FIELD_COLORS)], lw=0.6, alpha=0.8)
        axL.set_aspect('equal', adjustable='datalim')
        axL.set_title(f"{lab} \u2014 layout", fontsize=9)
        axL.set_xlabel("z (mm)"); axL.set_ylabel("y (mm)")
        # --- worst-field spot (bottom row) ---
        axS = axs[1, col]
        sp = lens.forward(); cent = sp.centroids(); rms = sp.rms_radius()
        fi = int(torch.argmax(rms))
        m = sp.valid[fi, 0]
        pts = ((sp.xy[fi, 0][m] - cent[fi]) * 1000.0).detach().numpy()
        axS.scatter(pts[:, 0], pts[:, 1], s=4,
                    color=FIELD_COLORS[fi % len(FIELD_COLORS)], alpha=0.6, edgecolors='none')
        axS.set_aspect('equal')
        esr = float(sp.effective_spot_radius()) * 1000
        axS.set_title(f"{lab} \u2014 spot @ {_field_labels(lens)[fi]}  (ESR {esr:.1f} \u00b5m)",
                      fontsize=9)
        axS.set_xlabel("x (\u00b5m)"); axS.set_ylabel("y (\u00b5m)")
        if view_um:
            axS.set_xlim(-view_um, view_um); axS.set_ylim(-view_um, view_um)
    lens.set_theta(theta_saved)
    fig.tight_layout()
    return fig


def plot_restoration(scene, blurred, restored):
    """Sharp / blurred / restored triptych with PSNR annotations."""
    def _psnr(a, b):
        with torch.no_grad():
            return 10.0 * np.log10(1.0 / max(float(((a - b) ** 2).mean()), 1e-12))
    fig, axs = _fig(1, 3, w=2.5, h=2.7)
    imgs = [scene, blurred, restored]
    titles = ["sharp (target)",
              f"blurred\nPSNR {_psnr(blurred, scene):.1f} dB",
              f"restored\nPSNR {_psnr(restored, scene):.1f} dB"]
    for ax, im, t in zip(axs, imgs, titles):
        ax.imshow(im.squeeze().detach().numpy(), cmap='gray', vmin=0, vmax=1)
        ax.set_title(t, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
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


def plot_spot_evolution(lens, recorder, n_show: int = 6, view_um: Optional[float] = None):
    """Grid of worst-field spot diagrams sampled across the optimization."""
    idx = np.linspace(0, recorder.n_frames - 1, min(n_show, recorder.n_frames))
    idx = sorted(set(int(round(i)) for i in idx))
    fig, axs = _fig(1, len(idx), w=2.2, h=2.5)
    if len(idx) == 1:
        axs = [axs]
    for ax, k in zip(axs, idx):
        pts, esr, fi = _spot_at(lens, recorder.thetas[k])
        ax.scatter(pts[:, 0], pts[:, 1], s=3, color=FIELD_COLORS[fi % len(FIELD_COLORS)],
                   alpha=0.6, edgecolors='none')
        ax.set_aspect('equal')
        ax.set_title(f"it {k}\nESR {esr:.0f} \u00b5m", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        if view_um:
            ax.set_xlim(-view_um, view_um); ax.set_ylim(-view_um, view_um)
    fig.suptitle("Spot evolution during optimization", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def animate_spot_evolution(lens, recorder, path: str = "spot_evolution.gif",
                           fps: int = 6, view_um: Optional[float] = None):
    """Write a GIF of the worst-field spot shrinking over iterations.

    Requires pillow (already a dependency). Returns the output path.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    # fix a common view box from the first frame if none given
    if view_um is None:
        pts0, _, _ = _spot_at(lens, recorder.thetas[0])
        view_um = float(np.abs(pts0).max()) * 1.1 + 1.0
    fig, ax = _fig(w=3.4, h=3.4)
    sc = ax.scatter([], [], s=4, color=FIELD_COLORS[0], alpha=0.6, edgecolors='none')
    ax.set_xlim(-view_um, view_um); ax.set_ylim(-view_um, view_um)
    ax.set_aspect('equal'); ax.set_xlabel("x (\u00b5m)"); ax.set_ylabel("y (\u00b5m)")
    ttl = ax.set_title("")

    def _update(k):
        pts, esr, fi = _spot_at(lens, recorder.thetas[k])
        sc.set_offsets(pts)
        sc.set_color(FIELD_COLORS[fi % len(FIELD_COLORS)])
        ttl.set_text(f"iteration {k}    ESR {esr:.1f} \u00b5m")
        return sc, ttl

    anim = FuncAnimation(fig, _update, frames=recorder.n_frames, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path
