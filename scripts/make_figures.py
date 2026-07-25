"""Regenerate every figure shipped in ``docs/figures/``.

One entry point so the README figures are reproducible from source rather than
hand-made. Run from the repo root:

    python scripts/make_figures.py                # -> docs/figures/
    python scripts/make_figures.py --out /tmp/f   # custom dir

Figures produced
----------------
  fig_viz_layout.png             redesigned layout: chief+marginal rays,
                                 entrance pupil, transparent glass, bends on the
                                 true surface curves
  fig_viz_layout_chromatic.png   on-axis F/d/C with the focus-region zoom inset
  fig_lens_layout.png            dense meridional fan (compatibility name)
  fig_spot_start.png             starting-design spot diagram
  fig_psf.png                    differentiable KDE geometric PSF
  fig_viz_compare.png            before/after (paper Fig. 4 style)
  fig_lm_convergence.png         TRA-LM merit curve
  fig_viz_evolution.png          spot-evolution grid across LM iterations
  fig_viz_evolution.gif          animated spot evolution
  fig_demo_restoration.png       capture + network restoration (from the demo)
  fig_demo_summary.png           convergence summary (from the demo)
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("OMP_PROC_BIND", "false")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("OMP_NUM_THREADS", "4")

# make the repo root importable no matter where this script is launched from
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import argparse
import matplotlib
matplotlib.use("Agg")
import torch

from e2e_optics.optics.raytrace import RotationallySymmetricLens, Surface
from e2e_optics.bridge.imaging import ConvolutionImaging
from e2e_optics.optimizer.lm import LevenbergMarquardt
from e2e_optics import viz

torch.manual_seed(0)
DT = torch.float64

# N-BK7 at the d-line (n_d) with its Abbe number, so the chromatic layout uses
# real dispersion. The two singlets carry glass; the air gaps carry none.
N_BK7, V_BK7 = 1.5168, 64.17


def build_toy(chromatic: bool, rings: int = 8):
    """Toy 2-element f/2 lens. ``chromatic`` decides whether it carries the
    F/d/C triplet (N-BK7 Abbe data, dispersion on) or a single d-line (mono)."""
    wl = [0.4861, 0.5876, 0.6563] if chromatic else [0.5876]
    S = Surface
    surfaces = [
        S(1 / 5.0,   0.0, [0., 0., 0.], 2.2, N_BK7, True,  2.6, V_BK7),
        S(-1 / 18.0, 0.0, [0., 0., 0.], 2.0, 1.0,   False, 2.8),
        S(1 / 9.0,   0.0, [0., 0., 0.], 2.2, N_BK7, False, 3.4, V_BK7),
        S(-1 / 9.0,  0.0, [0., 0., 0.], 2.1, 1.0,   False, 3.8),
    ]
    return RotationallySymmetricLens(
        surfaces, epd=5.0, fields_deg=[0.0, 15.0, 30.0], wavelengths_um=wl,
        n_pupil_rings=rings, variables=("curvature", "asph", "thickness"),
        dispersion=chromatic, dtype=DT)


def optimize_tra(lens, n_iters=40, callback=None):
    """Conventional spot minimization via LM on TRA residuals."""
    F = lens.fields_deg.numel(); W = lens.wavelengths_um.numel(); P = lens.pupil.shape[0]

    def resid(th):
        xy = lens.spot_from_theta(th).reshape(F, W, P, 2)
        c = xy.mean(dim=(1, 2), keepdim=True)
        return ((xy - c) / (F * W * P) ** 0.5).reshape(-1)

    lm = LevenbergMarquardt(resid, lens.get_theta(), lam0=1.0)
    lm.run(n_iters, callback=callback)
    return lm


def main(out="docs/figures"):
    os.makedirs(out, exist_ok=True)
    P = lambda name: os.path.join(out, name)

    # ---- layout figures (chromatic tracer, redesigned renderer) ----------
    lens_c = build_toy(chromatic=True)
    f = viz.plot_lens_layout(lens_c, rays="chief_marginal",
                             title="Toy 2-element lens \u2014 chief + marginal rays")
    f.savefig(P("fig_viz_layout.png"), dpi=200, bbox_inches="tight")

    f = viz.plot_lens_layout(lens_c, field_index=0, chromatic=True,
                             title="On-axis chromatic (F/d/C) + focus zoom")
    f.savefig(P("fig_viz_layout_chromatic.png"), dpi=200, bbox_inches="tight")

    f = viz.plot_lens_layout(lens_c, rays="fan", n_fan=9,
                             title="Meridional ray fan")
    f.savefig(P("fig_lens_layout.png"), dpi=200, bbox_inches="tight")

    # ---- starting-design diagnostics (monochromatic toy) -----------------
    lens = build_toy(chromatic=False)
    theta_start = lens.get_theta().clone()
    f = viz.plot_spot_diagram(lens, title="Starting design \u2014 spot diagram")
    f.savefig(P("fig_spot_start.png"), dpi=200, bbox_inches="tight")

    bridge = ConvolutionImaging(grid_size=25, pixel_pitch_mm=0.0113,
                                sigma_bins=2.0, noise_std_frac=0.002, seed=1)
    f = viz.plot_psf(lens, bridge, title="Differentiable geometric PSF (KDE)")
    f.savefig(P("fig_psf.png"), dpi=200, bbox_inches="tight")

    # ---- optimize + before/after + convergence + evolution ---------------
    rec = viz.OptimizationRecorder(lens)
    lm = optimize_tra(lens, 40, callback=rec)
    theta_opt = lm.theta.clone()

    f = viz.compare_lenses(lens, theta_start, theta_opt,
                           labels=("start", "optimized"), view_um=80)
    f.savefig(P("fig_viz_compare.png"), dpi=200, bbox_inches="tight")

    f = viz.plot_convergence(lm.history, title="Conventional TRA-LM lens design")
    f.savefig(P("fig_lm_convergence.png"), dpi=200, bbox_inches="tight")

    lens.set_theta(theta_start)
    f = viz.plot_spot_evolution(lens, rec, n_show=6)
    f.savefig(P("fig_viz_evolution.png"), dpi=200, bbox_inches="tight")
    viz.animate_spot_evolution(lens, rec, P("fig_viz_evolution.gif"), fps=6)
    lens.set_theta(theta_opt)
    print("layout / spot / evolution figures ->", out)

    # ---- demo restoration + summary (reuses the demo pipeline) -----------
    from demos.demo_toy_e2e import main as demo_main
    demo_main(out)
    print("demo restoration / summary ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    a = ap.parse_args()
    main(a.out)
