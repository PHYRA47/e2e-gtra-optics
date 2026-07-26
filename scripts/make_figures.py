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
  fig_layout_fan.png             dense meridional fan
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

from e2e_optics.optics.raytrace import RotationallySymmetricOptics, Surface
from e2e_optics.bridge.imaging import ConvolutionImaging
from e2e_optics.optimizer.lm import LevenbergMarquardt
from e2e_optics import viz

torch.manual_seed(0)
DT = torch.float64

# N-BK7 at the d-line (n_d) with its Abbe number, so the chromatic layout uses
# real dispersion. The two singlets carry glass; the air gaps carry none.
N_BK7, V_BK7 = 1.5168, 64.17


def build_toy(chromatic: bool, rings: int = 8):
    """Toy 2-element f/2 optics. ``chromatic`` decides whether it carries the
    F/d/C triplet (N-BK7 Abbe data, dispersion on) or a single d-line (mono)."""
    wl = [0.4861, 0.5876, 0.6563] if chromatic else [0.5876]
    S = Surface
    surfaces = [
        # semi_aperture=None => derived from the ray footprint (see demo comment)
        S(1 / 5.0,   0.0, [0., 0., 0.], 2.2, N_BK7, True,  None, V_BK7),
        S(-1 / 18.0, 0.0, [0., 0., 0.], 2.0, 1.0,   False, None),
        S(1 / 9.0,   0.0, [0., 0., 0.], 2.2, N_BK7, False, None, V_BK7),
        S(-1 / 9.0,  0.0, [0., 0., 0.], 2.1, 1.0,   False, None),
    ]
    return RotationallySymmetricOptics(
        surfaces, epd=5.0, fields_deg=[0.0, 15.0, 30.0], wavelengths_um=wl,
        n_pupil_rings=rings, variables=("curvature", "asph", "thickness"),
        dispersion=chromatic, dtype=DT)


def optimize_tra(optics, n_iters=40, callback=None):
    """Conventional spot minimization via LM on TRA residuals."""
    F = optics.fields_deg.numel(); W = optics.wavelengths_um.numel(); P = optics.pupil.shape[0]

    def resid(th):
        xy = optics.spot_from_theta(th).reshape(F, W, P, 2)
        c = xy.mean(dim=(1, 2), keepdim=True)
        return ((xy - c) / (F * W * P) ** 0.5).reshape(-1)

    lm = LevenbergMarquardt(resid, optics.get_theta(), lam0=1.0)
    lm.run(n_iters, callback=callback)
    return lm


def main(out="docs/figures"):
    os.makedirs(out, exist_ok=True)
    P = lambda name: os.path.join(out, name)

    # ---- layout figures (chromatic tracer, redesigned renderer) ----------
    opt_c = build_toy(chromatic=True)
    f = viz.plot_layout(opt_c, rays="chief_marginal",
                             title="Toy 2-element optics \u2014 chief + marginal rays")
    f.savefig(P("fig_viz_layout.png"), dpi=200, bbox_inches="tight")

    f = viz.plot_layout(opt_c, field_index=0, chromatic=True,
                             title="On-axis chromatic (F/d/C) + focus zoom")
    f.savefig(P("fig_viz_layout_chromatic.png"), dpi=200, bbox_inches="tight")

    f = viz.plot_layout(opt_c, rays="fan", n_fan=9,
                             title="Meridional ray fan")
    f.savefig(P("fig_layout_fan.png"), dpi=200, bbox_inches="tight")

    # ---- starting-design diagnostics (monochromatic toy) -----------------
    optics = build_toy(chromatic=False)
    theta_start = optics.get_theta().clone()
    f = viz.plot_spot_diagram(optics, title="Starting design \u2014 spot diagram")
    f.savefig(P("fig_spot_start.png"), dpi=200, bbox_inches="tight")

    bridge = ConvolutionImaging(grid_size=25, pixel_pitch_mm=0.0113,
                                sigma_bins=2.0, noise_std_frac=0.002, seed=1)

    # ---- optimize + before/after + convergence + evolution ---------------
    rec = viz.OptimizationRecorder(optics)
    lm = optimize_tra(optics, 40, callback=rec)
    theta_opt = lm.theta.clone()

    # `optimize_tra` traces functionally through `spot_from_theta`, so the
    # solution lives in `lm.theta` and the optics object still holds theta_start.
    # It must be written back explicitly or every downstream figure silently
    # renders the STARTING design.
    optics.set_theta(theta_opt)

    # The PSF is rendered for the CONVERGED design: that is what the pipeline
    # convolves with, and it is the only design whose ~20 um spot fits the
    # 25x25 @ 11.3 um grid. The starting design's spot radius is ~1.5 mm, so the
    # same grid would clip 99% of its rays (plot_psf now labels the clipping).
    f = viz.plot_psf(optics, bridge, title="Differentiable geometric PSF (KDE)")
    f.savefig(P("fig_psf.png"), dpi=200, bbox_inches="tight")

    # view_um=None -> each spot panel autoscales to its own data. A shared box
    # cannot show a 420 um starting spot and a 20 um converged one at once.
    f = viz.compare_designs(optics, theta_start, theta_opt,
                           labels=("start", "optimized"), view_um=None)
    f.savefig(P("fig_viz_compare.png"), dpi=200, bbox_inches="tight")

    f = viz.plot_convergence(lm.history, title="Conventional TRA-LM optics design")
    f.savefig(P("fig_lm_convergence.png"), dpi=200, bbox_inches="tight")

    optics.set_theta(theta_start)
    f = viz.plot_spot_evolution(optics, rec, n_show=6)
    f.savefig(P("fig_viz_evolution.png"), dpi=200, bbox_inches="tight")
    viz.animate_spot_evolution(optics, rec, P("fig_viz_evolution.gif"), fps=6)
    optics.set_theta(theta_opt)
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
