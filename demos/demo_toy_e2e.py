"""Toy end-to-end design demo -- the Fig. 4 problem from Cote et al. (2026).

Reproduces the *structure* of the paper's 2-element monochromatic toy: a small
2-element optics (f/2, EFL 10 mm, 60 deg FOV) imaging a concentric-ring chart onto
a 1024x1024-equivalent sensor, jointly designed with a restoration network.

Three stages, each printing metrics and saving a figure:

  1. STARTING DESIGN        -- weak 2-element optics, large aberration.
  2. CONVENTIONAL (TRA-LM)  -- minimize the spot with Levenberg-Marquardt on the
                              transverse ray aberration. This is the classical
                              optics baseline (smallest RMS spot).
  3. END-TO-END (GTRA-LM + Adam) -- alternate a GTRA optics step with an Adam
                              network step. The task loss is MSE between the
                              network's restored image and the sharp scene.

Run:  python -m e2e_optics.demos.demo_toy_e2e
It writes fig_demo_restoration.png / fig_demo_summary.png into the CWD.

Everything is CPU-friendly float64 and finishes in a couple of minutes. It is a
didactic reduction, not the paper's full experiment (single scene, small net,
few iterations); the numbers show the mechanism, not the paper's exact dB.
"""
from __future__ import annotations
import os
# --- OpenMP affinity guard: must precede torch import in this sandbox ---
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("OMP_PROC_BIND", "false")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import math
import numpy as np
import torch

from e2e_optics.optics.raytrace import RotationallySymmetricOptics, Surface
from e2e_optics.bridge.imaging import ConvolutionImaging
from e2e_optics.bridge.gtra import tra_residuals
from e2e_optics.algorithm.identity import IdentityRestoration
from e2e_optics.algorithm.unet import TinyUNet
from e2e_optics.optimizer.lm import LevenbergMarquardt
from e2e_optics.optimizer.joint import JointOptimizer, JointConfig

torch.manual_seed(0)
DT = torch.float64
N_GLASS = 1.5


# --------------------------------------------------------------------------- #
#  scene + metrics
# --------------------------------------------------------------------------- #
def ring_chart(size: int = 96, cycles: int = 7) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, size),
                            torch.linspace(-1, 1, size), indexing='ij')
    r = torch.sqrt(xx ** 2 + yy ** 2)
    img = (0.5 + 0.5 * torch.cos(2 * math.pi * cycles * r)) * (r < 1.0)
    return img.unsqueeze(0).to(DT)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    with torch.no_grad():
        return 10.0 * math.log10(1.0 / max(float(((a - b) ** 2).mean()), 1e-12))


def mse(a, b):
    return ((a - b) ** 2).mean()


# --------------------------------------------------------------------------- #
#  the starting 2-element design
# --------------------------------------------------------------------------- #
def build_toy(rings: int = 8, fields=(0.0, 15.0, 30.0)) -> RotationallySymmetricOptics:
    surfaces = [
        # semi_aperture left as None => DERIVED from the ray footprint. Declaring
        # it here would go stale the moment LM moves thickness/curvature: at the
        # TRA optimum this design needs 4.5 mm on the rear element where the old
        # hand-typed 3.8 mm would still be drawn, so the layout showed rays
        # passing outside the glass they refract at.
        Surface(1 / 5.0,   0.0, [0., 0., 0.], 2.2, N_GLASS, True),
        Surface(-1 / 18.0, 0.0, [0., 0., 0.], 2.0, 1.0,     False),
        Surface(1 / 9.0,   0.0, [0., 0., 0.], 2.2, N_GLASS, False),
        Surface(-1 / 9.0,  0.0, [0., 0., 0.], 2.1, 1.0,     False),
    ]
    return RotationallySymmetricOptics(
        surfaces, epd=5.0, fields_deg=list(fields), wavelengths_um=[0.5876],
        n_pupil_rings=rings, variables=("curvature", "asph", "thickness"), dtype=DT)


def optimize_tra(optics: RotationallySymmetricOptics, n_iters: int = 40):
    """Conventional spot minimization via LM on TRA residuals."""
    F = optics.fields_deg.numel(); W = optics.wavelengths_um.numel(); P = optics.pupil.shape[0]

    def resid(th):
        xy = optics.spot_from_theta(th).reshape(F, W, P, 2)
        c = xy.mean(dim=(1, 2), keepdim=True)
        return ((xy - c) / (F * W * P) ** 0.5).reshape(-1)

    lm = LevenbergMarquardt(resid, optics.get_theta(), lam0=1.0)
    lm.run(n_iters)
    optics.set_theta(lm.theta)
    return lm.history


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #
def main(outdir: str = "."):
    scene = ring_chart(96, cycles=7)
    sampler = lambda: (scene, scene)

    # 1. STARTING DESIGN --------------------------------------------------
    optics = build_toy()
    esr_start = optics.forward().effective_spot_radius().item() * 1000
    print(f"[1] starting design      ESR = {esr_start:7.2f} um")

    # 2. CONVENTIONAL TRA-LM ---------------------------------------------
    tra_hist = optimize_tra(optics, 40)
    esr_tra = optics.forward().effective_spot_radius().item() * 1000
    theta_tra = optics.get_theta().clone()
    print(f"[2] TRA-LM (min-spot)    ESR = {esr_tra:7.2f} um  "
          f"(merit {tra_hist[0]:.2e} -> {tra_hist[-1]:.2e})")

    # imaging model at paper-like pixel scale (spot ~ 1.8 px -> mild blur)
    bridge = ConvolutionImaging(grid_size=25, pixel_pitch_mm=0.0113,
                                sigma_bins=2.0, noise_std_frac=0.002, seed=1)
    cap_blur = bridge.simulate(optics.forward(), scene, add_noise=True)
    print(f"    blurred capture PSNR = {psnr(cap_blur, scene):5.2f} dB")

    # 3. END-TO-END: joint GTRA-LM optics + Adam net ------------------------
    net = TinyUNet(channels=1, width=16).to(DT)
    # (a) warm up the network on the TRA optics (network-only Adam)
    warm = JointOptimizer(optics, bridge, net, mse, sampler,
                          JointConfig(adam_lr=3e-3, algo_step=True, optics_step=False))
    warm.run(250)
    # (b) joint fine-tune: optics GTRA-LM + net Adam together
    joint = JointOptimizer(optics, bridge, net, mse, sampler,
                           JointConfig(lm_iters_per_step=2, adam_lr=1e-3,
                                       algo_step=True, optics_step=True))
    jhist = joint.run(30)
    esr_e2e = optics.forward().effective_spot_radius().item() * 1000
    # The joint stage moves the optics only slightly here: the TRA design is
    # already near the spot-metric optimum, so most of the task-loss reduction
    # is absorbed by the network. Report the parameter delta so "the optics
    # barely moved" is visible rather than something the reader must infer
    # from an ESR that rounds to the same value.
    dtheta = (optics.get_theta().detach() - theta_tra).abs().max().item()
    restored = net.restore(bridge.simulate(optics.forward(), scene, add_noise=True))
    print(f"[3] end-to-end (GTRA+Adam) ESR = {esr_e2e:7.2f} um  "
          f"restored PSNR = {psnr(restored, scene):5.2f} dB")
    print(f"    joint stage moved optics by max|d(theta)| = {dtheta:.2e}")

    _figures(outdir, scene, cap_blur, restored, tra_hist, jhist,
             esr_start, esr_tra, esr_e2e)
    return dict(esr_start=esr_start, esr_tra=esr_tra, esr_e2e=esr_e2e,
                joint_dtheta=dtheta,
                blur_psnr=psnr(cap_blur, scene), restored_psnr=psnr(restored, scene))


def _figures(outdir, scene, blurred, restored, tra_hist, jhist,
             esr_start, esr_tra, esr_e2e):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # restoration triptych
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, img, title in zip(
            axs, [scene, blurred, restored],
            ["sharp scene (target)",
             f"blurred capture\nPSNR {psnr(blurred, scene):.1f} dB",
             f"network restored\nPSNR {psnr(restored, scene):.1f} dB"]):
        ax.imshow(img.squeeze(0).detach().numpy(), cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("End-to-end toy: capture and restoration", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_demo_restoration.png"), dpi=200)

    # convergence summary
    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    a1.semilogy(range(len(tra_hist)), tra_hist, color='#2166ac', lw=1.5, marker='o', ms=2)
    a1.set_xlabel("LM iteration"); a1.set_ylabel(r"$\frac{1}{2}\|\ell_{\mathrm{TRA}}\|^2$")
    a1.set_title("Conventional TRA-LM optics design")
    a2.plot(range(len(jhist.task_loss)), jhist.task_loss, color='#b2182b', lw=1.5, marker='o', ms=2)
    a2.set_xlabel("joint iteration"); a2.set_ylabel("task MSE")
    a2.set_title("End-to-end joint fine-tune")
    fig2.tight_layout(); fig2.savefig(os.path.join(outdir, "fig_demo_summary.png"), dpi=200)


if __name__ == "__main__":
    stats = main(".")
    print("\nsummary:", {k: f"{v:.4g}" for k, v in stats.items()})
