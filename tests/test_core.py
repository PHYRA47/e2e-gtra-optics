"""Core correctness tests -- the numerical guarantees the framework rests on.

Run:  pytest -q            (from the repo root)
or:   python tests/test_core.py

These encode the sanity checks used while building the package:
  * ray-tracer gradient matches finite differences (both AD modes);
  * KDE PSF conserves energy and is shift-invariant in total energy;
  * GTRA reduces to TRA in the conventional case;
  * GTRA's surrogate gradient equals the true task-loss gradient (the paper's
    central guarantee);
  * LM converges on a known least-squares problem;
  * the toy lens spot shrinks under TRA-LM.
"""
import os
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("OMP_PROC_BIND", "false")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import math
import torch
import torch.func as tf

from e2e_optics.optics.raytrace import RotationallySymmetricLens, Surface
from e2e_optics.optics.base import SpotDiagram
from e2e_optics.bridge.kde_psf import kde_psf
from e2e_optics.bridge.gtra import tra_residuals, gtra_residuals
from e2e_optics.optimizer.lm import LevenbergMarquardt

DT = torch.float64


def _toy(rings=6, fields=(0.0, 15.0, 30.0)):
    s = [
        Surface(1 / 5.0,   0.0, [0., 0., 0.], 2.2, 1.5, True,  2.6),
        Surface(-1 / 18.0, 0.0, [0., 0., 0.], 2.0, 1.0, False, 2.8),
        Surface(1 / 9.0,   0.0, [0., 0., 0.], 2.2, 1.5, False, 3.4),
        Surface(-1 / 9.0,  0.0, [0., 0., 0.], 2.1, 1.0, False, 3.8),
    ]
    return RotationallySymmetricLens(s, epd=5.0, fields_deg=list(fields),
                                     wavelengths_um=[0.5876], n_pupil_rings=rings,
                                     variables=("curvature", "asph", "thickness"), dtype=DT)


def test_raytrace_gradient_matches_fd():
    # on-axis only: the central-difference reference is the weak link (the AD
    # modes agree with each other to ~1e-11); off-axis edge rays make FD noisy.
    lens = _toy(rings=4, fields=(0.0,))
    th = lens.get_theta()
    f = lambda t: lens.spot_from_theta(t).pow(2).sum()
    g = tf.grad(f)(th)
    eps = 1e-6
    gfd = torch.zeros_like(g)
    for i in range(th.numel()):
        d = torch.zeros_like(th); d[i] = eps
        gfd[i] = (f(th + d) - f(th - d)) / (2 * eps)
    rel = (g - gfd).norm() / gfd.norm()
    assert rel < 1e-4, rel


def test_forward_and_backward_ad_agree():
    # The strong gradient check: jacfwd (forward-mode, used by LM) and jacrev
    # (backward-mode) must agree on the ray tracer to ~machine precision.
    lens = _toy(rings=4, fields=(0.0, 15.0))
    th = lens.get_theta()
    Jf = tf.jacfwd(lens.spot_from_theta)(th)
    Jr = tf.jacrev(lens.spot_from_theta)(th)
    assert (Jf - Jr).norm() / Jr.norm() < 1e-10


def test_kde_energy_conserved():
    xy = torch.tensor([[[[0.011, 0.007]]]], dtype=DT)
    tot = []
    for s in torch.linspace(0, 0.02, 6):
        sp = SpotDiagram(xy=xy + s, valid=torch.ones(1, 1, 1, dtype=torch.bool))
        tot.append(float(kde_psf(sp, 33, 0.02, 2.0).sum()))
    assert all(abs(t - 1.0) < 1e-9 for t in tot), tot


def test_gtra_reduces_to_tra():
    lens = _toy(rings=5)
    sp = lens.forward()
    F, W, P = sp.xy.shape[:3]

    def L_esr(e):
        xy = e.reshape(F, W, P, 2); c = xy.mean(dim=(1, 2), keepdim=True)
        return 0.5 * ((xy - c) ** 2).sum() / (F * W * P)

    eps0 = sp.flat().detach()
    L = L_esr(eps0); gL = tf.grad(L_esr)(eps0)
    r_g = gtra_residuals(eps0, L, gL)
    r_t = tra_residuals(sp)
    assert abs(0.5 * float(r_g.dot(r_g)) - 0.5 * float(r_t.dot(r_t))) < 1e-9


def test_gtra_control_values_give_tra_elementwise():
    """The explicit reduction: feeding tra_control_values() into the GTRA code
    path must reproduce l_TRA element-for-element (not just in loss)."""
    from e2e_optics import tra_control_values
    lens = _toy(rings=5)
    sp = lens.forward()
    eps0 = sp.flat().detach()
    w, eps_prime = tra_control_values(sp)
    r_g = gtra_residuals(eps0, L=None, grad_L=None,
                         w_override=w, eps_prime_override=eps_prime)
    r_t = tra_residuals(sp)
    assert torch.allclose(r_g, r_t, atol=1e-12), \
        float((r_g - r_t).abs().max())


def test_gtra_gradient_equals_task_gradient():
    lens = _toy(rings=5)
    th = lens.get_theta()
    F, W, P = lens.fields_deg.numel(), lens.wavelengths_um.numel(), lens.pupil.shape[0]

    def task(e):
        xy = e.reshape(F, W, P, 2); c = xy.mean(dim=(1, 2), keepdim=True)
        r2 = ((xy - c) ** 2).sum(-1)
        return torch.log1p(1e4 * r2).mean() + 0.3 * (r2 ** 0.5).mean()

    g_true = tf.grad(lambda t: task(lens.spot_from_theta(t)))(th)
    eps0 = lens.spot_from_theta(th).detach()
    L = task(eps0); gL = tf.grad(task)(eps0)
    resid = lambda t: gtra_residuals(lens.spot_from_theta(t), L, gL, eps0=eps0)
    r0 = resid(th); J = tf.jacfwd(resid)(th)
    g_gtra = J.t() @ r0
    cos = float(torch.dot(g_true, g_gtra) / (g_true.norm() * g_gtra.norm()))
    assert cos > 1 - 1e-6, cos


def test_lm_converges_linear():
    torch.manual_seed(0)
    A = torch.randn(30, 4, dtype=DT); x = torch.randn(4, dtype=DT); b = A @ x
    lm = LevenbergMarquardt(lambda z: A @ z - b, torch.zeros(4, dtype=DT))
    lm.run(25)
    assert (lm.theta - x).norm() < 1e-8


def test_toy_spot_shrinks_under_lm():
    lens = _toy(rings=6)
    F, W, P = lens.fields_deg.numel(), lens.wavelengths_um.numel(), lens.pupil.shape[0]
    esr0 = lens.forward().effective_spot_radius().item()

    def resid(t):
        xy = lens.spot_from_theta(t).reshape(F, W, P, 2)
        c = xy.mean(dim=(1, 2), keepdim=True)
        return ((xy - c) / (F * W * P) ** 0.5).reshape(-1)

    lm = LevenbergMarquardt(resid, lens.get_theta(), lam0=1.0); lm.run(30)
    lens.set_theta(lm.theta)
    esr1 = lens.forward().effective_spot_radius().item()
    assert esr1 < 0.25 * esr0, (esr0, esr1)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("PASS", name)
