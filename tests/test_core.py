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


def _toy_dispersive(dispersion=True, wavelengths=(0.4861, 0.5876, 0.6563),
                    rings=6):
    """Toy lens with N-BK7 Abbe data on the two glass elements."""
    NBK7, VBK7 = 1.5168, 64.17
    s = [
        Surface(1 / 5.0,   0.0, [0., 0., 0.], 2.2, NBK7, True,  2.6, VBK7),
        Surface(-1 / 18.0, 0.0, [0., 0., 0.], 2.0, 1.0,  False, 2.8),
        Surface(1 / 9.0,   0.0, [0., 0., 0.], 2.2, NBK7, False, 3.4, VBK7),
        Surface(-1 / 9.0,  0.0, [0., 0., 0.], 2.1, 1.0,  False, 3.8),
    ]
    return RotationallySymmetricLens(s, epd=5.0, fields_deg=(0.0, 15.0, 30.0),
                                     wavelengths_um=wavelengths, n_pupil_rings=rings,
                                     dispersion=dispersion, dtype=DT)


def test_hartmann_index_matches_nbk7():
    """The Hartmann model reproduces catalogue N-BK7 indices at F/d/C."""
    from e2e_optics.optics.raytrace import hartmann_index
    nF = hartmann_index(1.5168, 64.17, 0.4861)
    nd = hartmann_index(1.5168, 64.17, 0.5876)
    nC = hartmann_index(1.5168, 64.17, 0.6563)
    assert abs(nd - 1.5168) < 1e-4, nd
    assert abs(nF - 1.52238) < 1e-3, nF
    assert abs(nC - 1.51432) < 1e-3, nC
    assert nF > nd > nC                       # normal dispersion


def test_dispersion_off_is_monochromatic():
    """dispersion=False makes every wavelength trace the scalar d-line index."""
    lens = _toy_dispersive(dispersion=False)
    # all wavelengths share n_after -> identical spots per wavelength
    xy = lens.forward().xy                    # (F, W, P, 2)
    for wi in range(1, xy.shape[1]):
        assert torch.allclose(xy[:, 0], xy[:, wi], atol=1e-9), wi


def test_dispersion_dline_matches_scalar():
    """With dispersion on, the d-line result is bit-for-bit the scalar trace.

    The Hartmann model returns exactly n_d at the d-line by construction, so a
    single-wavelength trace at 0.5876 um and the d-line slice of a multi-colour
    trace must agree to machine precision (same glass, same pupil sampling).
    """
    scalar = _toy_dispersive(dispersion=True, wavelengths=(0.5876,))   # W=1 d-line
    chrom = _toy_dispersive(dispersion=True)                           # F/d/C
    xy_s = scalar.forward().xy[:, 0]
    xy_d = chrom.forward().xy[:, 1]                                    # d is index 1
    assert torch.allclose(xy_s, xy_d, atol=1e-12), (xy_s - xy_d).abs().max()


def test_chromatic_spot_spread():
    """Multiple wavelengths through real glass produce measurable colour."""
    lens = _toy_dispersive(dispersion=True)
    xy = lens.forward().xy                                        # (F, W, P, 2)
    # axial colour: on-axis RMS differs across wavelengths
    rms = []
    for wi in range(xy.shape[1]):
        c = xy[0, wi].mean(0)
        rms.append(((xy[0, wi] - c) ** 2).sum(-1).sqrt().mean().item())
    assert max(rms) - min(rms) > 1e-3, rms                       # mm-scale spread
    # lateral colour: 30-deg centroid shifts with wavelength
    cy = [xy[2, wi].mean(0)[1].item() for wi in range(xy.shape[1])]
    assert max(cy) - min(cy) > 1e-3, cy


# ---------------------------------------------------------------------------
# Derived element apertures (Supp. S2.1: aperture specified by EPD/f-number,
# element semi-diameters are never inputs)
# ---------------------------------------------------------------------------
def _toy_derived(rings=6, fields=(0.0, 15.0, 30.0)):
    """Same toy, but with every semi_aperture left as None (derived)."""
    s = [
        Surface(1 / 5.0,   0.0, [0., 0., 0.], 2.2, 1.5, True),
        Surface(-1 / 18.0, 0.0, [0., 0., 0.], 2.0, 1.0, False),
        Surface(1 / 9.0,   0.0, [0., 0., 0.], 2.2, 1.5, False),
        Surface(-1 / 9.0,  0.0, [0., 0., 0.], 2.1, 1.0, False),
    ]
    return RotationallySymmetricLens(s, epd=5.0, fields_deg=list(fields),
                                     wavelengths_um=[0.5876], n_pupil_rings=rings,
                                     variables=("curvature", "asph", "thickness"), dtype=DT)


def test_derived_apertures_contain_all_rays():
    """No traced ray may lie outside the element it refracts at."""
    lens = _toy_derived()
    sa = lens.semi_aperture
    r = lens.ray_probes()["r"]                     # (S,F,W,P)
    for si in range(lens.n_surfaces):
        assert float(r[si].max()) <= sa[si] + 1e-12, (si, float(r[si].max()), sa[si])


def test_derived_apertures_track_the_design():
    """Hardcoded apertures go stale under optimization; derived ones cannot."""
    lens = _toy_derived()
    before = list(lens.semi_aperture)
    th = lens.get_theta().clone()
    th[-1] = th[-1] + 2.0                          # stretch a thickness
    lens.set_theta(th)
    after = list(lens.semi_aperture)
    assert max(abs(a - b) for a, b in zip(after, before)) > 1e-3, (before, after)
    # and containment still holds at the new design
    r = lens.ray_probes()["r"]
    for si in range(lens.n_surfaces):
        assert float(r[si].max()) <= after[si] + 1e-12


def test_declared_aperture_overrides_and_reports_overflow():
    """An explicit semi_aperture is honoured and its overflow is reported."""
    lens = _toy()                                  # the old hardcoded values
    rows = lens.aperture_report()
    assert [r["declared"] for r in rows] == [2.6, 2.8, 3.4, 3.8]
    assert lens.semi_aperture == [2.6, 2.8, 3.4, 3.8]
    # the front two surfaces are known to be undersized for a 30-deg field
    assert rows[0]["overflow"] > 0.1 and rows[1]["overflow"] > 0.1
    assert rows[2]["overflow"] == 0.0 and rows[3]["overflow"] == 0.0


def test_probes_do_not_perturb_the_trace():
    """probes=True must leave traced values and gradients bit-identical."""
    lens = _toy_derived(rings=4)
    packed = lens._pack()
    xy0, v0 = lens._trace_packed(packed)
    xy1, v1, pk = lens._trace_packed(packed, probes=True)
    assert torch.equal(xy0, xy1) and torch.equal(v0, v1)
    th_a = lens.get_theta().clone().requires_grad_(True)
    g0 = torch.autograd.grad(lens.spot_from_theta(th_a).pow(2).sum(), th_a)[0]

    def loss(t):
        p = lens._pack().clone().to(t.dtype)
        p[lens.var_mask] = t
        return lens._trace_packed(p, probes=True)[0].reshape(-1).pow(2).sum()

    th_b = lens.get_theta().clone().requires_grad_(True)
    g1 = torch.autograd.grad(loss(th_b), th_b)[0]
    assert torch.equal(g0, g1), (g0 - g1).abs().max()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("PASS", name)
