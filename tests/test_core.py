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


# ---------------------------------------------------------------------------
# Soft geometric constraints as LM residuals (Supp. S2.2.2, Eq. S41-S47)
# ---------------------------------------------------------------------------
from e2e_optics.optimizer import constraints as _C


def test_ramp_is_one_sided():
    x = torch.tensor([-2.0, -1e-9, 0.0, 1e-9, 3.0], dtype=DT)
    r = _C.ramp(x)
    assert torch.all(r[:3] == 0) and r[3] > 0 and r[4] == 3.0
    # satisfied constraints must contribute no gradient either
    x = torch.tensor([-1.0], dtype=DT, requires_grad=True)
    g = torch.autograd.grad(_C.ramp(x).sum(), x, allow_unused=True)[0]
    assert g is None or float(g) == 0.0


def test_satisfied_constraints_are_exactly_zero():
    """A design inside every bound must produce an all-zero residual block."""
    lens = _toy_derived(rings=4)
    pk = lens.ray_probes()
    # bounds so loose that nothing can violate them
    r = _C.geometric_residuals(pk, tz_min=-1e3, tz_max=1e3,
                               theta_max_deg=89.999, normal_max_deg=89.999)
    assert r.numel() > 0 and float(r.abs().max()) == 0.0


def test_spacing_kinds_read_off_the_optics():
    """tz has one more entry than surfaces: pupil hop, spacings, image gap."""
    lens = _toy_derived(rings=4)
    kinds = _C.spacing_kinds_from_optics(lens)
    assert len(kinds) == lens.ray_probes()["tz"].shape[0]
    assert kinds[0] == "air" and kinds[-1] == "image"
    # the toy is glass / air / glass / air: spacings after surfaces 0 and 2 are glass
    assert kinds[1] == "glass" and kinds[2] == "air" and kinds[3] == "glass"


def test_ray_path_residual_activates_on_violation():
    """Collapsing an element thickness must raise l_RP above zero."""
    lens = _toy_derived(rings=4)
    gc = _C.GeometricConstraints.from_optics(lens, min_glass=0.25, min_air=0.1)
    assert float(_C.ray_path_residuals(lens.ray_probes(), gc.tz_min, gc.tz_max).max()) \
        < 0.05, "start design should be near-compliant"
    th = lens.get_theta().clone()
    idx = torch.nonzero(lens.var_mask).squeeze(-1).tolist()
    t0 = lens.n_surfaces * (2 + lens.max_asph)          # thickness block offset
    slots = [i for i, g in enumerate(idx) if g >= t0]
    th[slots[0]] = 0.05                                  # 2.2 mm -> 0.05 mm
    lens.set_theta(th)
    bad = _C.ray_path_residuals(lens.ray_probes(), gc.tz_min, gc.tz_max)
    assert float(bad.max()) > 0.0 and int((bad > 0).sum()) > 0


def test_geometric_residuals_are_differentiable():
    """The constraint block must give usable gradients w.r.t. theta."""
    lens = _toy_derived(rings=4)
    gc = _C.GeometricConstraints.from_optics(lens, min_glass=2.5)   # deliberately tight
    th = lens.get_theta().clone().requires_grad_(True)

    def f(t):
        p = lens._pack().clone().to(t.dtype)
        p[lens.var_mask] = t
        return gc(lens._trace_packed(p, probes=True)[2]).pow(2).sum()

    val = f(th)
    assert float(val) > 0.0, "tight bound should be violated"
    g = torch.autograd.grad(val, th)[0]
    assert torch.isfinite(g).all() and float(g.abs().max()) > 0.0


def test_constraints_rescue_a_violating_design():
    """LM with l_RP recovers a collapsed thickness; without it the design fails.

    This is the whole point of soft residuals: the TRA-only run drives the merit
    to ~0 by producing a degenerate lens whose rays stop tracing, while the
    constrained run climbs back toward the bound.
    """
    lens = _toy_derived(rings=4, fields=(0.0, 20.0))
    gc = _C.GeometricConstraints.from_optics(lens, min_glass=0.25, min_air=0.1)
    F = lens.fields_deg.numel(); W = lens.wavelengths_um.numel()
    P = lens.pupil.shape[0]
    idx = torch.nonzero(lens.var_mask).squeeze(-1).tolist()
    t0 = lens.n_surfaces * (2 + lens.max_asph)
    slot = [i for i, g in enumerate(idx) if g >= t0][0]

    def run(use_gc):
        ln = _toy_derived(rings=4, fields=(0.0, 20.0))
        th = ln.get_theta().clone(); th[slot] = 0.05
        ln.set_theta(th)

        def resid(t):
            p = ln._pack().clone().to(t.dtype); p[ln.var_mask] = t
            xy, _, pk = ln._trace_packed(p, probes=True)
            c = xy.mean(dim=(1, 2), keepdim=True)
            tra = ((xy - c) / (F * W * P) ** 0.5).reshape(-1)
            return torch.cat([tra, gc(pk)]) if use_gc else tra

        lm = LevenbergMarquardt(resid, th.clone(), lam0=1.0)
        lm.run(40)
        ln.set_theta(lm.theta)
        return float(ln.ray_probes()["tz"][1].min())

    tz_free = run(False)
    tz_con = run(True)
    assert tz_con > 0.2, f"constrained run did not approach the bound: {tz_con}"
    # the unconstrained run either diverges (NaN) or ignores the bound entirely
    assert (tz_free != tz_free) or abs(tz_free - 0.25) > abs(tz_con - 0.25)


# ---------------------------------------------------------------------------
# Spot-evolution visualization: one fixed field per row
# ---------------------------------------------------------------------------
def test_spot_at_field_selection_is_explicit():
    """field_index=None follows the worst field; an int pins one fixed field."""
    from e2e_optics import viz as _V
    lens = _toy_derived(rings=4, fields=(0.0, 15.0, 30.0))
    th = lens.get_theta().clone()
    _, _, fi_auto, _ = _V._spot_at(lens, th)                      # worst field
    for want in range(lens.fields_deg.numel()):
        _, _, got, rms = _V._spot_at(lens, th, field_index=want)
        assert got == want, f"asked for field {want}, plotted {got}"
        assert rms > 0.0
    # the automatic choice must really be the largest RMS field
    r = lens.forward().rms_radius()
    assert fi_auto == int(torch.argmax(r))


def test_worst_field_changes_during_a_run():
    """The defect the per-field grid fixes: the worst field is not one field.

    If this ever stops holding for the toy the grid is still correct, but the
    single-row worst-field plot would no longer be actively misleading -- so the
    test documents WHY the layout changed rather than asserting cosmetics.
    """
    from e2e_optics import viz as _V
    lens = _toy_derived(rings=4, fields=(0.0, 15.0, 30.0))
    F = lens.fields_deg.numel(); W = lens.wavelengths_um.numel()
    P = lens.pupil.shape[0]

    def resid(t):
        xy = lens.spot_from_theta(t).reshape(F, W, P, 2)
        c = xy.mean(dim=(1, 2), keepdim=True)
        return ((xy - c) / (F * W * P) ** 0.5).reshape(-1)

    recorder = _V.OptimizationRecorder(lens)
    LevenbergMarquardt(resid, lens.get_theta(), lam0=1.0).run(25, callback=recorder)
    worst = [_V._spot_at(lens, recorder.thetas[k])[2] for k in range(recorder.n_frames)]
    assert len(set(worst)) > 1, "expected the limiting field to change hands"


def test_evolution_grid_has_one_row_per_field():
    from e2e_optics import viz as _V
    lens = _toy_derived(rings=4, fields=(0.0, 20.0))
    recorder = _V.OptimizationRecorder(lens)
    recorder(lens.get_theta())          # recorder accepts a bare tensor
    fig = _V.plot_spot_evolution(lens, recorder, n_show=2)
    # 2 fields x 2 iterations
    assert len(fig.axes) == 4, f"expected a 2x2 grid, got {len(fig.axes)} axes"
    fig2 = _V.plot_spot_evolution(lens, recorder, n_show=2, fields=[1])
    assert len(fig2.axes) == 2
    _V.plt.close(fig); _V.plt.close(fig2)


def test_field_convergence_curve_covers_every_field():
    from e2e_optics import viz as _V
    lens = _toy_derived(rings=4, fields=(0.0, 20.0))
    recorder = _V.OptimizationRecorder(lens)
    for _ in range(3):
        recorder(lens.get_theta())
    fig = _V.plot_field_convergence(lens, recorder)
    ax = fig.axes[0]
    # one line per field plus the design-wide ESR curve
    assert len(ax.lines) >= lens.fields_deg.numel() + 1
    assert ax.get_yscale() == "log"
    _V.plt.close(fig)


def test_distortion_residuals_is_an_honest_stub():
    try:
        _C.distortion_residuals()
    except NotImplementedError as e:
        assert "S2.2.3" in str(e) or "paraxial" in str(e)
    else:
        raise AssertionError("expected NotImplementedError")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("PASS", name)
