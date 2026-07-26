# Work report — building the GTRA end-to-end optics package

Branch `fix/derived-apertures-and-naming`, commits `11a4228..3432940`.
34 files changed, +2335 / −285 lines. 37 tests pass.

This is a narrative record of what was built and, more usefully, what was found
to be wrong along the way. Every number quoted here was measured by running the
code, not recalled — the defects section in particular exists because several
figures printed numbers that did not match what they drew.

---

## 1. What the package is

A modular PyTorch implementation of Côté et al., *Generalized Aberrations for
Processing-Aware Optical Design* (SIGGRAPH 2026, DOI 10.1145/3817055), built so
the three stages can be swapped independently and gradients still flow end to
end.

```
optics  ──▶  bridge  ──▶  algorithm
(θ: curvature,  (GTRA residuals,   (reconstruction
 conic, asph,    KDE PSF,           or task network)
 thickness)      imaging model)
        ◀──────  backprop  ──────◀
```

| stage | package | swap in by |
|-------|---------|-----------|
| optics | `e2e_optics/optics/` | subclassing `BaseOptics` |
| bridge | `e2e_optics/bridge/` | subclassing `BaseBridge` |
| algorithm | `e2e_optics/algorithm/` | any `nn.Module` |
| optimizer | `e2e_optics/optimizer/` | `LevenbergMarquardt` or `JointOptimizer` |

4082 lines of Python across 22 modules, 37 tests, 15 figures.

### The idea being implemented

Conventional optical design optimizes a *vector* of transverse ray aberration
(TRA) residuals — thousands of them, one per (field, wavelength, pupil) sample —
which makes the Jacobian tall and lets Levenberg–Marquardt work. End-to-end
design optimizes a *scalar* task loss, so the Jacobian has one row, LM degenerates,
and everyone falls back to slow first-order methods.

GTRA restores the tall Jacobian. It lifts the scalar loss back onto the residual
vector:

```
ℓ_GTRA(θ) = √w · (ε(θ) − ε′),    w = ‖∇L_ε₀‖² / (2 L_ε₀),
                                  ε′ = ε₀ − 2 L_ε₀ ∇L_ε₀ / ‖∇L_ε₀‖²
```

recomputed every LM iteration. It reduces exactly to TRA when `w = 1/(wpf)` and
`ε′ = ε̄` — which `test_gtra_reduces_to_tra` asserts numerically.

The implementation detail that makes it practical is the AD split: `∇L_ε₀` comes
from backward mode through the *whole* pipeline including the network, while the
Jacobian `J ≈ √w · ∂ε/∂θ` comes from forward mode through the *ray tracer only*.
The network never enters the Jacobian.

---

## 2. Defects found and fixed

Most of the session's value is here. Each of these was caught by a number in a
figure disagreeing with a number in a log, and each turned out to be real.

### 2.1 Semi-apertures went stale during optimization

Both toys declared element semi-diameters by hand. Those are not design inputs —
the paper specifies aperture by entrance pupil diameter or f-number (Supp. S2.1)
and lets element size follow from the light the system carries. Running 40 TRA-LM
iterations and re-measuring showed how far the declared values drift:

| surface | declared | required @ start | required @ optimum | overflow |
|---------|---------:|-----------------:|-------------------:|---------:|
| 0 | 2.60 mm | 3.140 mm | 2.913 mm | +0.313 mm |
| 1 | 2.80 mm | 3.128 mm | 3.141 mm | +0.341 mm |
| 2 | 3.40 mm | 2.812 mm | 4.537 mm | **+1.137 mm** |
| 3 | 3.80 mm | 2.661 mm | 4.574 mm | +0.774 mm |

Every surface ends up carrying light outside its drawn rim; the rear element
misses by 20 %. In the layout figure this appeared as rays visibly refracting
*above* the glass they were drawn to refract at — correct ray tracing plus a
wrong aperture annotation.

Fixed by deriving apertures from the ray footprint. Declared values still
override (for a stop, or a part already manufactured), and everything is
detached: apertures are geometry for drawing and reporting, never optimization
variables, so there is no gradient to lose. Full note: `docs/semi_apertures.md`.

### 2.2 The layout fan drew light the stop blocks

Deriving apertures immediately exposed a second bug. `_limiting_radius` sized the
ray fan from a clear semi-aperture — the aperture stop's own if one is flagged,
else the smallest. The toy flags a **front stop**, so the fan launched at surface
0's derived radius of **3.297 mm** against an entrance-pupil radius of **2.500 mm**.
The figure was drawing rays the stop physically blocks.

Both branches overflow here (the smallest derived aperture is 2.794 mm, also
above the pupil), so the fix is a hard cap at `epd/2` regardless of branch.

> Note on process: the first version of this write-up attributed the overflow to
> the `min()` branch. It was the `is_stop` branch. The test passed either way
> because both radii exceed the pupil, so the prose could drift from the code
> without anything failing. `test_layout_fan_never_exceeds_the_entrance_pupil`
> now asserts *which* branch is live, so the documented scenario is pinned.

### 2.3 The layout tracer invented ray intersections

`ray_paths` — used only for drawing — had no validity mask, unlike the main
`_trace_packed`, which masks unreachable rays, back-marching and TIR. The
out-of-pupil rays from §2.2 miss the rear element at 30°, and Newton then
converged on the **far branch of the conic**:

```
z: [ 0.00  1.36  3.16  7.55  26.44  11.94]   ← vertex at z=26.4 mm
y: [ 3.06  3.84  3.76  4.68  13.87  13.87]      past a sensor at z=11.9 mm
```

The polyline doubles back from beyond the image plane. At full excursion this
produced the 69 mm ray that had been visible in the layout figures all along.

Fixed by applying the same reachability / back-march / TIR tests as the main
tracer and writing NaN for a truncated ray's remaining vertices, so the polyline
simply stops where the ray does.

### 2.4 The view clamp annotated limits it wasn't drawing

The layout clamps its y-view to the element extent so a diverging ray cannot
shrink the glass to a few pixels, and prints how far the rays actually reach.
The compare panel annotated **±7.5 mm** while visibly drawing **±11.5 mm**.

Cause: the cross-section sets an equal aspect with `adjustable='datalim'`, under
which matplotlib silently discards fixed y-limits to satisfy the aspect — it
even warns, *"Ignoring fixed y limits to fulfill fixed data aspect"*. The clamp
now switches to `adjustable='box'` before setting limits.

With §2.2 and §2.3 fixed, no ray on either toy escapes the element extent, so the
clamp no longer fires at all. It remains as a guard for designs that genuinely
diverge, and `test_layout_view_clamp_honours_its_annotation` drives it on a
synthetic 60 mm excursion to keep it honest.

### 2.5 Figures labelled with the wrong statistic

The compare figure drew the **worst field's** spot and labelled it with the
**design-wide ESR** — the mean over fields, necessarily smaller, understating the
spot on screen by about 1.6×. The evolution figure had the same conflation.

Both now lead with the drawn field's own RMS and print the design-wide mean
separately, explicitly labelled as a mean.

### 2.6 Figures were being written to the wrong directory

The figure script resolves `docs/figures` relative to the working directory. It
had been run from the workspace root, so the *repository's* tracked figures were
stale by several fixes while the ones under review looked correct. Regenerating
from inside the repository resolved it — worth knowing before trusting a figure.

### 2.7 A measurement error, not a package bug

While building the drift table in §2.1, before-and-after apertures came back
bit-identical, which cannot be true when curvature and thickness have moved. The
cause was in the measurement: `optimize_tra` returns the LM object and does not
write the result back into the optics, so the second reading was taken on an
unmodified design. **Call `optics.set_theta(lm.theta)` explicitly.** Recorded here
because it will silently produce null results for the next person who forgets.

---

## 3. Physics implemented

| component | reference | notes |
|-----------|-----------|-------|
| Hartmann dispersion `n(λ)=A+C/(λ−B)` | Eq. 8, Supp. S4 | validated against N-BK7 |
| Even-asphere sag | Eq. 9 | conic + `r⁴, r⁶, …` |
| Vector Snell refraction | Supp. S12–S13 | clamped TIR discriminant, validity mask |
| Newton ray marching | Supp. S1.3 | per-ray true intersection z |
| Concentric-ring pupil | Supp. S2.1 | equal-area rings |
| RMS spot radius / ESR | Supp. S23 | |
| KDE geometric PSF | Supp. S1.4.2 | triangular kernel, energy-conserving |
| GTRA residual lift | Eq. 6, Supp. S66–S68 | reduces to TRA — asserted by test |
| LM with adaptive damping | Eq. 11, Supp. S62–S65 | `D²` running-max, floor 1e-6 |
| Soft geometric constraints | Supp. S2.2.2, Eq. S41–S47 | one-sided ramps as LM residuals |

Ray tracing was validated against CODE V to ±0.001 µm in the source paper; our
gradient path is checked against finite differences by
`test_raytrace_gradient_matches_fd`, and forward/backward AD agreement by
`test_forward_and_backward_ad_agree`.

### Honest stubs

Unimplemented paper features raise `NotImplementedError` with a pointer to the
relevant supplementary section rather than silently returning an approximation:

- diffractive surfaces (Supp. S1.2.3) — grating-modified refraction *is* in the
  tracer; the surface type is the hook
- ray aiming / vignetting (Supp. S1.3.2)
- diffraction-compensated PSF (Supp. S1.4.3)
- distortion residuals (Supp. S2.2.3, Eq. S48–S49)
- Wiener deconvolution baseline (Supp. S1.6)

`test_distortion_residuals_is_an_honest_stub` asserts one of these still raises
with its section reference attached, so a stub cannot quietly become a wrong
answer.

---

## 4. Results on the toy

Two-element N-BK7 design, 3 fields (0°, 15°, 30°), 3 wavelengths (F/d/C),
EPD 5 mm.

| stage | ESR | image quality |
|-------|----:|---------------|
| starting design | 367.4 µm | — |
| TRA-LM (40 iters) | 20.53 µm | blurred capture 15.29 dB PSNR |
| end-to-end (GTRA + Adam) | 20.53 µm | restored 35.34 dB PSNR |

LM merit drops 8.57e−02 → 2.28e−04, converging by iteration ~16.

**Caveat, stated plainly:** the joint stage moves the optics by only
`max|Δθ| = 1.13e−06`, so the end-to-end ESR is identical to the TRA result. The
pipeline is wired and differentiable — gradients reach θ and the tests confirm
it — but on this toy the joint stage is not doing optical work. The restoration
gain is the network learning to invert a fixed PSF. Before using this toy to
demonstrate co-design, that needs investigating: likely candidates are the task
loss being too easy (a network that reaches 35 dB has little pressure to reshape
the optics), the learning-rate balance between θ and network weights, or the toy
having too few optical degrees of freedom to trade against.

---

## 5. Naming

The framework traces a stack of *surfaces*, and the paper's scope includes
diffractive elements. Calling the stage a "lens" would have made a DOE or
metasurface class read as a subclass of a lens. 291 occurrences were audited and
renamed — `RotationallySymmetricLens → RotationallySymmetricOptics`,
`plot_lens_layout → plot_layout`, `compare_lenses → compare_designs`, and so on.
Deprecated aliases keep existing scripts working, and
`test_deprecated_aliases_still_resolve` covers them.

---

## 6. Test suite

37 tests, runnable with no framework:

```bash
PYTHONPATH=. python tests/test_core.py
```

Grouped by what they protect: gradient correctness (finite differences,
forward/backward agreement), GTRA identities (reduction to TRA, gradient
equality), LM convergence, dispersion against N-BK7, aperture derivation,
constraint activation, PSF energy conservation and off-grid handling, API
naming, and the visualization invariants added for the defects in §2.

The visualization tests deserve a note: §2.2 and §2.4 were both cases where a
figure was wrong while every test passed. Figure invariants that can be stated
numerically — the fan cannot exceed the pupil, no vertex may sit beyond the image
plane, an annotated limit must survive `draw()` — are now asserted rather than
eyeballed.

---

## 7. Reproducing

```bash
PYTHONPATH=. python tests/test_core.py       # 37 tests
PYTHONPATH=. python demos/demo_toy_e2e.py    # three-stage demo
PYTHONPATH=. python scripts/make_figures.py  # regenerate docs/figures
```

Run from the repository root — the figure script resolves `docs/figures`
relative to the working directory (§2.6).

---

## 8. Suggested next steps

1. Diagnose the flat joint stage (§4) — it is the one result that does not yet
   demonstrate the paper's central claim.
2. Implement ray aiming and vignetting (Supp. S1.3.2, Eqs. S19–S22); currently a
   declared aperture narrower than the footprint is drawn but does not clip.
3. Add a diffractive surface type on the existing grating-modified refraction.
4. Scale past the two-element toy — the LM path is the interesting one, since
   that is where GTRA is meant to pay off against first-order methods.
