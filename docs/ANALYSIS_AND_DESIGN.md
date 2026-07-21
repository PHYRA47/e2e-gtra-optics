# Generalized Aberrations for Processing-Aware Optical Design
## Paper analysis and a modular implementation architecture

**Source paper:** Geoffroi Côté, Ethan Tseng, Felix Heide, *Generalized Aberrations
for Processing-Aware Optical Design*, ACM Transactions on Graphics (2026).
DOI: 10.1145/3817055 · Project: https://light.princeton.edu/generalized-aberrations

This document does two things:

1. **Cuts the paper thoroughly** — the problem it solves, the key idea (GTRA), and
   every moving part of the pipeline, with the equations you actually need to write code.
2. **Maps every concept onto a modular, swappable code architecture** — the three-part
   end-to-end pipeline you described (optics → bridge → algorithm) plus the optimization
   engine that ties them together through backpropagation.

The companion package `e2e_optics/` implements a **minimal-but-complete runnable slice**
of this architecture: every one of the three parts is present and connected by backprop,
and the whole thing is validated on the paper's own two-element toy problem (Fig. 4).
Advanced features (optimizable vignetting, diffraction-compensated PSFs, glass-mesh
constraints, spatially-varying convolution, the dual-NAFNet + Wiener IRM) live behind
clean extension hooks so you can grow the framework one piece at a time.

---

## Part 0 — The one-paragraph summary

Modern cameras are optimized *end-to-end*: the lens and the image-processing network are
trained together against a single downstream loss (e.g. restored-image quality). The
established, robust optical-design optimizer — **Levenberg–Marquardt (LM)** — needs a
*vector* of residuals (one per ray), but a deep-learning loss is a single *scalar*. That
mismatch makes the Jacobian rank-deficient and breaks LM, so the whole field falls back on
SGD/Adam, which converge slowly and get stuck. The paper's contribution, **Generalized
Transverse Ray Aberrations (GTRA)**, is a way to *lift* the scalar loss back into a
per-ray residual vector that has the same value and gradient as the scalar loss at the
current point — restoring the structure LM needs while remaining faithful to the
task objective. This lets you run the robust LM machinery of conventional lens design on
task-driven / end-to-end problems.

The single most important sentence for your implementation: **GTRA is a transformation that
lives in the "bridge" between optics and algorithm.** It converts `(spot_diagram, scalar_loss,
∂loss/∂spot_diagram)` into a per-ray least-squares residual. That is the "middle part" of
your pipeline, and it is where the Jacobian / spot-diagram / "all that good stuff" belongs.

---

## Part 1 — The problem: why LM breaks in end-to-end design

### 1.1 Conventional optical design (the thing that works)

Conventional lens design minimizes a **sum-of-squares merit function**

$$L(\theta) = \tfrac12 \lVert \ell(\theta) \rVert^2, \tag{2}$$

where `θ ∈ Rᴺ` are the lens variables (curvatures, spacings, aspheric coefficients, glass)
and `ℓ ∈ Rᴹ` is a vector of **residuals**. The dominant residual is the **Transverse Ray
Aberration (TRA)** — for every traced ray, how far it lands from the ideal image point:

$$\ell_{\text{TRA}}(\theta) = \tfrac{1}{\sqrt{fwp}}\,\big(\epsilon(\theta) - \bar\epsilon(\theta)\big) \in \mathbb{R}^{2fwp}, \tag{3}$$

where `ε(θ)` is the **spot diagram** — the (x, y) landing coordinates of all rays at the image
plane, for `f` fields × `w` wavelengths × `p` pupil positions — and `ε̄` is the per-field centroid.
`‖ℓ_TRA‖` for one field is exactly its **effective spot radius**.

Because `M = 2fwp` (thousands of residuals) is much larger than `N` (dozens of variables), the
Jacobian `J ∈ Rᴹˣᴺ` has full column rank, and **Levenberg–Marquardt** works beautifully:

$$\Delta\theta = -\big(J^\top J + \lambda D^2\big)^{-1} J^\top \ell_0. \tag{11}$$

### 1.2 Task-driven / end-to-end design (where it breaks)

Task-driven design replaces the hand-crafted merit function with a **scalar** downstream loss
`L_TD(θ)` — e.g. MSE between a restored image and the ground truth. If you naively use that
scalar as your only residual (`M = 1`), then `J ∈ R¹ˣᴺ` and `JᵀJ` has rank 1: it is
**rank-deficient** for `N > 1`. LM's normal equations become singular and the method fails.
This is the "dimensionality mismatch" — the crux of the paper (§3.2).

The paper proves this matters even in conventional design: replacing the full TRA residual
with the *1-D effective spot radius* `ℓ_ESR = [‖ℓ_TRA‖]` slows LM down and yields a worse
solution (Fig. 4b) — because it throws away per-ray sensitivity. Preserving per-ray
information in `J` is what makes LM powerful.

**Everyone's workaround is SGD/Adam.** They work with a scalar loss, but (per §2 "SGD
Optimization and Limitations") they converge slowly, get stuck in local minima, and lack
scale invariance — so high-degree aspheric coefficients destabilize them. The paper's whole
motivation is to bring LM's robustness to the task-driven setting.

---

## Part 2 — The key idea: GTRA (the "lift")

GTRA (§3.3, Supp. §S3.1) constructs a **vector-valued sum-of-squares surrogate in the
spot-diagram domain that exactly matches the scalar loss's value and gradient at the current
iterate.** You start from a scalar objective and convert it into per-ray residuals while
preserving the local task gradient — restoring the dimensionality LM needs.

Let `ε₀`, `L_ε₀`, and `∇L_ε₀` be the spot diagram, scalar loss, and its gradient
*with respect to the spot diagram* at the current step (gradient via **backward-mode AD**).
A locally gradient-preserving linearization gives

$$L_{\text{TD}}(\theta) \approx \tfrac12\,\underbrace{\frac{\lVert\nabla L_{\epsilon_0}\rVert^2}{2 L_{\epsilon_0}}}_{w}\; \left\lVert\, \epsilon(\theta) - \underbrace{\left(\epsilon_0 - 2\,\frac{L_{\epsilon_0}\,\nabla L_{\epsilon_0}}{\lVert\nabla L_{\epsilon_0}\rVert^2}\right)}_{\epsilon'} \,\right\rVert^2. \tag{5 / S66}$$

So the **GTRA residual** is

$$\boxed{\;\ell_{\text{GTRA}}(\theta) = \sqrt{w}\,\big(\epsilon(\theta) - \epsilon'\big) \in \mathbb{R}^{2fwp}\;} \tag{6 / S68}$$

with the scalar weight `w` and per-ray control targets `ε'` **recomputed every LM iteration**
from the current loss and its gradient. Read the two pieces physically:

- `w = ‖∇L_ε₀‖² / (2 L_ε₀)` — a single scalar step size that makes the surrogate's value match `L_TD`.
- `ε' = ε₀ − 2 L_ε₀ ∇L_ε₀ / ‖∇L_ε₀‖²` — a per-ray **target** the rays should move toward. Unlike TRA
  (which aims every ray at the fixed centroid `ε̄`), GTRA aims each ray at a target derived from
  the task gradient, and re-aims every iteration.

**GTRA generalizes TRA.** Expanding the conventional `L = ½‖ℓ_TRA‖²` through Eq. (5) recovers
TRA exactly as the special case `w = 1/(wpf)` and `ε' = ε̄` (§3.3, "Relation to Conventional TRA").

### 2.1 Why this is *cheap* — the AD split (this is the crux for coding)

The Jacobian LM needs is `J = ∂ℓ_GTRA/∂θ`. Because `ℓ_GTRA = √w (ε(θ) − ε')` and **both `w`
and `ε'` are treated as constants** during the LM solve, the Jacobian is (up to the scalar `√w`)
just `∂ε/∂θ` — the Jacobian of the *ray tracer only*. So:

| quantity | how | cost scales with | through what part of the pipeline |
|---|---|---|---|
| `∇L_ε₀ = ∂L_TD/∂ε` | **backward-mode** AD | scalar output → cheap | the *whole* expensive pipeline (sim + restoration + loss) — once per LM step |
| `J = ∂ℓ_GTRA/∂θ ≈ √w ∂ε/∂θ` | **forward-mode** AD | `N` variables | **only the ray tracer** (`θ → ε`) — the cheap part |

Forward-mode AD is **never** applied through the expensive image simulation / neural network.
That is the entire point: a GTRA iteration costs ≈ one conventional TRA iteration (ray trace +
its forward-mode Jacobian) **plus** one forward+backward pass of the scalar loss (like one SGD
step). See Supp. §S3.1 "Jacobian matrix" and §3.3 "Derivatives".

> **Scope caveat from the paper (§1, "Scope and Limitations"):** GTRA assumes the loss depends
> on `θ` *only through the spot diagram `ε`*. Wave-optics-only simulations or detector-to-scene
> ray tracing don't expose spot diagrams and would need extra handling; extra dependencies can
> be added as additional residuals concatenated onto `ℓ_GTRA`.

---

## Part 3 — The full end-to-end pipeline, stage by stage

Data flow (Fig. 3, Fig. 6):

```
     θ (lens params)                    φ (network params)
        │                                     │
        ▼                                     │
  ┌───────────┐   ε      ┌──────────────────┐ │  I'   ┌──────────────┐  I''
  │  OPTICS   │ ───────► │      BRIDGE      │─┼─────► │  ALGORITHM   │ ──────►  L(I, I'')
  │ ray trace │  spot    │ PSF + imaging    │ │       │ restoration  │           scalar
  └───────────┘ diagrams │ sim (+ GTRA lift)│ │       │  network     │           loss
                         └──────────────────┘ │       └──────────────┘
                                               ▼
                          backward-mode AD:  ∂L/∂ε   ──►  GTRA lift  ──►  LM update on θ
                          backward-mode AD:  ∂L/∂φ   ──────────────────►  Adam update on φ
```

### 3.1 OPTICS — differentiable ray tracing → spot diagrams (Supp. §S1.1–S1.3)

Trace `n_r = f·w·p` rays from object to image plane; output the spot diagram `ε`.

- **Dispersion (Hartmann model, Eq. 8 / S4):** `n(λ) = A + C/(λ − B)`, with `A,B,C` derived
  analytically from glass variables `(n_d, v_d, ΔP_g,F)`.
- **Propagation + ray marching (Eq. S8–S10):** advance `r' = r + t·d`; solve `z̃(r+t·d) = z`
  for `t` — closed form for spherical/flat surfaces, Newton's method for aspheres.
- **Even-asphere sag (Eq. 9):** `z̃(r) = (c r²)/(1 + √(1 − (1+k)c²r²)) + Σ aₖ r²ᵏ`.
- **Refraction — vector Snell (Eq. S11–S13):** `d' = μd + (cosθ' − μ cosθ) n`, with
  `cos²θ' = 1 − μ²(1 − γ²)` (a total-internal-reflection failure when this goes negative).
- **Diffractive surfaces (Eq. S15–S18):** grating-modified refraction using the phase gradient
  `∂ϕ/∂r`, computed by AD from a continuous phase profile `ϕ(r) = Σ bₖ r²ᵏ` (Eq. 10).
- **Ray initialization + pupil sampling (Supp. §S1.3.1):** fill the entrance pupil on jittered
  concentric rings (ring `i` holds `2i−1` rays), replicate across fields and wavelengths.
- **Vignetting + ray aiming (Supp. §S1.3.2):** three vignetting factors per (field, wavelength);
  linearized ray aiming to hit the aperture-stop edge. *(Advanced — stubbed in v1.)*
- **Spot diagram + RMS spot radius (Eq. S23):** the field-wise RMS landing spread; `‖ℓ_TRA‖`.

**Validated in the paper against CODE V to ±0.001 µm** (Supp. Table S3) — this is a faithful
geometric ray tracer, not an approximation.

### 3.2 BRIDGE — spot diagrams → PSF → simulated image, and the GTRA lift (Supp. §S1.4–S1.5, §S3.1)

This is the "middle part". Two responsibilities:

**(a) Forward imaging model** (differentiable, so backward-mode AD can reach `∂L/∂ε`):

1. **Geometric PSF via KDE (Supp. §S1.4.2):** naive ray-counting into bins is
   non-differentiable, so spread each ray's energy with a **2-D triangular kernel** of support
   `σ = 2` bins. The antisymmetry of the 1-D kernel about `±σ/4` makes each ray deposit equal
   total energy regardless of sub-bin position — energy-conserving and differentiable. Output
   `K ∈ R^(f×Wₚ×Hₚ×w)`.
2. **Diffraction compensation (Supp. §S1.4.3):** each ray also carries an **Airy field pattern**
   `U_Airy(r) ∝ 2J₁(kNAr)/(kNAr)` (Eq. 12/S25); take `√(geometric PSF)` as a flat-phase field,
   convolve with the Airy field, square to get intensity. Prevents the optimizer from chasing
   physically-impossible sub-diffraction-limited PSFs. *(Advanced — stubbed in v1; v1 uses the
   geometric KDE PSF.)*
3. **PSF grid + spatially-varying convolution (Supp. §S1.5):** combine wavelengths → RGB via a
   spectral weight matrix, interpolate field PSFs across sensor regions, rotate by shears,
   overlap-add convolution with a Hann window. *(Advanced — v1 uses a single shift-invariant PSF
   per patch, which the paper notes is a valid approximation over small patches, §4.2.)*
4. **Noise (Supp. §S1.5, §4.2):** additive white Gaussian, σ = 0.5 % of max pixel value.

**(b) The GTRA lift** — `gtra_residuals(ε, L, ∇L_ε)` → `√w (ε − ε')` per Eq. (6). This is the
one function that converts the algorithm's scalar loss into optics residuals. It also handles
**control-value clipping** (Supp. §S3.1 "Bounding control values"): clip `ε'` to the PSF grid
and rescale to preserve the scalar objective.

### 3.3 ALGORITHM — image restoration model (Supp. §S3.2)

Turns the degraded capture `I'` into a restored image `I''`. The paper's IRM is
**Wiener deconvolution sandwiched between two NAFNet U-Nets** (~3M params), with the Wiener
SNR as a learnable parameter. It is trained with MAE via Adam.

- **Image-driven design is the special case where the IRM is the identity** (§4). That is the
  toy problem of Fig. 4 — no network, the optics alone must produce a good image.
- v1 ships `IdentityRestoration` (for Fig. 4) and a small `TinyUNet` (to show the network slot
  works and trains on CPU); the Wiener + dual-NAFNet IRM is a documented extension.

### 3.4 ENGINE — the LM optimizer and the joint loop (Supp. §S2.3.2, §S3.3)

**LM update (Eq. 11 / S62):** `Δθ = −(JᵀJ + λD²)⁻¹ Jᵀℓ₀`, with

- **Damping matrix `D²`** = diagonal of `JᵀJ` for scale invariance, stabilized by a
  **running maximum** (Eq. S64): `D_{k+1} = β·max(D_k, diag(JᵀJ)^½) + (1−β)·diag(JᵀJ)^½`,
  initialized with a floor `ε = 1e−6` (Eq. S65). Prevents "parameter evaporation".
- **Adaptive damping factor `λ`** (§3.5, Supp. §S2.3.2): start at `λ₀ = 1`; on a **loss
  decrease** divide by `DF = 3` (toward Gauss–Newton); on a **loss increase** multiply by
  `IF = 2` (toward gradient descent).
- **Step rejection** (tolerance `TF = 1`): accept a step iff `L(θ+Δθ) < L(θ)`. This is what makes
  the LM convergence curves monotonic (Fig. 4b,c). Attempted steps can blow the loss up by orders
  of magnitude, so this is essential.
- **Jacobian by forward-mode AD** (`torch.func.jacfwd`) — scales with `N`, not `M`.

**Joint optimization step (Supp. §S3.3.1)** — for each training iteration:
1. Sample a minibatch of image patches + sensor regions (pseudo-random region sampling for
   balanced FOV coverage).
2. **Lens step:** freeze IRM; compute spot diagrams (with forward-mode AD), run the imaging sim +
   restoration + loss, get `∇L_ε` by backward-mode AD, lift to GTRA, take **one LM step** on `θ`.
3. **IRM step:** freeze lens; take **one Adam step** on `φ` (MAE loss).
4. Repeat. The IRM co-adapts to the lens and vice-versa.

---

## Part 4 — Architecture: how the paper maps onto the code

The design goal you gave: **three swappable parts + an optimizer, simple enough for a
non-software-engineer to extend.** The package uses one small idea — an **abstract base class
(ABC) per stage that fixes the tensor contract** — so any concrete implementation that honors
the contract drops in without touching the rest.

```
e2e_optics/
├── pyproject.toml            # pip-installable
├── e2e_optics/
│   ├── config.py             # one dataclass; picks optics/bridge/algorithm by name
│   ├── optics/
│   │   ├── base.py           # BaseOptics ABC:  forward(θ) -> SpotDiagram
│   │   └── raytrace.py       # RotationallySymmetricLens (asphere + Snell + Hartmann)
│   ├── bridge/
│   │   ├── base.py           # BaseBridge ABC:  simulate(spot, scene) -> I'
│   │   ├── kde_psf.py        # differentiable KDE geometric PSF
│   │   ├── imaging.py        # convolution + noise imaging model
│   │   └── gtra.py           # gtra_residuals(ε, L, ∇L_ε) -> residual   ← THE LIFT
│   ├── algorithm/
│   │   ├── base.py           # BaseRestoration ABC:  restore(I') -> I''
│   │   ├── identity.py       # IdentityRestoration (image-driven design)
│   │   └── unet.py           # TinyUNet + Wiener stub
│   └── engine/
│       ├── lm.py             # LevenbergMarquardt (forward-mode Jacobian, damping, rejection)
│       ├── constraints.py    # soft-constraint residuals (airspace, angles, …)
│       └── joint.py          # alternating LM(lens) / Adam(IRM) end-to-end loop
├── demos/
│   └── demo_toy_e2e.py       # reproduces Fig. 4 toy 2-element E2E design
└── tests/                    # gradient checks, energy conservation, LM on known minimum
```

### 4.1 The three contracts (what makes parts swappable)

| Stage | ABC | Input → Output | Units / conventions |
|---|---|---|---|
| **Optics** | `BaseOptics.forward()` | `θ` (params, held internally) → `SpotDiagram` with `.xy` shape `(f, w, p, 2)` | mm at image plane; +z along optical axis; fields as angles |
| **Bridge** | `BaseBridge.simulate()` | `SpotDiagram`, scene `I` `(C,H,W)` → capture `I'` `(C,H,W)` | intensity in [0,1]; PSF energy-normalized |
| **Algorithm** | `BaseRestoration.restore()` | `I'` `(C,H,W)` [, PSF] → `I''` `(C,H,W)` | same intensity convention |
| **Lift** (bridge) | `gtra_residuals()` | `ε` `(2fwp,)`, scalar `L`, `∇L_ε` `(2fwp,)` → residual `(2fwp,)` | pure function of the three inputs |
| **Engine** | `LevenbergMarquardt.step()` | residual fn `θ→ℓ`, current `θ` → new `θ` | operates on a flat `θ` vector |

Because the contract is *only* these tensor shapes, you can:
- **swap the optics** — replace `RotationallySymmetricLens` with a metasurface/DOE model, as long
  as it emits a `SpotDiagram` (or, per the paper's caveat, add auxiliary residuals);
- **swap the bridge** — replace KDE with the diffraction-compensated PSF, or replace
  shift-invariant convolution with the spatially-varying overlap-add — `gtra.py` is untouched;
- **swap the algorithm** — drop in the Wiener + dual-NAFNet IRM, or a detection head for a
  perception task, without touching optics or engine.

### 4.2 Where "all that good stuff" you mentioned lives

| Your words | Paper concept | Code location |
|---|---|---|
| "the optics part" | differentiable ray tracing, spot diagrams | `optics/raytrace.py` |
| "how data is transferred … the middle part" | PSF estimation + imaging simulation | `bridge/kde_psf.py`, `bridge/imaging.py` |
| "where the GTRA is implemented" | the scalar→vector lift | `bridge/gtra.py` |
| "the spot diagram implementation" | `ε(θ)` | `optics/raytrace.py` → `SpotDiagram` |
| "the Jacobian … all that good stuff" | forward-mode `∂ℓ/∂θ` | `engine/lm.py` |
| "the algorithm part … different neural networks" | image restoration model | `algorithm/` |
| "connected by backpropagation" | AD split (fwd for J, bwd for ∇L_ε) + alternating LM/Adam | `engine/joint.py` |

---

## Part 5 — v1 scope: what runs, what is stubbed

**Implemented and validated in v1 (the runnable slice):**
- Rotationally-symmetric ray tracer: even-asphere sag, vector Snell, Hartmann dispersion,
  concentric-ring pupil sampling → spot diagrams.
- Differentiable KDE geometric PSF (triangular kernel, energy-conserving).
- Shift-invariant convolution imaging model + additive Gaussian noise.
- GTRA lift with control-value clipping; verified to reduce to TRA and to match the scalar
  gradient to first order.
- Full LM engine: forward-mode Jacobian, `diag(JᵀJ)` damping with running-maximum
  stabilization, adaptive `λ`, step rejection.
- `IdentityRestoration` (image-driven) and `TinyUNet`.
- Alternating LM/Adam joint loop.
- **Fig. 4 toy reproduction:** conventional TRA design vs task-driven GTRA design on the
  concentric-ring chart; GTRA yields a larger spot radius but higher task PSNR.

**Stubbed behind interfaces (documented extension points) in v1:**
- Optimizable vignetting + full ray aiming (`optics/raytrace.py` has the hooks).
- Diffraction-compensated PSF (`bridge/kde_psf.py` documents the Airy-field recipe).
- PSF grid + spatially-varying overlap-add convolution (`bridge/imaging.py`).
- Glass-catalog mesh constraints (`engine/constraints.py`).
- Wiener + dual-NAFNet IRM (`algorithm/unet.py`).

---

## Part 6 — Extension guide (how to grow each part)

1. **A new optics model.** Subclass `BaseOptics`, register trainable params in `__init__`,
   implement `forward()` to return a `SpotDiagram`. If your model doesn't produce spot diagrams
   (pure wave optics), either emit an equivalent sampling or add auxiliary residuals in
   `engine/constraints.py` per the paper's §1 caveat.
2. **A new bridge / PSF model.** Subclass `BaseBridge`. The imaging model must stay
   differentiable end-to-end so backward-mode AD can produce `∂L/∂ε`. The GTRA lift in
   `bridge/gtra.py` is model-agnostic and needs no change.
3. **A new restoration / task network.** Subclass `BaseRestoration`. For a *perception* task
   (detection, segmentation) swap the loss in the joint loop — GTRA is loss-agnostic, so the
   only requirement is that the loss is a differentiable scalar of the simulated capture.
4. **New constraints.** Add residual functions in `engine/constraints.py`; they concatenate onto
   `ℓ_GTRA` and are picked up by LM automatically.
5. **Tuning LM.** All LM knobs (`λ₀, β, ε, IF, DF, TF`) are constructor arguments with the
   paper's defaults (Supp. Table S7).

---

## Appendix — equation-to-code cross-reference

| Eq. | Meaning | Function |
|---|---|---|
| (8)/S4 | Hartmann dispersion `n(λ)` | `optics.raytrace.hartmann_index` |
| (9) | even-asphere sag | `optics.raytrace.asphere_sag` |
| S8–S10 | propagation + Newton ray march | `optics.raytrace._march_to_surface` |
| S12–S13 | vector Snell refraction | `optics.raytrace.refract` |
| S15–S18 | grating-modified refraction (DOE) | `optics.raytrace.diffract` (stub) |
| S23 | RMS spot radius | `optics.raytrace.SpotDiagram.rms_radius` |
| S25/(12) | Airy field pattern | `bridge.kde_psf.airy_field` (stub) |
| (3) | TRA residual | `bridge.gtra.tra_residuals` |
| (6)/S66/S68 | **GTRA residual (the lift)** | `bridge.gtra.gtra_residuals` |
| (11)/S62 | LM update | `engine.lm.LevenbergMarquardt.step` |
| S64/S65 | damping-matrix running max | `engine.lm.LevenbergMarquardt._update_damping` |
| S3.3.1 | joint LM/Adam step | `engine.joint.JointOptimizer.step` |
