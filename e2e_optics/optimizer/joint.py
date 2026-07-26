"""Joint end-to-end optimizer -- the loop that ties all three parts together.

Implements the alternating scheme of Cote et al. (2026), Supp. S3.3.1:

  For each outer iteration:
    (1) OPTICS STEP  -- optics params theta by GTRA + Levenberg-Marquardt,
                      with the restoration network FROZEN.
    (2) ALGO STEP  -- restoration network params by Adam (SGD-family),
                      with the optics FROZEN.

The GTRA optics step is where the paper's key AD split lives:

  a. Forward the WHOLE pipeline once at the current theta, compute the scalar
     task loss L, and get grad_L = dL/d(eps) by BACKWARD-mode AD. eps is the
     spot diagram; the graph runs eps -> PSF -> simulate -> restore -> loss.
     This is the ONE expensive backward pass per optics step.
  b. Freeze (w, eps') from (L, grad_L) -- the GTRA lift (bridge.gtra).
  c. Run LM on residual(theta) = gtra_residuals(spot(theta), L, grad_L). Its
     Jacobian J = sqrt(w) d eps/d theta is taken by FORWARD-mode AD through the
     RAY TRACER ONLY -- image sim and network are not in this graph.

Because the LM residual only re-traces rays (cheap) while the task gradient is
computed once by backward-mode, we get LM's fast convergence at SGD-like
per-iteration cost. When the restoration network is ``IdentityRestoration`` this
reduces to image-driven design (the Fig. 4 toy).

The class is intentionally small and readable -- it is the file a contributor
will read first to understand how the three parts connect.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional
import torch

from ..optics.base import BaseOptics
from ..bridge.base import BaseBridge
from ..algorithm.base import BaseRestoration
from ..bridge.gtra import gtra_residuals
from .lm import LevenbergMarquardt


@dataclass
class JointConfig:
    lm_iters_per_step: int = 3        # inner LM iterations per outer optics step
    adam_lr: float = 1e-3
    adam_steps_per_step: int = 1      # inner Adam steps per outer algo step
    optics_step: bool = True            # set False to freeze optics (train net only)
    algo_step: bool = True            # set False to freeze net (pure optics design)
    grid_half_extent: Optional[torch.Tensor] = None   # PSF-grid clip for eps'
    verbose: bool = False


@dataclass
class JointHistory:
    task_loss: List[float] = field(default_factory=list)
    optics_esr_um: List[float] = field(default_factory=list)


class JointOptimizer:
    def __init__(self,
                 optics: BaseOptics,
                 bridge: BaseBridge,
                 algorithm: BaseRestoration,
                 loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 scene_sampler: Callable[[], "tuple[torch.Tensor, torch.Tensor]"],
                 config: Optional[JointConfig] = None):
        """
        optics        : BaseOptics with differentiable spot_from_theta.
        bridge        : BaseBridge (PSF + simulate). Must expose .psf(spot) and
                        .simulate(spot, scene) using a spot diagram.
        algorithm     : BaseRestoration (nn.Module for a trainable IRM, or
                        IdentityRestoration for image-driven design).
        loss_fn       : (restored, target) -> scalar task loss.
        scene_sampler : callable -> (scene, target), each (C,H,W) in [0,1].
        """
        self.optics = optics
        self.bridge = bridge
        self.algorithm = algorithm
        self.loss_fn = loss_fn
        self.scene_sampler = scene_sampler
        self.cfg = config or JointConfig()
        self.history = JointHistory()

        net_params = [p for p in getattr(algorithm, 'parameters', lambda: [])()]
        self._has_net = len(net_params) > 0
        self.opt = torch.optim.Adam(net_params, lr=self.cfg.adam_lr) if self._has_net else None

    # ------------------------------------------------------------------ #
    #  the full differentiable forward: theta -> task loss                #
    # ------------------------------------------------------------------ #
    def _task_loss_from_spot(self, spot, scene, target):
        """eps (spot) -> PSF -> capture -> restore -> loss. Differentiable."""
        capture = self.bridge.simulate(spot, scene, add_noise=True)
        restored = self.algorithm.restore(capture, psf=None)
        return self.loss_fn(restored, target)

    def _optics_step(self, scene, target):
        """One GTRA+LM optics step (network frozen)."""
        optics = self.optics
        theta = optics.get_theta().detach().clone().requires_grad_(True)

        # --- (a) whole-pipeline forward + backward-mode grad wrt eps ---
        spot = optics.spot_from_theta(theta)          # (2FWP,) carries grad to theta
        eps0 = spot.detach()
        # we need dL/d eps, so make a leaf eps and rebuild the spot object from it
        eps_leaf = eps0.clone().requires_grad_(True)
        spot_obj = optics.spot_object_from_flat(eps_leaf)
        L = self._task_loss_from_spot(spot_obj, scene, target)
        grad_L, = torch.autograd.grad(L, eps_leaf)    # backward-mode, ONE pass
        L_val = float(L.detach())

        # --- (b,c) GTRA lift frozen, LM with forward-mode ray-tracer Jacobian ---
        def residual(th):
            eps = optics.spot_from_theta(th)
            return gtra_residuals(eps, L_val, grad_L, eps0=eps0,
                                  grid_half_extent=self.cfg.grid_half_extent)
        lm = LevenbergMarquardt(residual, theta.detach())
        theta_new = lm.run(self.cfg.lm_iters_per_step)
        optics.set_theta(theta_new.detach())
        return L_val

    def _algo_step(self, scene, target):
        """One Adam step on the restoration network (optics frozen)."""
        if not self._has_net:
            return None
        self.opt.zero_grad()
        with torch.no_grad():
            spot = self.optics.forward()              # optics frozen
        capture = self.bridge.simulate(spot, scene, add_noise=True)
        restored = self.algorithm.restore(capture, psf=None)
        loss = self.loss_fn(restored, target)
        loss.backward()
        self.opt.step()
        return float(loss.detach())

    # ------------------------------------------------------------------ #
    def step(self):
        scene, target = self.scene_sampler()
        L_optics = self._optics_step(scene, target) if self.cfg.optics_step else None
        L_algo = self._algo_step(scene, target) if self.cfg.algo_step else None
        # record the most task-relevant loss available
        L_rec = L_algo if L_algo is not None else L_optics
        if L_rec is not None:
            self.history.task_loss.append(L_rec)
        esr = self.optics.forward().effective_spot_radius().item() * 1000.0
        self.history.optics_esr_um.append(esr)
        return {"optics_loss": L_optics, "algo_loss": L_algo, "esr_um": esr}

    def run(self, n_iters: int):
        for i in range(n_iters):
            info = self.step()
            if self.cfg.verbose:
                ll = info['optics_loss']; al = info['algo_loss']
                print(f"iter {i:3d}  "
                      f"optics_loss={ll:.4e}  " if ll is not None else f"iter {i:3d}  "
                      + (f"algo_loss={al:.4e}  " if al is not None else "")
                      + f"ESR={info['esr_um']:.1f}um")
        return self.history

