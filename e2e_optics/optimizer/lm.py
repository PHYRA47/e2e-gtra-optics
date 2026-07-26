"""Levenberg-Marquardt optimizer with a forward-mode-AD Jacobian.

Solves min_theta 1/2 ||l(theta)||^2 by the damped Gauss-Newton update
(Cote et al. 2026, Eq. 11 / Supp. S62):

    delta = -(J^T J + lambda D^2)^{-1} J^T l0

with the paper's stabilizations (Supp. S2.3.2):
  * D^2 = diag(J^T J), the Marquardt scale-invariant damping, kept as a
    RUNNING MAXIMUM (Eq. S64) with a floor eps (Eq. S65) so parameters that
    momentarily have small sensitivity don't "evaporate";
  * ADAPTIVE lambda: divide by DF on an accepted (loss-decreasing) step
    (-> Gauss-Newton), multiply by IF on a rejected step (-> gradient descent);
  * STEP REJECTION with tolerance TF=1: accept iff the new loss < old loss.
    Rejected steps are undone; this makes the convergence curve monotonic.

The Jacobian J = d l / d theta is computed with FORWARD-mode AD
(torch.func.jacfwd), whose cost scales with N (few optics variables), not M
(thousands of residuals). For GTRA, the residual function passed here should
close over the (constant) weight w and target eps', so that J is effectively
sqrt(w) * d(spot diagram)/d theta -- the ray-tracer Jacobian only.

Parameter defaults follow Supp. Table S7.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional
import torch
import torch.func as tf


@dataclass
class LMState:
    theta: torch.Tensor
    loss: float
    lam: float
    damping: torch.Tensor          # sqrt(diag(J^T J)) running-max (length N)
    accepted: bool
    iteration: int


class LevenbergMarquardt:
    def __init__(self,
                 residual_fn: Callable[[torch.Tensor], torch.Tensor],
                 theta0: torch.Tensor,
                 lam0: float = 1.0,
                 inc_factor: float = 2.0,     # IF -- multiply lambda on reject
                 dec_factor: float = 3.0,     # DF -- divide lambda on accept
                 tol_factor: float = 1.0,     # TF -- accept iff new<old
                 beta: float = 0.99,          # running-max mixing (Eq. S64)
                 floor: float = 1e-6,         # damping floor eps (Eq. S65)
                 lam_min: float = 1e-9,
                 lam_max: float = 1e9):
        """
        residual_fn : theta (N,) -> residual l (M,). Must be differentiable via
                      torch.func.jacfwd (i.e. a pure function of theta).
        theta0      : initial parameter vector (N,), any dtype (float64 advised).
        """
        self.residual_fn = residual_fn
        self.theta = theta0.clone()
        self.lam = float(lam0)
        self.IF, self.DF, self.TF = inc_factor, dec_factor, tol_factor
        self.beta, self.floor = beta, floor
        self.lam_min, self.lam_max = lam_min, lam_max
        self.N = theta0.numel()
        self.damping = torch.full((self.N,), floor, dtype=theta0.dtype)
        self.history: List[float] = []
        r0 = residual_fn(self.theta)
        self.loss = 0.5 * float(r0.dot(r0))
        self.history.append(self.loss)

    @staticmethod
    def _loss(r: torch.Tensor) -> float:
        return 0.5 * float(r.dot(r))

    def _update_damping(self, diagJTJ: torch.Tensor):
        """Running-maximum damping (Eq. S64) with floor (Eq. S65)."""
        cur = torch.sqrt(diagJTJ.clamp_min(0.0)).clamp_min(self.floor)
        self.damping = self.beta * torch.maximum(self.damping, cur) + (1 - self.beta) * cur

    def step(self) -> LMState:
        """One LM iteration: compute J, propose a step, accept/reject, adapt lambda."""
        theta = self.theta
        r0 = self.residual_fn(theta)
        loss0 = self._loss(r0)
        J = tf.jacfwd(self.residual_fn)(theta)         # (M,N) forward-mode
        JT = J.t()
        JTJ = JT @ J                                   # (N,N)
        g = JT @ r0                                    # (N,)
        self._update_damping(torch.diagonal(JTJ))
        D2 = torch.diag(self.damping ** 2)

        accepted = False
        it_lam = self.lam
        # try steps, increasing lambda until the loss decreases (or give up)
        for _ in range(30):
            A = JTJ + it_lam * D2
            try:
                delta = torch.linalg.solve(A, -g)
            except RuntimeError:
                delta = -torch.linalg.lstsq(A, g).solution
            theta_new = theta + delta
            r_new = self.residual_fn(theta_new)
            loss_new = self._loss(r_new)
            if loss_new < self.TF * loss0:             # accept
                self.theta = theta_new
                self.loss = loss_new
                self.lam = max(it_lam / self.DF, self.lam_min)
                accepted = True
                break
            else:                                      # reject: more damping
                it_lam = min(it_lam * self.IF, self.lam_max)
        if not accepted:
            self.lam = it_lam                          # keep the raised damping
            self.loss = loss0
        self.history.append(self.loss)
        return LMState(self.theta.clone(), self.loss, self.lam,
                       self.damping.clone(), accepted, len(self.history) - 1)

    def run(self, n_iters: int, verbose: bool = False,
            callback: Optional[Callable[[LMState], None]] = None) -> torch.Tensor:
        for _ in range(n_iters):
            st = self.step()
            if verbose:
                print(f"  LM it {st.iteration:3d}  loss={st.loss:.6e}  "
                      f"lam={st.lam:.2e}  {'acc' if st.accepted else 'REJ'}")
            if callback is not None:
                callback(st)
        return self.theta

