"""e2e_optics -- modular PyTorch framework for GTRA-based end-to-end
optics-algorithm co-design.

Implements the three-part pipeline of Cote, Tseng & Heide,
"Generalized Aberrations for Processing-Aware Optical Design" (ACM TOG 2026):

    optics  ->  bridge  ->  algorithm
    (rays)      (PSF +       (restoration
                 GTRA lift)   network)

tied together by a Levenberg-Marquardt engine (lens) + Adam (network).

Each stage is an abstract base class with a fixed tensor contract, so any
concrete implementation that honors the contract is swappable without
touching the rest. See docs/ANALYSIS_AND_DESIGN.md.
"""

# --- Sandbox / restricted-container OpenMP guard -------------------------
# Some containers disallow pthread_setaffinity_np(); the OpenMP runtime that
# ships with numpy/torch aborts (OMP Error #179) unless affinity binding is
# turned off. Set these BEFORE numpy/torch import anywhere in the process.
import os as _os
_os.environ.setdefault("KMP_AFFINITY", "disabled")
_os.environ.setdefault("OMP_PROC_BIND", "false")
_os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
_os.environ.setdefault("OMP_NUM_THREADS", "4")
# -------------------------------------------------------------------------

from .optics.base import BaseOptics, SpotDiagram
from .optics.raytrace import RotationallySymmetricLens, Surface
from .bridge.base import BaseBridge
from .bridge.kde_psf import kde_psf
from .bridge.imaging import ConvolutionImaging
from .bridge.gtra import gtra_residuals, tra_residuals
from .algorithm.base import BaseRestoration
from .algorithm.identity import IdentityRestoration
from .algorithm.unet import TinyUNet
from .engine.lm import LevenbergMarquardt
from .engine.joint import JointOptimizer
from .config import PipelineConfig, build_pipeline

__version__ = "0.1.0"

__all__ = [
    "BaseOptics", "SpotDiagram", "RotationallySymmetricLens", "Surface",
    "BaseBridge", "kde_psf", "ConvolutionImaging",
    "gtra_residuals", "tra_residuals",
    "BaseRestoration", "IdentityRestoration", "TinyUNet",
    "LevenbergMarquardt", "JointOptimizer",
    "PipelineConfig", "build_pipeline",
]
