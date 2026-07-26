"""One small config surface so the three parts are swapped *by name*.

    cfg = PipelineConfig(optics="rotational_symmetric", bridge="kde_conv",
                         algorithm="identity")
    optics, bridge, algo = build_pipeline(cfg, optics_kwargs={...}, ...)

Register a new implementation by adding it to the relevant registry dict below;
no other code changes are needed. This keeps the "swap a part" workflow to a
single string change in a config.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict


# --- registries: name -> class -----------------------------------------------
def _optics_registry():
    from .optics.raytrace import RotationallySymmetricOptics
    return {"rotational_symmetric": RotationallySymmetricOptics}


def _bridge_registry():
    from .bridge.imaging import ConvolutionImaging
    return {"kde_conv": ConvolutionImaging}


def _algorithm_registry():
    from .algorithm.identity import IdentityRestoration
    from .algorithm.unet import TinyUNet
    return {"identity": IdentityRestoration, "tiny_unet": TinyUNet}


@dataclass
class PipelineConfig:
    optics: str = "rotational_symmetric"
    bridge: str = "kde_conv"
    algorithm: str = "identity"
    optics_kwargs: Dict[str, Any] = field(default_factory=dict)
    bridge_kwargs: Dict[str, Any] = field(default_factory=dict)
    algorithm_kwargs: Dict[str, Any] = field(default_factory=dict)


def build_pipeline(cfg: PipelineConfig):
    """Instantiate (optics, bridge, algorithm) from a config."""
    optics = _optics_registry()[cfg.optics](**cfg.optics_kwargs)
    bridge = _bridge_registry()[cfg.bridge](**cfg.bridge_kwargs)
    algo = _algorithm_registry()[cfg.algorithm](**cfg.algorithm_kwargs)
    return optics, bridge, algo
