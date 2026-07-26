from .base import BaseOptics, SpotDiagram
from .raytrace import RotationallySymmetricOptics, Surface
__all__ = ["BaseOptics", "SpotDiagram", "RotationallySymmetricOptics", "Surface"]

# --- Backward-compatible aliases -----------------------------------------
# The API was renamed to be element-agnostic: a `RotationallySymmetricOptics`
# is any rotationally symmetric stack of refractive OR diffractive surfaces,
# not specifically a lens. The old names still resolve so existing scripts and
# notebooks keep working.
RotationallySymmetricLens = RotationallySymmetricOptics
__all__ += ["RotationallySymmetricLens"]
