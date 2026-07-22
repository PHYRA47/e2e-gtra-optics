"""Identity restoration -- the image-driven design special case.

Cote et al. (2026), sec. 4: image-driven design is the special case of
end-to-end design where the image restoration model (IRM) is the IDENTITY.
The optics alone must produce a good image; the task loss is computed directly
on the simulated capture. This is exactly the Fig. 4 toy problem.

Having it as a first-class ``BaseRestoration`` means the same joint loop drives
both image-driven and full end-to-end design -- you swap this for a network.
"""
from __future__ import annotations
from typing import Optional
import torch
from .base import BaseRestoration


class IdentityRestoration(BaseRestoration):
    conditioned = False

    def restore(self, degraded: torch.Tensor,
                psf: Optional[torch.Tensor] = None) -> torch.Tensor:
        return degraded

