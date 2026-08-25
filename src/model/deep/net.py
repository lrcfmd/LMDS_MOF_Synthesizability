"""Deep SVDD network architecture for the synthesizability model.

Vendored verbatim (architecture only) from the original tool's
``app/Encoder.py`` + ``Deep-SVDD-PyTorch/src/base/base_net.py`` so the module
structure — and therefore the ``state_dict`` keys — exactly match the saved
checkpoint (``artifacts/deep_model.tar``). Only the pieces needed for inference
are kept; the training machinery and autoencoder are dropped.
"""

import logging

import torch
import torch.nn as nn

from .modules import PMA, SAB

# Input feature dimension: 205 scaled metal descriptors + 2048-bit Morgan
# fingerprint of the linker.
INPUT_DIM = 2253


class BaseNet(nn.Module):
    """Base class for all neural networks (from Deep-SVDD-PyTorch)."""

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.rep_dim = None

    def forward(self, *input):
        raise NotImplementedError


class PairsEncoder(BaseNet):
    """Set-Transformer encoder mapping a (metal, linker) pair to a 100-d point."""

    def __init__(self):
        super().__init__()
        self.rep_dim = 100
        self.seq = nn.Sequential(
            SAB(dim_in=INPUT_DIM, dim_out=1000, num_heads=10),
            SAB(dim_in=1000, dim_out=500, num_heads=10),
            SAB(dim_in=500, dim_out=100, num_heads=10),
            PMA(dim=100, num_heads=5, num_seeds=1),
        )

    def forward(self, x):
        x = torch.split(x, INPUT_DIM, dim=1)
        x = torch.stack(x).transpose(0, 1)
        return self.seq(x).squeeze()
