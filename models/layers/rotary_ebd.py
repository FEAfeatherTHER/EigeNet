from __future__ import annotations
from typing import Callable

import math
from copy import deepcopy
from random import random, randrange
from functools import partial, wraps
from itertools import chain
from collections import namedtuple
from contextlib import nullcontext
from dataclasses import dataclass
from packaging import version

import torch
from torch.amp import autocast
import torch.nn.functional as F
from torch import nn, einsum, tensor, Tensor, cat, stack, arange, is_tensor
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map
from torch.nn import Module, ModuleList, ModuleDict

from loguru import logger

from x_transformers.attend import Attend, Intermediates
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper

import einx
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, reduce, pack, unpack

def exists(val):
    return val is not None

class RotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        use_xpos = False,
        scale_base = 512,
        interpolation_factor = 1.,
        base = 10000,
        base_rescale_factor = 1.
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        base *= base_rescale_factor ** (dim / (dim - 2))

        inv_freq = 1. / (base ** (arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        assert interpolation_factor >= 1.
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.register_buffer('scale', None)
            return

        scale = (arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.register_buffer('scale', scale)

    def forward_from_mod_seq_lens(self, mod_seq_lens):
        mod0_seq_len, mod1_seq_len = mod_seq_lens
        freqs0, scale0 = self.forward_from_seq_len(mod0_seq_len)
        freqs1, scale1 = self.forward_from_seq_len(mod1_seq_len)
        freqs = torch.cat([freqs0, freqs1], dim = 1)
        if scale0 == scale1:
            scale = scale0
        else:
            import random
            random.seed(114)
            scale = random.choice([scale0, scale1])
        return freqs, scale

    def forward_from_seq_len(self, seq_len):
        device = self.inv_freq.device

        t = arange(seq_len, device = device)
        return self.forward(t)

    @autocast('cuda', enabled = False)
    def forward(self, t, offset = 0):
        max_pos = t.max() + 1
        
        if t.ndim == 1:
            t = rearrange(t, 'n -> 1 n')

        freqs = torch.einsum('b i , j -> b i j', t.type_as(self.inv_freq), self.inv_freq) / self.interpolation_factor
        freqs = stack((freqs, freqs), dim = -1)
        freqs = rearrange(freqs, '... d r -> ... (d r)')

        if not exists(self.scale):
            return freqs, 1.

        power = (t - (max_pos // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, '... n -> ... n 1')
        scale = stack((scale, scale), dim = -1)
        scale = rearrange(scale, '... d r -> ... (d r)')

        return freqs, scale
