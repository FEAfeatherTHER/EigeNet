"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""
# ruff: noqa: F722 F821

from __future__ import annotations
from einops import rearrange
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from models.layers.rotary_ebd import RotaryEmbedding
#from x_transformers.x_transformers import RotaryEmbedding
from models.layers.modules import (
    AdaLayerNorm_Final,
    ConvPositionEmbedding,
    DiTBlock,
    TimestepEmbedding,
    precompute_freqs_cis,
)

class AADiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        aa_order_list = ['f', 'g', 'f', 'g', 'f', 'g', 'f', 'g'],
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        latent_dim=100,
        qk_norm=None,
        pe_attn_head=None,
        attn_backend="torch",  # "torch" | "flash_attn"
        attn_mask_enabled=False,
        long_skip_connection=False,
        checkpoint_activations=False,
    ):
        super().__init__()
        assert len(aa_order_list) == depth, "The length of aa_order_list must be equal to the depth of the model"
        self.aa_order_list = aa_order_list
        self.time_embed = TimestepEmbedding(dim)
        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    pe_attn_head=pe_attn_head,
                    attn_backend=attn_backend,
                    attn_mask_enabled=attn_mask_enabled,
                )
                for _ in range(depth)
            ]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None
        self.norm_out = AdaLayerNorm_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim * 2, latent_dim)
        self.proj_in = nn.Linear(latent_dim, dim)

        self.checkpoint_activations = checkpoint_activations

        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out AdaLN layers in DiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def ckpt_wrapper(self, module):
        # https://github.com/chuanyangjin/fast-DiT/blob/main/models.py
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def _process_frame_attention(self, x, ebd, attn_mask, B, S, P, C, idx):
        """
        Process frame attention blocks.
        """
        # If needed, reshape tokens:
        if x.shape != (B * S, P, C):
            x = x.view(B, S, P, C).view(B * S, P, C)
        if attn_mask.shape != (B * S, P):
            frame_attn_mask = attn_mask.view(B, S, P).view(B * S, P)
        if ebd.shape[0] != B * S:
            frame_ebd = ebd.repeat(S, 1)

        intermediates = []
        seq_len = x.shape[1]
        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        x = self.transformer_blocks[idx](x, frame_ebd, mask=frame_attn_mask, rope=rope)
        idx += 1
        intermediates.append(x.view(B, S, P, C))

        return x, idx, intermediates
    
    def _process_global_attention(self, x, ebd, attn_mask, B, S, P, C, idx):
        """
        Process glbal attention blocks.
        """
        # If needed, reshape tokens:
        if x.shape != (B, S*P, C):
            x = x.view(B, S, P, C).view(B, S*P, C)

        intermediates = []
        seq_len = x.shape[1]
        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        x = self.transformer_blocks[idx](x, ebd, mask=attn_mask, rope=rope)
        idx += 1
        intermediates.append(x.view(B, S, P, C))

        return x, idx, intermediates


    def forward(
        self,
        x: float["b n t d"],  # nosied input 
        time: float["b"] | float[""],  # time step
        attn_mask: bool["b n"] | None = None,
    ):
        B, S, P, C = x.shape
        if time.ndim == 0:
            time = time.repeat(B)
        t_embedding = self.time_embed(time)

        embedding = t_embedding
       
        x = self.proj_in(x)
        if self.long_skip_connection is not None:
            residual = rearrange(x, "b n t d -> b (n t) d")
        
        idx = 0
        for attn_type in  self.aa_order_list:
            if attn_type == 'f':
                x, idx, frame_intermediates = self._process_frame_attention(x, embedding, attn_mask, B, S, P, C, idx)
            elif attn_type == 'g':
                x, idx, global_intermediates = self._process_global_attention(x, embedding, attn_mask, B, S, P, C, idx)
            else:
                raise ValueError(f"Invalid attention type: {attn_type}")
        if 'f' in self.aa_order_list:
            last_frame_x = rearrange(frame_intermediates[-1], "b n t c -> b (n t) c", n = S) # (B, S*P, C)
        if 'g' in self.aa_order_list:
            last_global_x = rearrange(global_intermediates[-1], "b s p c -> b (s p) c", s = S) # (B, S*P, C)
        del frame_intermediates, global_intermediates
        if self.long_skip_connection is not None:
            last_frame_x  = self.long_skip_connection(torch.cat((last_frame_x, residual), dim=-1))
            last_global_x = self.long_skip_connection(torch.cat((last_global_x, residual), dim=-1))

        last_frame_x = self.norm_out(last_frame_x, embedding)
        last_global_x = self.norm_out(last_global_x, embedding)
        x = torch.cat((last_frame_x, last_global_x), dim=-1)
        output = self.proj_out(x)
        return output
