from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from models.layers.modules import (
    RMSNorm,
    FeedForward,
    Attention,
    AttnProcessor
)
from einops import rearrange
from x_transformers.x_transformers import RotaryEmbedding

class Transformer_Block(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head,
        ff_mult=4,
        dropout=0.1,
        qk_norm=None,
        pe_attn_head=None,
        attn_backend="torch",  # "torch" or "flash_attn"
        attn_mask_enabled=True,
        return_attn=False,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(dim, eps=1e-6)
        self.return_attn = return_attn
        self.attn = Attention(
            processor=AttnProcessor(
                pe_attn_head=pe_attn_head,
                attn_backend=attn_backend,
                attn_mask_enabled=attn_mask_enabled,
                return_attn_weight=self.return_attn,
            ),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            qk_norm=qk_norm,
        )
        self.ff_norm = RMSNorm(dim, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, mask=None, rope=None):
        if self.return_attn:
            attn_out, attn_weight = self.attn(self.attn_norm(x), mask=mask, rope=rope)
        else:
            attn_out = self.attn(self.attn_norm(x), mask=mask, rope=rope)
        x = x + attn_out
        x = x + self.ff(self.ff_norm(x))
        if self.return_attn:
            return x, attn_weight
        else:
            return x

class Transformer_Decoder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=4,
        heads=4,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        #latent_dim=64,
        qk_norm=None,
        conv_layers=0,
        pe_attn_head=None,
        attn_backend="torch",  # "torch" | "flash_attn"
        attn_mask_enabled=False,
        checkpoint_activations=False,
        return_attn= False,
    ):
        super().__init__()
        self.rotary_embed = RotaryEmbedding(dim_head)
        self.dim = dim
        self.depth = depth
        self.return_attn = return_attn
        self.transformer_blocks = nn.ModuleList(
            [
                Transformer_Block(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    pe_attn_head=pe_attn_head,
                    attn_backend=attn_backend,
                    attn_mask_enabled=attn_mask_enabled,
                    return_attn = self.return_attn,
                )
                for _ in range(depth)
            ]
        )
        #self.proj_out = nn.Linear(dim, latent_dim)
        self.checkpoint_activations = checkpoint_activations
        

    def forward(self, x, mask=None, rope=None):
        batch, seq_len = x.shape[0], x.shape[1]
        rope = self.rotary_embed.forward_from_seq_len(seq_len)
        #x = self.proj_in(x)
        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, mask, rope, use_reentrant=False)
            else:
                if self.return_attn:
                    x, attn_weight = block(x, mask=mask, rope=rope)
                else:
                    x = block(x, mask=mask, rope=rope)
        #x = self.proj_out(x)
        return x

class AA_Transformer_Decoder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=4,
        aa_order_list = ['f', 'g', 'f', 'g'],
        heads=4,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        qk_norm=None,
        conv_layers=0,
        pe_attn_head=None,
        attn_backend="torch",  # "torch" | "flash_attn"
        attn_mask_enabled=False,
        checkpoint_activations=False,
        return_attn= False,
    ):
        super().__init__()
        assert len(aa_order_list) == depth, "The length of aa_order_list must be equal to the depth of the model"
        self.aa_order_list = aa_order_list
        
        self.rotary_embed = RotaryEmbedding(dim_head)
        self.dim = dim
        self.depth = depth
        self.return_attn = return_attn

        self.transformer_blocks = nn.ModuleList(
            [
                Transformer_Block(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    pe_attn_head=pe_attn_head,
                    attn_backend=attn_backend,
                    attn_mask_enabled=attn_mask_enabled,
                    return_attn = self.return_attn,
                )
                for _ in range(depth)
            ]
        )
        self.checkpoint_activations = checkpoint_activations

    def _process_frame_attention(self, x, B, S, P, C, idx):
        """
        Process frame attention blocks.
        """
        # If needed, reshape tokens:
        if x.shape != (B * S, P, C):
            x = x.contiguous().view(B, S, P, C).view(B * S, P, C)

        intermediates = []
        seq_len = x.shape[1]
        rope = self.rotary_embed.forward_from_seq_len(seq_len)
        if self.return_attn:
            x, attn_weight = self.transformer_blocks[idx](x, rope=rope)
        else:
            x = self.transformer_blocks[idx](x, rope=rope)
        idx += 1
        intermediates.append(rearrange(x, "(b s) p c -> b s p c", s = S))

        return x, idx, intermediates
    
    def _process_global_attention(self, x, B, S, P, C, idx, gP = 0):
        """
        Process glbal attention blocks.
        """
        # If needed, reshape tokens:
        if gP == 0:
            if x.shape != (B, S*P, C):
                x = x.contiguous().view(B, S, P, C).view(B, S*P, C)
        else:
            if x.shape != (B, S*P + gP, C):
                raise ValueError(f"The shape of x is not correct: {x.shape}")

        intermediates = []
        seq_len = x.shape[1]
        rope = self.rotary_embed.forward_from_seq_len(seq_len)
        if self.return_attn:
            x, attn_weight = self.transformer_blocks[idx](x, rope=rope)
        else:
            x = self.transformer_blocks[idx](x, rope=rope)
        idx += 1
        intermediates.append(rearrange(x[:, gP:], "b (s p) c -> b s p c", s = S))

        return x, idx, intermediates 

    def forward(self, x, mask=None):
        B, S, P, C = x.shape
        idx = 0
        for attn_type in self.aa_order_list:
            if attn_type == 'f':
                x, idx, frame_intermediates = self._process_frame_attention(x, B, S, P, C, idx)
            elif attn_type == 'g':
                x, idx, global_intermediates = self._process_global_attention(x, B, S, P, C, idx)
            else:
                raise ValueError(f"Invalid attention type: {attn_type}")
        last_frame_x = frame_intermediates[-1] # (B, S, P, C)
        last_global_x = global_intermediates[-1] # (B, S, P, C)
        x = torch.cat((last_frame_x, last_global_x), dim=-1) # (B, S, P, C*2)
        return x