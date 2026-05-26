from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from models.layers.modules import (
    RMSNorm,
    FeedForward,
    Attention,
    AttnProcessor,
    CrossAttention,
    CrossAttnProcessor,
    AdaLayerNorm_Final,
    DiTBlock,
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

class CA_Transformer_Block(nn.Module):
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
        self.attn_norm_x = RMSNorm(dim, eps=1e-6)
        self.attn_norm_c = RMSNorm(dim, eps=1e-6)
        self.return_attn = return_attn
        self.attn = CrossAttention(
            processor=CrossAttnProcessor(
                pe_attn_head=pe_attn_head,
                attn_backend=attn_backend,
                attn_mask_enabled=attn_mask_enabled,
                return_attn_weight=self.return_attn,
            ),
            dim=dim,
            context_dim = None,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            qk_norm=qk_norm,
        )
        
        self.ff_norm = RMSNorm(dim, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, c, x_mask=None, c_mask=None, x_rope=None, c_rope=None):
        if self.return_attn:
            raise AssertionError("Return attention is not supported for cross transformer block for now")
        else:
            attn_out = self.attn(self.attn_norm_x(x), c=self.attn_norm_c(c), x_mask=x_mask, c_mask=c_mask, x_rope=x_rope, c_rope=c_rope)
        x = x + attn_out
        x = x + self.ff(self.ff_norm(x))
        if self.return_attn:
            raise AssertionError("Return attention is not supported for cross transformer block for now")
        else:
            return x

class CA_Transformer_Decoder(nn.Module):
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
                CA_Transformer_Block(
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
        

    def forward(self, x, c, x_mask=None, c_mask=None, x_rope=None, c_rope=None):
        batch, x_seq_len, c_seq_len = x.shape[0], x.shape[1], c.shape[1]
        x_rope = self.rotary_embed.forward_from_seq_len(x_seq_len)
        c_rope = self.rotary_embed.forward_from_seq_len(c_seq_len)
        #x = self.proj_in(x)
        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c, x_mask, c_mask, x_rope, c_rope, use_reentrant=False)
            else:
                if self.return_attn:
                    raise AssertionError("Return attention is not supported for cross transformer block for now")
                else:
                    x = block(x, c=c, x_mask=x_mask, c_mask=c_mask, x_rope=x_rope, c_rope=c_rope)
        #x = self.proj_out(x)
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

        self.frame_attention_blocks = nn.ModuleList(
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
        self.global_attention_blocks = nn.ModuleList(
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

    def _process_frame_attention(self, x, B, N, T, D, idx, mask = None):
        """
        Process frame attention blocks.
        args:
            x: (B, N, T, D)
            mask: (B, N, T)
        return:
            x: (B, N, T, D)
            idx: int
            intermediates: [(B, N, T, D)]
        """
        # arrange mask
        if mask is not None:
            if not isinstance(mask, bool):
                mask = mask.bool()
            mask = rearrange(mask, "b n t -> (b n) t")

        intermediates = []
        assert x.shape == (B, N, T, D), f"The shape of x is not correct: {x.shape}"
        rope = self.rotary_embed.forward_from_seq_len(T)
        x = rearrange(x, "b n t d -> (b n) t d")
        if self.return_attn:
            x, attn_weight = self.frame_attention_blocks[idx](x, rope=rope, mask = mask)
        else:
            x = self.frame_attention_blocks[idx](x, rope=rope, mask = mask)
        idx += 1
        x = rearrange(x, "(b n) t d -> b n t d", n = N)
        intermediates.append(x)
        return x, idx, intermediates
    
    def _process_global_attention(self, x, B, N, T, D, idx, g = None, gT = 0, mask = None):
        """
        Process glbal attention blocks.
        args:
            x: (B, N, T, D)
            g: (B, gT, D) work as global prefix
        return:
            x: (B, N, T, D)
            g: (B, gT, D)
            idx: int
            intermediates: [(B, N, T, D)]
        """
        # arrange mask
        if mask is not None:
            if not isinstance(mask, bool):
                mask = mask.bool()
            mask = rearrange(mask, "b n t -> b (n t)")

        if x.shape != (B, N, T, D):
            raise ValueError(f"The shape of x is not correct: {x.shape}")

        intermediates = []
        
        x = rearrange(x, "b n t d -> b (n t) d")
        if g is not None and gT > 0:
            x = torch.cat([g, x], dim = 1)
        rope = self.rotary_embed.forward_from_seq_len(x.shape[1])
        if self.return_attn:
            x, attn_weight = self.global_attention_blocks[idx](x, rope=rope, mask = mask)
        else:
            x = self.global_attention_blocks[idx](x, rope=rope, mask = mask)
        idx += 1
        if g is not None and gT > 0:
            g = x[:, :gT]
        x = x[:, gT:]
        x = rearrange(x, "b (n t) d -> b n t d", n = N)
        intermediates.append(x)

        return x, g, idx, intermediates 

    def forward(self, x, mask=None):
        B, N, T, D = x.shape
        idx = 0
        output_list = []
        for attn_type in self.aa_order_list:
            if attn_type == 'f':
                x, idx, frame_intermediates = self._process_frame_attention(x, B, N, T, D, idx)
            elif attn_type == 'g':
                x, idx, global_intermediates = self._process_global_attention(x, B, N, T, D, idx)
            else:
                raise ValueError(f"Invalid attention type: {attn_type}")
            concat_inter = torch.cat((frame_intermediates[-1], global_intermediates[-1]), dim=-1)
            output_list.append(concat_inter)
            del concat_inter
        x = output_list[-1]
        return x

class AttentionPool(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head,
        pe_attn_head=None,
        attn_backend="torch",
        dropout=0.1,
    ):
        super().__init__()
        # 1. learnable query
        self.latent_query = nn.Parameter(torch.randn(1, dim))
        
        # 2. normalization layer
        self.query_norm = RMSNorm(dim, eps=1e-6)
        self.context_norm = RMSNorm(dim, eps=1e-6)
        
        # 3. cross attention
        self.attn = CrossAttention(
            processor=CrossAttnProcessor(
                pe_attn_head=pe_attn_head,
                attn_backend=attn_backend,
                attn_mask_enabled=False, 
                return_attn_weight=False,
            ),
            dim=dim,
            context_dim=dim, 
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )
        
        # 4. post projection
        self.post_proj = nn.Linear(dim, dim)

    def forward(self, c, c_mask=None, c_rope=None):
        """
        argsL
            c: input features, shape (B, N, dim)
            c_mask: padding mask
            c_rope: rotary embedding
        return:
            pooled_feat: shape (B, dim)
        """
        B = c.size(0)
        
        # 1. prepare query
        q = self.latent_query.unsqueeze(0).expand(B, -1, -1)
        
        # 2. normalization
        q = self.query_norm(q)
        c = self.context_norm(c)
        
        # 3. cross attention aggregation
        pooled_feat = self.attn(q, c=c, c_mask=c_mask, c_rope=c_rope)
        
        # 4. reduce dimension and output
        pooled_feat = pooled_feat.squeeze(1)
        pooled_feat = self.post_proj(pooled_feat)
        
        return pooled_feat

"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""
# ruff: noqa: F722 F821


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
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
        self.proj_out = nn.Linear(dim, latent_dim)
        #self.proj_in = nn.Linear(latent_dim, dim)

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


    def forward(
        self,
        x: float["b n d"],  # nosied input 
        global_ebd: float["b d"] | None = None,
        attn_mask: bool["b n"] | None = None,
        seq_lens: (int, int) | None = None,
    ):
        batch, seq_len = x.shape[0], x.shape[1]

        embedding = global_ebd
        if seq_lens is not None:
            rope = self.rotary_embed.forward_from_mod_seq_lens(seq_lens)
        else:
            rope = self.rotary_embed.forward_from_seq_len(seq_len)
       
        #x = self.proj_in(x)
        if self.long_skip_connection is not None:
            residual = x
        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                # https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, embedding, attn_mask, rope, use_reentrant=False)
            else:
                x = block(x, embedding, mask=attn_mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, embedding)
        output = self.proj_out(x)
        return output
