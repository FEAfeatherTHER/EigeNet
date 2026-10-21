# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from copy import deepcopy
import torch
import numpy as np
import torch.nn as nn
import math
from einops import rearrange
import random
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel
from typing import List, Optional, Tuple, Union
from einops import rearrange
from transformers.models.llama.modeling_llama import BaseModelOutputWithPast

from models.layers.transformer_decoder import Transformer_Decoder, AA_Transformer_Decoder, DiT
from models.layers.patch_embed import PatchEmbed
from models.layers.mlp import Mlp
from models.layers.modules import (ResNet18,
                                SinusPositionEmbedding)

class AdaLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 3)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None):
        emb = self.linear(self.silu(emb))
        shift, scale, gate = torch.chunk(emb, 3, dim=-1)
        x = x + gate[:, None] * (self.norm(x) * (1 + scale[:, None]) + shift[:, None])
        return x

class EigeNetTransformer(nn.Module):
    def __init__(
        self,
        duration = 0.5,
        frame_rate = 50,
        cfg=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.aligner_activate = cfg.aligner.activate
        pre_dim = (cfg.pre_dim if hasattr(cfg, 'pre_dim') else 512)
        self.hidden_size = (self.cfg.hidden_size 
            if hasattr(cfg, 'hidden_size') else 1024)
        self.ir_decoder_hs = (self.cfg.av_transformer.arch.dim 
            if hasattr(cfg.av_transformer.arch, 'dim') else 1024)
        
        self.depth_patch_size = tuple(cfg.depth_patch_size 
            if hasattr(cfg, 'depth_patch_size') else (16, 16))
        self.depth_resol = tuple(cfg.depth_resol 
            if hasattr(cfg, 'depth_resol') else (256, 512))
        
        self.depth_channel = (cfg.depth_channel 
            if hasattr(cfg, 'depth_channel') else 3)
        
        self.depth_dim = (cfg.depth_encoder.vit_arch.dim 
            if hasattr(cfg.depth_encoder.vit_arch, 'dim') else 512)
        self.depth_patchify = PatchEmbed(img_size=self.depth_resol, 
            patch_size=self.depth_patch_size, 
            in_chans=self.depth_channel * 2, 
            embed_dim=self.depth_dim)
        self.depth_encoder = Transformer_Decoder(**cfg.depth_encoder.vit_arch)
            
        assert self.depth_dim * 2 == self.hidden_size, "The depth_dim * 2 is not equal to the hidden_size"

        self.loc_embedding = nn.Sequential(SinusPositionEmbedding(dim=pre_dim), 
                Mlp(3 *pre_dim, self.depth_dim, self.depth_dim))
        
        self.aa_order_list = self.cfg.av_transformer.arch.aa_order_list
        self.aa_depth = self.cfg.av_transformer.arch.depth
        assert self.aa_depth == len(self.aa_order_list), "The depth is not equal to the length of aa_order_list"
        
        self.av_transformer = AA_Transformer_Decoder(**cfg.av_transformer.arch)

        frame_size = int(duration * frame_rate)
        ir_time_interval = torch.linspace(0, duration, frame_size)
        self.register_buffer("ir_time_interval", ir_time_interval)
        self.ir_time_interval_embedding = nn.Sequential(SinusPositionEmbedding(dim=pre_dim), 
            Mlp(pre_dim, self.hidden_size, self.hidden_size))
        self.ir_decoder = nn.Linear(2*self.hidden_size, self.hidden_size)
        if self.aligner_activate:
            self.aligner_av_transformer = DiT(
                dim = self.hidden_size,
                depth = 1,
                heads = 4,
                dim_head = self.hidden_size // 4,
                dropout = 0.1,
                ff_mult = 2,
                latent_dim = self.hidden_size,
                qk_norm = "rms_norm",
                pe_attn_head = None,
                attn_backend = "flex",
                attn_mask_enabled = False,
                long_skip_connection = False,
                checkpoint_activations = False,
            )
            self.align_out_proj = nn.Linear(self.hidden_size, 8)
        
        self.reset_parameters()
       
    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, nn.MultiheadAttention):
                if m._qkv_same_embed_dim:
                    nn.init.normal_(m.in_proj_weight, std=0.02)
                else:
                    nn.init.normal_(m.q_proj_weight, std=0.02)
                    nn.init.normal_(m.k_proj_weight, std=0.02)
                    nn.init.normal_(m.v_proj_weight, std=0.02)

                if m.in_proj_bias is not None:
                    nn.init.constant_(m.in_proj_bias, 0.0)
                    nn.init.constant_(m.out_proj.bias, 0.0)
                if m.bias_k is not None:
                    nn.init.xavier_normal_(m.bias_k)
                if m.bias_v is not None:
                    nn.init.xavier_normal_(m.bias_v)

            elif (
                isinstance(m, nn.Conv1d)
                or isinstance(m, nn.ConvTranspose1d)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.ConvTranspose2d)
            ):
                m.weight.data.normal_(0.0, 0.02)

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

        self.apply(_reset_parameters)
    
    def get_loc_tokens(self, loc):
        """
        Args:
           loc: (B, N, 3)
        Returns:
           loc_tokens: (B, N, 1, dim)
        """
        bsz, irs_num, _ = loc.shape
        loc = rearrange(loc, "b n c -> (b n c)")
        loc_tokens = self.loc_embedding[0](loc)#(B*N*3, pre_dim)
        loc_tokens = loc_tokens.reshape(-1, 3, loc_tokens.shape[-1]) #（B*N, 3, pre_dim）
        new_bsz = loc_tokens.shape[0]
        loc_tokens = loc_tokens.reshape(new_bsz, -1) #(B*N, 3*pre_dim)
        loc_tokens = self.loc_embedding[1](loc_tokens).unsqueeze(1) #(B*N, 1, dim)
        loc_tokens = loc_tokens.reshape(bsz, irs_num, 1, loc_tokens.shape[-1]) #(B, N, 1, dim)
        return loc_tokens
    
    def get_depth_tokens(self, depth_map):
        """
        Args:
           depth_map: (B, 3, 256, 512)
        Returns:
           depth_tokens: (B, 1, 512)
        """
        depth_tokens = self.depth_patchify(depth_map)#(b, pw*ph, depth_dim)
        depth_tokens = self.depth_encoder(depth_tokens)#(b, pw*ph, depth_dim)
        #mean pool
        depth_tokens = depth_tokens.mean(dim = 1, keepdim = True) #(b, 1, depth_dim)

        return depth_tokens #(b, 1, 512)

    def get_time_tokens(self, time_interval):
        """
        Args:
           time_interval:(T,)
        Returns:
           time_tokens: (T, dim)
        """
        time_ebd = self.ir_time_interval_embedding(time_interval) #(T, dim)
        return time_ebd

    def prepare_tokens(self, ref_ir_z, depth_map, loc):
        """
        Args:
           depth_map: (B, 3, 256, 512)
           loc: (B, K+1, 3), references followed by target
           ref_ir_z: (B, K, T, 1024), reference encodings only
        Returns:
           label_tokens: (B, N, 1, dim)
           ir_tokens: (B, N, t, dim)
        """
        irs_num = loc.shape[1]
        loc_tokens = self.get_loc_tokens(loc) #(B, N, 1, 0.5*dim)
        geo_map = depth_map.unsqueeze(1) - loc.unsqueeze(-1).unsqueeze(-1)#(B, N, 3, 256, 512)
        # expand avoids an extra copy before cat materializes the geometry input.
        geo_map = torch.cat([geo_map, depth_map.unsqueeze(1).expand(-1, irs_num, -1, -1, -1)], dim = 2)#(B, N, 6, 256, 512)
        B, N, _, H, W = geo_map.shape
        geo_map = rearrange(geo_map, "b n c h w -> (b n) c h w")
        geo_tokens = self.get_depth_tokens(geo_map) #(B * N, 1, 0.5*dim)
        geo_tokens = rearrange(geo_tokens, "(b n) 1 d -> b n 1 d", n = N) #(B, N, 1, 0.5*dim)
        label_tokens = torch.cat([loc_tokens, geo_tokens], dim = -1) #(B, N, 1, dim)
        time_tokens = self.get_time_tokens(self.ir_time_interval) #(T, dim)
        tgt_ir_init_tokens = time_tokens.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1) #(B, 1, T, dim)
        # The target is initialized from time/geometry, never from its audio.
        # Its DAC encoding is therefore no longer computed by the caller.
        ref_ir_tokens = ref_ir_z
        if self.aligner_activate:
            tgt_ir_init_tokens = tgt_ir_init_tokens.squeeze(1)# (B, T, dim)
            tgt_label_tokens = label_tokens[:, -1, 0]#(B, dim)
            tgt_ir_init_tokens = self.aligner_av_transformer(tgt_ir_init_tokens, tgt_label_tokens)#(B, T, dim)
            align_feature = self.align_out_proj(tgt_ir_init_tokens)#(B, T, 8)
            tgt_ir_init_tokens = tgt_ir_init_tokens.unsqueeze(1)#(B, 1, T, dim)
            ir_tokens = torch.cat([ref_ir_tokens, tgt_ir_init_tokens], dim = 1) #(B, N, 1, dim)
            return label_tokens, ir_tokens, align_feature
        else:
            ir_tokens = torch.cat([ref_ir_tokens, tgt_ir_init_tokens], dim = 1) #(B, N, 1, dim)
            return label_tokens, ir_tokens

    def aggregator(self, ir_tokens, label_tokens):
        """
        Args:
           ir_tokens: (B, N, T, D)
           label_tokens: (B, N, 1, D)
        Returns:
           final target tokens: (B, T, 2D)
        """
        B, N, T, D = ir_tokens.shape
        frame_idx = 0
        global_idx = 0
        for aa_block in self.aa_order_list:
            for attn_type in aa_block:
                if attn_type == 'f':
                    f_x = torch.cat([label_tokens, ir_tokens], dim = -2)#(B, N, T+1, D)
                    f_x, frame_idx, frame_intermediates = self.av_transformer._process_frame_attention(f_x, B, N, T+1, D, frame_idx)
                    ir_tokens = f_x[:,:,1:]
                    label_tokens = f_x[:,:,0:1]
                elif attn_type == 'g':
                    g_x = torch.cat([label_tokens, ir_tokens], dim = -2)#(B, N, T+1, D)
                    g_x, g, global_idx, global_intermediates = self.av_transformer._process_global_attention(g_x, B, N, T+1, D, global_idx, gT = 0)
                    ir_tokens = g_x[:,:,1:]
                    label_tokens = g_x[:,:,0:1]
                else:
                    raise ValueError(f"Invalid attention type: {attn_type}")
        # Only the final target tokens reach ir_decoder. Earlier concatenations
        # and reference/label slices had no consumers; the attention path stays intact.
        return torch.cat(
            [frame_intermediates[-1][:, -1, 1:], global_intermediates[-1][:, -1, 1:]],
            dim=-1,
        )


    def forward(self, ref_ir_z, cc_depth_map, cc_src_loc):
        """
        Args:
           ref_ir_z: (B, K, T, D), reference encodings only
           cc_depth_map: (B, 3, 256, 512)
           cc_src_loc: (B, K+1, 3), references followed by target
        Return:
           tgt_ir_tokens: (B, T, D)
        """
        # Infer the total source count from positions; the audio input excludes
        # the target. This preserves its time-based initialization for every K.
        if cc_src_loc.shape[1] != ref_ir_z.shape[1] + 1:
            raise ValueError("Source positions must contain all references followed by one target")
        if self.aligner_activate:
            label_tokens, ir_tokens, align_feature = self.prepare_tokens(ref_ir_z, cc_depth_map, cc_src_loc)
        else:
            label_tokens, ir_tokens = self.prepare_tokens(ref_ir_z, cc_depth_map, cc_src_loc)
        tgt_ir_tokens = self.aggregator(ir_tokens, label_tokens)
        out_tgt_ir_tokens = self.ir_decoder(tgt_ir_tokens)#(B, T, D)
        
        if self.aligner_activate:
            return out_tgt_ir_tokens, align_feature
        else:
            return out_tgt_ir_tokens
