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

#from models.backbone.llama_nar import DiffLlama
#from models.backbone.aadit import AADiT
#from models.backbone.dit import DiT
from models.layers.transformer_decoder import Transformer_Decoder, AA_Transformer_Decoder
from models.layers.patch_embed import PatchEmbed
from models.layers.dinov3 import DinoV3Encoder, create_dinov3
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

class FlowMatchingTransformer(nn.Module):
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
        
        
        if cfg.depth_encoder.vit:
            self.depth_encoder_name = 'vit'
            self.depth_dim = (cfg.depth_encoder.vit_arch.dim 
                if hasattr(cfg.depth_encoder.vit_arch, 'dim') else 512)
            self.depth_patchify = PatchEmbed(img_size=self.depth_resol, 
                patch_size=self.depth_patch_size, 
                in_chans=self.depth_channel * 2, 
                embed_dim=self.depth_dim)
            self.depth_encoder = Transformer_Decoder(**cfg.depth_encoder.vit_arch)
        else:
            self.depth_encoder_name = 'dinov3'
            self.depth_dim = cfg.depth_encoder.dinov3_arch.output_dim
            
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
            self.aligner_layer = cfg.aligner.layer
            env_time_interval = torch.linspace(0, duration, frame_size)
            self.register_buffer("env_time_interval", env_time_interval)
            self.env_time_interval_embedding = nn.Sequential(SinusPositionEmbedding(dim=pre_dim), 
            Mlp(pre_dim, self.hidden_size, self.hidden_size))
            self.aligner_label_proj = nn.Linear(2*self.hidden_size, self.hidden_size)
            self.aligner_map = AdaLayerNorm(self.hidden_size)
        self.reset_parameters()
        if self.depth_encoder_name == 'dinov3':
            self.depth_encoder = DinoV3Encoder(**cfg.depth_encoder.dinov3_arch)
       
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
        if self.depth_encoder_name == 'vit':
            depth_tokens = self.depth_patchify(depth_map)#(b, pw*ph, depth_dim)
            depth_tokens = self.depth_encoder(depth_tokens)#(b, pw*ph, depth_dim)
            #mean pool
            depth_tokens = depth_tokens.mean(dim = 1, keepdim = True) #(b, 1, depth_dim)
        else:
            depth_tokens = self.depth_encoder(depth_map).unsqueeze(1)#(b, 1, depth_dim)
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

    def prepare_tokens(self, ir_z, depth_map, loc, irs_num, bsz):
        """
        Args:
           depth_map: (B, 3, 256, 512)
           loc: (B, N, 3)
           ir_z: (B, N, T, 1024)
           irs_num: int
           bsz: int
        Returns:
           label_tokens: (B, N, 1, dim)
           ir_tokens: (B, N, t, dim)
        """
        loc_tokens = self.get_loc_tokens(loc) #(B, N, 1, 0.5*dim)
        geo_map = depth_map.unsqueeze(1) - loc.unsqueeze(-1).unsqueeze(-1)#(B, N, 3, 256, 512)
        geo_map = torch.cat([geo_map, depth_map.unsqueeze(1).repeat(1, irs_num, 1, 1, 1)], dim = 2)#(B, N, 6, 256, 512)
        B, N, _, H, W = geo_map.shape
        geo_map = rearrange(geo_map, "b n c h w -> (b n) c h w")
        geo_tokens = self.get_depth_tokens(geo_map) #(B * N, 1, 0.5*dim)
        geo_tokens = rearrange(geo_tokens, "(b n) 1 d -> b n 1 d", n = N) #(B, N, 1, 0.5*dim)
        label_tokens = torch.cat([loc_tokens, geo_tokens], dim = -1) #(B, N, 1, dim)
        time_tokens = self.get_time_tokens(self.ir_time_interval) #(T, dim)
        tgt_ir_init_tokens = time_tokens.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1) #(B, 1, T, dim)
        ref_ir_tokens = ir_z[:,:-1]#(B, N-1, t, 1024)
        ir_tokens = torch.cat([ref_ir_tokens, tgt_ir_init_tokens], dim = 1) #(B, N, 1, dim)
        return label_tokens, ir_tokens

    def aggregator(self, ir_tokens, label_tokens):
        """
        Args:
           ir_tokens: (B, N, T, D)
           label_tokens: (B, N, 1, D)
        Returns:
           ir_tokens: (B, N, T, 2D)
           label_tokens: (B, N, 1, D)
        """
        B, N, T, D = ir_tokens.shape
        frame_idx = 0
        global_idx = 0
        out_list = []
        label_list = []
        mask = torch.ones(B, N, T+1, device = ir_tokens.device)
        mask[:, :-1, 1:] = 0

        visible_reference_ac_token = 4
        
        mask[:, :visible_reference_ac_token, 1:] = 1
        mask = mask.bool()
        for depth_idx, aa_block in enumerate(self.aa_order_list):
            for attn_type in aa_block:
                if attn_type == 'f':
                    f_x = torch.cat([label_tokens, ir_tokens], dim = -2)#(B, N, T+1, D)
                    f_x, frame_idx, frame_intermediates = self.av_transformer._process_frame_attention(f_x, B, N, T+1, D, frame_idx, mask = mask)
                    ir_tokens = f_x[:,:,1:]
                    label_tokens = f_x[:,:,0:1]
                elif attn_type == 'g':
                    g_x = torch.cat([label_tokens, ir_tokens], dim = -2)#(B, N, T+1, D)
                    g_x, g, global_idx, global_intermediates = self.av_transformer._process_global_attention(g_x, B, N, T+1, D, global_idx, gT = 0, mask = mask)
                    ir_tokens = g_x[:,:,1:]
                    label_tokens = g_x[:,:,0:1]
                else:
                    raise ValueError(f"Invalid attention type: {attn_type}")
            for i in range(len(frame_intermediates)):
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim = -1)#(B, N, T, 2D)
                out_list.append(concat_inter)
            del concat_inter
        return out_list


    def forward(self, ir_z, cc_depth_map, cc_src_loc, align_feat_ir = None):
        """
        Args:
           ir_z: (B, N, D, D)
           cc_depth_map: (B, 3, 256, 512)
           cc_src_loc: (B, N, 3)
        Return:
           tgt_ir_tokens: (B, T, D)
        """
        bsz, irs_num = ir_z.shape[0], ir_z.shape[1]
        label_tokens, ir_tokens = self.prepare_tokens(ir_z, cc_depth_map, cc_src_loc, irs_num, bsz)
        # print(f"检查pre_embedding输出")
        # print(f"label_tokens: {label_tokens.shape}") #(B, N, 1, D), 
        # print(f"ir_tokens: {ir_tokens.shape}") #(B, N, T, D)
        # print("--------------------------------")
        # #exit()
        out_list = self.aggregator(ir_tokens, label_tokens)
        # print(f"检查aggregator输出")
        # print(f"out_list: {len(out_list)}")
        # print(f"out_list[0]: {out_list[0].shape}")
        # print("--------------------------------")
        last_concat_inter = out_list[-1] # (B, N, T, 2D)
        tgt_ir_tokens = last_concat_inter[:, -1, 1:]#(B, T, 2D)
        tgt_ir_tokens = self.ir_decoder(tgt_ir_tokens)#(B, T, D)
        # print(f"检查ir_decoder输出")
        # print(f"tgt_ir_tokens: {tgt_ir_tokens.shape}") #(B, T, D)
        # print("--------------------------------")
        #exit()
        
        if self.aligner_activate:
            label_tokens = out_list[self.aligner_layer-1][:,:,1] #(B, N, 2D)
            B, N, _ = label_tokens.shape
            label_tokens = rearrange(label_tokens, "b n d -> (b n) d")#(B*N, 2D)
            label_tokens = self.aligner_label_proj(label_tokens)#(B*N, D)
            env_tokens = self.env_time_interval_embedding(self.env_time_interval)#(T, 2D)
            env_tokens = env_tokens.unsqueeze(0).repeat(B*N, 1, 1)#(B*N, T, D)

            env_tokens = self.aligner_map(env_tokens, label_tokens)#(B*N, T, D)
            del out_list
            return tgt_ir_tokens, env_tokens
        else:
            del out_list
            return tgt_ir_tokens

   