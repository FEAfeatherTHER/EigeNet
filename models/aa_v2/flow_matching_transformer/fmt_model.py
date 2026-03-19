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
from models.layers.mlp import Mlp
from models.layers.modules import (ResNet18,
                                SinusPositionEmbedding)

class FlowMatchingTransformer(nn.Module):
    def __init__(
        self,
        duration = 0.5,
        frame_rate = 50,
        cfg=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.ap = cfg.ap
        self.env = cfg.env
        pre_dim = (cfg.pre_dim if hasattr(cfg, 'pre_dim') else 512)
        self.hidden_size = (self.cfg.hidden_size 
            if hasattr(cfg, 'hidden_size') else 1024)
        self.ir_decoder_hs = (self.cfg.ir_decoder_hs 
            if hasattr(cfg, 'ir_decoder_hs') else 1024)
        self.env_decoder_hs = (self.cfg.env_decoder_hs 
            if hasattr(cfg, 'env_decoder_hs') else 256)
        self.depth_patch_size = tuple(cfg.depth_patch_size 
            if hasattr(cfg, 'depth_patch_size') else (16, 32))
        self.ap_patch_size = tuple(cfg.ap_patch_size 
            if hasattr(cfg, 'ap_patch_size') else (2, 4))
        self.depth_resol = tuple(cfg.depth_resol 
            if hasattr(cfg, 'depth_resol') else (256, 512))
        self.ap_resol = tuple(cfg.ap_resol 
            if hasattr(cfg, 'ap_resol') else (32, 64))
        
        self.depth_dim = cfg.depth_dim
        self.depth_channel = cfg.depth_channel
        self.depth_patchify = PatchEmbed(img_size=self.depth_resol, 
            patch_size=self.depth_patch_size, 
            in_chans=self.depth_channel, 
            embed_dim=self.depth_dim)
        if self.ap:

            self.ap_dim = cfg.ap_dim
            self.ap_channel = cfg.ap_channel
            self.ap_tensor = nn.Parameter(torch.ones(1, self.ap_channel, self.ap_resol[0], self.ap_resol[1]), requires_grad=True)
            self.ap_patchify = PatchEmbed(img_size=self.ap_resol, 
                patch_size=self.ap_patch_size, 
                in_chans=self.ap_channel, 
                embed_dim=self.ap_dim)
        else:
            self.depth_in_proj = nn.Linear(self.depth_dim, self.hidden_size)

        self.aa_order_list = self.cfg.av_transformer.arch.aa_order_list
        self.aa_depth = self.cfg.av_transformer.arch.depth
        assert self.aa_depth == len(self.aa_order_list), "The depth is not equal to the length of aa_order_list"

        self.av_transformer = AA_Transformer_Decoder(**cfg.av_transformer.arch)

        self.loc_embedding = nn.Sequential(SinusPositionEmbedding(dim=pre_dim), 
                nn.Linear(3 *pre_dim, self.hidden_size))

        frame_size = int(duration * frame_rate)
        ir_time_interval = torch.linspace(0, duration, frame_size)
        self.register_buffer("ir_time_interval", ir_time_interval)
        self.ir_time_interval_embedding = nn.Sequential(SinusPositionEmbedding(dim=pre_dim), 
            nn.Linear(pre_dim, self.hidden_size))
        self.ir_decoder = nn.Sequential(nn.Linear(2*self.hidden_size, self.ir_decoder_hs),
            Transformer_Decoder(**cfg.ir_decoder.arch), 
            nn.Linear(self.ir_decoder_hs, self.hidden_size))

        if self.env:
            env_time_interval = torch.linspace(0, duration, frame_size)
            self.register_buffer("env_time_interval", env_time_interval)
            self.env_time_interval_embedding = nn.Sequential(SinusPositionEmbedding(dim=pre_dim), 
                nn.Linear(pre_dim, self.env_decoder_hs))
            self.env_decoder = nn.Sequential(Transformer_Decoder(**cfg.env_decoder.arch),
                nn.Linear(self.env_decoder_hs, self.hidden_size))
            self.loc_proj = nn.Linear(2*self.hidden_size, self.env_decoder_hs)
            self.room_proj = nn.Linear(self.hidden_size, self.env_decoder_hs)

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

    def get_time_tokens(self, time_interval, type = "ir"):
        """
        Args:
           time_interval:(T,)
        Returns:
           time_tokens: (T, dim)
        """
        if type == "ir":
            time_ebd = self.ir_time_interval_embedding(time_interval) #(T, dim)
        elif type == "env":
            time_ebd = self.env_time_interval_embedding(time_interval) #(T, dim)
        else:
            raise ValueError(f"Invalid time type: {type}")
        return time_ebd
    
    def prepare_tokens(self, depth_map, loc, irs_num, bsz):
        """
        Args:
           depth_map: (B, 3, 256, 512)
           loc: (B, N, 3)
           irs_num: int
           bsz: int
        Returns:
           loc_tokens: (B, N, 1, dim)
           room_tokens: (B, 1, dim)
           time_tokens: (B, 1, T, dim)
        """
        bsz = depth_map.shape[0]
        ir_time_tokens = self.get_time_tokens(self.ir_time_interval, type = "ir") #(T, dim)
        ir_time_tokens = ir_time_tokens.unsqueeze(0).unsqueeze(0).repeat(bsz, 1, 1, 1) #(B, 1, T, dim)
        loc_tokens = self.get_loc_tokens(loc) #(B, N, 1, dim)
        depth_tokens  = self.depth_patchify(depth_map) #(B, ph*pw, depth_dim)
        if not self.ap:
            depth_tokens = self.depth_in_proj(depth_tokens) #(B, ph*pw, dim)
            room_tokens = depth_tokens
        else:
            ap_tokens = self.ap_patchify(self.ap_tensor).repeat(bsz, 1, 1) #(B, ph*pw, ap_dim)
            room_tokens = torch.cat([depth_tokens, ap_tokens], dim = -1) #(B, ph*pw, dim)
        return loc_tokens, room_tokens, ir_time_tokens

    def aggregator(self, ref_view_tokens, tgt_view_tokens, room_tokens):
        """
        Args:
           ref_view_tokens: (B, N-1, t+1, dim)
           tgt_view_tokens: (B, 1, t+1, dim)
           room_tokens: (B, ph*pw, dim)
        Returns:
           ir_tokens: (B, N, t, dim*2)
           room_tokens: (B, ph*pw, dim)
           loc_tokens: (B, N, 1, dim*2)
        """
        all_view_tokens = torch.cat([ref_view_tokens, tgt_view_tokens], dim = 1)#(B, N, t+1, dim)
        patch_num = room_tokens.shape[1]
        B, S, P, C = all_view_tokens.shape
        idx = 0
        for attn_type in self.aa_order_list:
            if attn_type == 'f':
                x = all_view_tokens
                x, idx, frame_intermediates = self.av_transformer._process_frame_attention(x, B, S, P, C, idx)
                all_view_tokens = rearrange(x, "(b s) p c -> b s p c", s = S)
            elif attn_type == 'g':
                all_view_tokens = rearrange(all_view_tokens, "b s p c -> b (s p) c")
                x = torch.cat([room_tokens, all_view_tokens], dim = 1)
                x, idx, global_intermediates = self.av_transformer._process_global_attention(x, B, S, P, C, idx, gP = patch_num)
                room_tokens = x[:, :patch_num, :]
                all_view_tokens = x[:, patch_num:, :]
                all_view_tokens = rearrange(all_view_tokens, "b (s p) c -> b s p c", s = S)
            else:
                raise ValueError(f"Invalid attention type: {attn_type}")
        last_frame_all_view_tokens = frame_intermediates[-1]
        last_global_all_view_tokens = global_intermediates[-1]
        last_all_view_tokens = torch.cat([last_frame_all_view_tokens, last_global_all_view_tokens], dim = -1)
        last_all_view_ir_tokens = last_all_view_tokens[..., 1:, :]
        last_all_view_loc_tokens = last_all_view_tokens[..., 0:1, :]
        del last_all_view_tokens, last_frame_all_view_tokens, last_global_all_view_tokens, global_intermediates, frame_intermediates
        return last_all_view_ir_tokens, room_tokens, last_all_view_loc_tokens

    def env_decode(self, room_tokens, loc_tokens):
        loc_tokens = self.loc_proj(loc_tokens)
        room_tokens = self.room_proj(room_tokens)
        bsz = room_tokens.shape[0]
        env_tokens = self.get_time_tokens(self.env_time_interval, type = "env")#(T, dim)
        env_token_num = env_tokens.shape[0]
        env_tokens = env_tokens.unsqueeze(0).repeat(bsz, 1, 1)#(B, T, dim)
        x = torch.cat([room_tokens, loc_tokens, env_tokens], dim = 1)#(B, ph*pw+T+1, dim)
        x = self.env_decoder(x)#(B, ph*pw+T+1, dim)
        env_tokens = x[:, -env_token_num:, :]#(B, T, dim)
        return env_tokens
        

    def forward(self, ir_z, cc_depth_map, cc_src_loc):
        """
        Args:
           ir_z: (B, N, t, 1024)
           cc_depth_map: (B, 3, 256, 512)
           cc_src_loc: (B, N, 3)
        """
        bsz, irs_num = ir_z.shape[0], ir_z.shape[1]
        loc_tokens, room_tokens, ir_time_tokens = self.prepare_tokens(cc_depth_map, cc_src_loc, irs_num, bsz)
        # print(f"检查pre_embedding输出")
        # print(f"loc_tokens: {loc_tokens.shape}") #(B, N, 1, dim)
        # print(f"room_tokens: {room_tokens.shape}") #(B, ph*pw, dim)
        # print(f"ir_time_tokens: {ir_time_tokens.shape}") #(B, 1, T, dim)
        # print("--------------------------------")
        # #exit()
        ref_ir_tokens = ir_z[:,:-1]#(B, N-1, t, 1024)
        tgt_view_tokens = torch.cat([loc_tokens[:,-1:], ir_time_tokens], dim = -2)#(B, 1, 1+t, dim)
        ref_view_tokens = torch.cat([loc_tokens[:,:-1], ref_ir_tokens], dim = -2)#(B, N-1, 1+t, dim)

        ir_tokens, room_tokens, loc_tokens = self.aggregator(ref_view_tokens, tgt_view_tokens, room_tokens)
        # print(f"检查aggregator输出")
        # print(f"ir_tokens: {ir_tokens.shape}") #(B, N, t, dim*2)
        # print(f"room_tokens: {room_tokens.shape}") #(B, ph*pw, dim)
        # print(f"loc_tokens: {loc_tokens.shape}") #(B, N, 1, dim*2)
        # print("--------------------------------")
        # #exit()
        
        tgt_loc_tokens = loc_tokens[:, -1]#(B, 1, dim)
        tgt_ir_tokens = ir_tokens[:, -1]#(B, t, dim)
        tgt_ir_tokens = self.ir_decoder(tgt_ir_tokens)#(B, t, dim)
        # print(f"tgt_loc_tokens: {tgt_loc_tokens.shape}") #(B, 1, dim)
        # print(f"tgt_ir_tokens: {tgt_ir_tokens.shape}") #(B, t, dim)
        # print("--------------------------------")
        # exit()
        if self.env:
            env_tokens = self.env_decode(room_tokens, tgt_loc_tokens)
            return tgt_ir_tokens, env_tokens
        else:
            return tgt_ir_tokens

   