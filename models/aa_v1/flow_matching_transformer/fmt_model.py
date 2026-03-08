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
random.seed(114)
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel
from typing import List, Optional, Tuple, Union
from einops import rearrange
from transformers.models.llama.modeling_llama import BaseModelOutputWithPast

#from models.backbone.llama_nar import DiffLlama
#from models.backbone.aadit import AADiT
#from models.backbone.dit import DiT
from models.layers.transformer_encoder import Transformer_Encoder, AA_Transformer_Encoder
from models.layers.patch_embed import PatchEmbed
from models.layers.mlp import Mlp
from models.layers.modules import (SoftEmbedding, 
                                ConvNeXtV2Block,
                                ConvPositionEmbedding,
                                SinusPositionEmbedding)


class InputEmbedding(nn.Module):
    def __init__(self, x_dim, cond_dim, out_dim):
        super().__init__()
        self.linear = nn.Sequential(nn.SiLU(), nn.Linear(x_dim+cond_dim, out_dim))
    def forward(self, x, cond):
        """
            x: (B, N*T, D)
            cond: (B, N, T ,D)
        """
        bsz, irs_num, frames, _ = cond.shape
        x = rearrange(x, "b (n t) c -> b n t c", n = irs_num)#(B, N, T, hidden_size)
        frames = x.shape[2]
        x = x.reshape(-1, frames, x.shape[-1]) #(B*N, T, ...)
        cond = cond.reshape(-1, frames, cond.shape[-1]) #(B*N, T, ...)
        x = torch.cat([x, cond], dim=-1)
        x = self.linear(x)#(B*N, T, hidden_size)
        x = x.reshape(bsz, -1, x.shape[-1])#(B, N*T, hidden_size)
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
        
        hidden_size = self.cfg.hidden_size
        self.midi_dim = (cfg.midi_dim 
            if hasattr(cfg, 'midi_dim') else 512
        )
       
        self.hidden_size = hidden_size
        self.v_dim = cfg.visual_transformer.arch.dim
        pre_dim = 128

        self.visual_transformer = Transformer_Encoder(**cfg.visual_transformer.arch)
        self.av_transformer = AA_Transformer_Encoder(**cfg.AV_transformer.arch)
        
        self.loc_embedding = SinusPositionEmbedding(dim=pre_dim)
        self.loc_mlp = nn.Linear(3*pre_dim, self.v_dim)

        self.patchify = PatchEmbed(img_size=(256, 512), 
            patch_size=(16, 16), 
            in_chans=3, 
            embed_dim=self.v_dim)

        frame_size = int(duration * frame_rate)
        time_interval = torch.linspace(0, duration, frame_size)
        self.register_buffer("time_interval", time_interval)
        self.time_interval_embedding = SinusPositionEmbedding(dim=pre_dim)
        self.time_interval_mlp = nn.Linear(pre_dim, self.hidden_size)

        self.out_proj = Mlp(in_features=self.hidden_size, hidden_features=self.hidden_size * 2, out_features=self.hidden_size)
        
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
    
    def get_spatial_tokens(self, depth_map, loc):
        """
        Args:
           depth_map: (B, 3, 256, 512)
           loc: (B, N, 3)
        """
        bsz, irs_num, _ = loc.shape

        loc = loc.reshape(-1, loc.shape[-1])

        loc = loc.reshape(-1)

        loc_tokens = self.loc_embedding(loc)#(B*N*3, pre_dim)
        loc_tokens = loc_tokens.reshape(-1, 3, loc_tokens.shape[-1]) #（B*N, 3, pre_dim）
        new_bsz = loc_tokens.shape[0]
        loc_tokens = loc_tokens.reshape(new_bsz, -1) #(B*N, 3*pre_dim)
        loc_tokens = self.loc_mlp(loc_tokens).unsqueeze(1) #(B*N, 1, dim)
        loc_tokens = loc_tokens.reshape(bsz, irs_num, 1, loc_tokens.shape[-1]) #(B, N, 1, dim)
        depth_tokens = self.patchify(depth_map) #(B, h*w, dim)
        return loc_tokens, depth_tokens, bsz, irs_num

    def time_embedding(self, time_interval):
        """
        time_interval:(T,)
        """
        time_ebd = self.time_interval_embedding(time_interval) #(T, pre_dim)
        time_ebd = self.time_interval_mlp(time_ebd) #(T, dim)
        return time_ebd

    def pre_embedding(self, depth_map, loc):
        time_tokens = self.time_embedding(self.time_interval) #(T, dim)

        loc_tokens, depth_tokens, bsz, irs_num = self.get_spatial_tokens(depth_map, loc)
        depth_tokens = depth_tokens.repeat(irs_num, 1, 1)
        loc_tokens = rearrange(loc_tokens, "b n t c -> (b n) t c")
        v_tokens = torch.cat([loc_tokens, depth_tokens], dim=1)
        v_tokens = self.visual_transformer(v_tokens)
        depth_tokens = v_tokens[:, 1:]
        loc_tokens = v_tokens[:, :1]
        depth_global_tokens = torch.mean(depth_tokens, dim=1, keepdim=True)
        depth_global_tokens = rearrange(depth_global_tokens, "(b n) t c -> b n t c", n = irs_num)
        loc_tokens = rearrange(loc_tokens, "(b n) t c -> b n t c", n = irs_num)

        time_tokens = time_tokens.unsqueeze(0).unsqueeze(0).repeat(bsz, 1, 1, 1) #(B, 1, T, dim)
        return loc_tokens, depth_global_tokens, time_tokens

    def audiovisual_encoder(self, ref_view_av_tokens, tgt_view_av_tokens):
        all_view_av_tokens = torch.cat([ref_view_av_tokens, tgt_view_av_tokens], dim = 1)#(B, N, t+2, dim)
        tgt_ir_hidden_states = self.av_transformer(all_view_av_tokens)#(B, N, t+2, dim)
        
        return tgt_ir_hidden_states

    def forward(self, ir_z, cc_depth_map, cc_src_loc):
        """
        Args:
           ir_z: (B, N, t, 1024)
           cc_depth_map: (B, 3, 256, 512)
           cc_src_loc: (B, N, 3)
        """
        # print(f"*"*20)
        # print(f"检查fmt_model输入")
        # print(f"ir_z: {ir_z.shape}")
        # print(f"cc_depth_map: {cc_depth_map.shape}")
        # print(f"cc_src_loc: {cc_src_loc.shape}")
        # print("--------------------------------")
        # # exit()

        loc_tokens, depth_tokens, time_tokens = self.pre_embedding(cc_depth_map, cc_src_loc)
        #loc_tokens: (B, N, 1, dim), 
        # depth_tokens: (B, N, 1, dim), 
        # time_tokens: (B, 1, T, dim)
        
        # print(f"*"*20)
        # print(f"检查pre_embedding输出")
        # print(f"loc_tokens: {loc_tokens.shape}")
        # print(f"depth_tokens: {depth_tokens.shape}")
        # print(f"time_tokens: {time_tokens.shape}")
        # print("--------------------------------")
        # #exit()
        ref_ir_tokens = ir_z[:,:-1]#(B, N-1, t, 1024)
        tgt_view_av_tokens = torch.cat([loc_tokens[:,-1:], depth_tokens[:,-1:], time_tokens], dim = -2)#(B, 1, 2+t, dim)
        ref_view_av_tokens = torch.cat([loc_tokens[:,:-1], depth_tokens[:,:-1], ref_ir_tokens], dim = -2)#(B, N-1, t+2, dim)

        tgt_ir_hidden_states = self.audiovisual_encoder(ref_view_av_tokens, tgt_view_av_tokens)#(B, N, t+2, dim)

        pred_tgt_z = self.out_proj(tgt_ir_hidden_states)#(B, t, dim)
        
        return pred_tgt_z

   