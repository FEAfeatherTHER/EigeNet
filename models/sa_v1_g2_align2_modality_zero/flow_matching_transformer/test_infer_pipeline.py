import math
import json
import librosa
import torch
import torch.nn.functional as F
#import torchaudio
import accelerate
import numpy as np
import os
import yaml
from einops import rearrange
import random
import math
import numpy as np

from models.sa_v1_g2_align2_modality_zero.flow_matching_transformer.fmt_model import FlowMatchingTransformer
from models.layers.dac_codec import DAC
from utils.util import load_config


# Flow Matching Transformer
def build_fmt_model(cfg, device):
    model = FlowMatchingTransformer(cfg=cfg.model.flow_matching_transformer)
    model.eval()
    model.to(device)
    return model

def load_checkpoint(build_model_func, cfg, ckpt_path, device):
    model = build_model_func(cfg, device)
    accelerate.load_checkpoint_and_dispatch(model, ckpt_path)
    return model

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total_params < 1e6:
        return f"{total_params} params"  # Parameters
    elif total_params < 1e9:
        return f"{total_params / 1e6:.2f} M"  # Millions
    else:
        return f"{total_params / 1e9:.2f} B"  # Billions

def load_wav(sample_rate, wav_path, device):
    speech = librosa.load(wav_path, sr=sample_rate, mono = False)[0]
    # if len(speech.shape) == 1:
    #     speech = np.stack([speech, speech], axis=0)
    speech_tensor = torch.tensor(speech).unsqueeze(0).to(device)
    return speech_tensor

class InferencePipeline:
    def __init__(
        self,
        fmt_cfg_path=None,
        fmt_ckpt_path=None,
        align_activate=False,
        device=None,
    ):
        self.device = device
        self.align_activate = align_activate
        self.fmt_cfg = load_config(fmt_cfg_path)
        self.fmt_model = load_checkpoint(
            build_fmt_model, self.fmt_cfg, fmt_ckpt_path, device
        )
        print(f"#Params of Flow Matching model: {count_parameters(self.fmt_model)}")
        self.sample_rate = self.fmt_cfg.preprocess.sample_rate
        self._build_output_model()
    
    def _build_output_model(self):
        self.audio_codec = DAC(self.fmt_cfg.model.DAC.path)
        self.audio_codec.eval()
        self.audio_codec.to(self.device)
    

    @torch.no_grad()
    def encode_audio_to_z(self, audio): # 
        
        bsz, irs_num, channel, audio_len = audio.shape
        audio = audio.reshape(-1, channel, audio_len)

        z = self.audio_codec.encode_z(audio)#(B*N, 1024, t)

        dim, frames = z.shape[-2], z.shape[-1]
        z = z.reshape(bsz, irs_num, dim, frames) #(B, N, 1024, t)
        z = z.transpose(-1, -2) #(B, N, t, 1024)
        return z
    
    def decode_z_to_audio(self, z):
        pred_audio = self.audio_codec.decode_z(z)
        return pred_audio

    def inference_fm(
        self,
        batch,
    ):  
        
        all_ir = batch["all_ir"].to(self.device)
        all_cc_src_loc = batch["all_cc_src_loc"].to(self.device)
        cc_depth_map = batch["cc_depth_map"].to(self.device)
        
        all_ir_z = self.encode_audio_to_z(all_ir)
        irs_num = all_ir_z.shape[1]
        ref_num = irs_num - 1

        with torch.no_grad():
            if self.align_activate:
                predict_ir_z, predict_align_feat = self.fmt_model.forward(all_ir_z, cc_depth_map, all_cc_src_loc)
            else:
                predict_ir_z = self.fmt_model.forward(all_ir_z, cc_depth_map, all_cc_src_loc)
            predict_ir_z = predict_ir_z.transpose(1, 2) #(b, t, d) -> (b, d, t)
            synthesized_audio = self.decode_z_to_audio(predict_ir_z)   
            return synthesized_audio.cpu().numpy()
            