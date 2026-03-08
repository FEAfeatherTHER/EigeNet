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
from models.mgm_v2_1.flow_matching_transformer.fmt_model import FlowMatchingTransformer
from models.mgm_v2_1.base.dac_codec import DAC
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
        device=None,
    ):
        self.device = device
        self.fmt_cfg = load_config(fmt_cfg_path)
        self.fmt_model = load_checkpoint(
            build_fmt_model, self.fmt_cfg, fmt_ckpt_path, device
        )
        print(f"#Params of Flow Matching model: {count_parameters(self.fmt_model)}")
        self.sample_rate = self.fmt_cfg.preprocess.sample_rate
        self.audio_codec = DAC(self.fmt_cfg.model.DAC.path)
        self.audio_codec.eval()
        self.audio_codec.to(self.device)
    
    @torch.no_grad()
    def encode_audio(self, audio): 
        bsz, irs_num, channel, audio_len = audio.shape
        audio = audio.reshape(-1, channel, audio_len)

        #codes = self.audio_codec.module.encode(audio) #(B*N, 12, t)
        codes = self.audio_codec.encode(audio)

        depth, frames = codes.shape[-2], codes.shape[-1]
        codes = codes.reshape(bsz, irs_num, depth, frames) #(B, N, 12, t)
        codes = codes.transpose(-1, -2) #(B, N, 12, t) -> (B, N, t, 12)

        return codes

    def inference_fm(
        self,
        batch
    ):  
        
        all_ir = batch["all_ir"].to(self.device)
        all_cc_src_loc = batch["all_cc_src_loc"].to(self.device)
        cc_depth_map = batch["cc_depth_map"].to(self.device)
        
        all_ir_codes = self.encode_audio(all_ir)
        irs_num = all_ir_codes.shape[1]
        ref_num = irs_num - 1
        ref_ac_tokens = all_ir_codes[:, :ref_num] #(B, N-1, T, 12)

        cond = (cc_depth_map, all_cc_src_loc)

        with torch.no_grad():
            predict_seq = self.fmt_model.reverse_diffusion(
                cond = cond,
                ref_ac_tokens = ref_ac_tokens,
              
            )#(b,t,d)

            predict_codes = predict_seq.transpose(1, 2) #(b, t, d) -> (b, d, t)
            synthesized_audio = self.audio_codec.decode(predict_codes)
            return synthesized_audio.cpu().numpy()
           