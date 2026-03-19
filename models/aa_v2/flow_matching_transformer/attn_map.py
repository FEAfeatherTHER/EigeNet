import numpy as np
import os
import sys
import argparse
import json
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import soundfile as sf
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) # AnyTrainer
sys.path.insert(0, src) # AnyTrainer

import random
random.seed(42)
from einops import rearrange
from utils.util import load_config
from models.dataset.acousticrooms_dataset import frame2mask, _load_and_cut_audio
from models.dataset.utils import get_3d_point_camera_coord, convert_equirect_to_camera_coord
from models.loss.evaluator import Evaluator
from models.aa_v2.flow_matching_transformer.test_infer_pipeline import InferencePipeline as RIR_InferencePipeline
from models.layers.utils import compute_metrics, plot_waveform
from models.aa_v2.flow_matching_transformer.test_infer_debug import testset_acousticrooms_dataset

import matplotlib.pyplot as plt
import seaborn as sns

def get_envelope(signal):
    """
    get the envelope of the signal 
    
    Args:
    - signal: Tensor, shape of (batch, samples) or (samples,)
    """
    if signal.dim() == 1:
        signal = signal.unsqueeze(0)  # (1, samples)

    # --- (Hilbert Transform) ---
    N = signal.shape[-1]
    Xf = torch.fft.fft(signal)
    h = torch.zeros(N, device=signal.device)
    
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2
        
    Xf = Xf * h
    z = torch.fft.ifft(Xf)
    envelope = torch.abs(z)[...,:N]
    return envelope  

def smooth_envelope(envelope, window_size=512, normalized = False):
    # --- (Moving Average) ---
    N = envelope.shape[-1]
    kernel = torch.ones((1, 1, window_size), device=envelope.device) / window_size
    pad_size = window_size // 2
    envelope_padded = F.pad(envelope.unsqueeze(1), (pad_size, pad_size), mode='constant', value=0)
    smoothed_envelope = F.conv1d(envelope_padded, kernel)
    
    if normalized:
        env_max_val = torch.max(smoothed_envelope, dim=-1, keepdim=True)[0]
        smoothed_envelope = smoothed_envelope / (env_max_val + 1e-7)

    return smoothed_envelope[:, 0, :N]


def plot_attn(attn_tensor, head_idx=None, png_path = None):
    # 转换为 numpy
    data = attn_tensor
    
    if head_idx is not None:
        # 画特定的某一个头
        map_to_plot = data[0, head_idx]
        title = f"Head {head_idx}"
    else:
        # 画所有头的平均
        map_to_plot = data[0].mean(axis=0)
        title = "Average Attention"

    plt.figure(figsize=(10, 8))
    sns.heatmap(map_to_plot, cmap='viridis', xticklabels=10, yticklabels=10)
    plt.title(title)
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    if png_path is not None:
        plt.savefig(png_path)

def visualize_attn_map(args):
    device = args.device
    cfg = load_config(args.fmt_cfg)
    bsz = args.bsz
    model_name = os.path.splitext(os.path.basename(args.fmt_cfg))[0]
    dataset_json = args.dataset_json
    dynamic_reference_count = 1
    # classifier_free_guidance = args.classifier_free_guidance
    # print(f"Inference with Classifier free guidance: {classifier_free_guidance}")
    print(f"Pipeline initialized on device: {device}")
    infer_pipeline = RIR_InferencePipeline(
        fmt_cfg_path=args.fmt_cfg,
        fmt_ckpt_path=args.fmt_ckpt,
        device=args.device
    )
    
    """定义需要可视化的层"""
    layer_name = 'env_decoder'
    layer_num = cfg.model.flow_matching_transformer.env_decoder.arch.depth
    attn_maps = {f"layer_{i}": None for i in range(layer_num)}
    def get_attn_hook(layer_id):
        def hook(module, module_input, module_output):
            # module_output 是 (x, weights) 的 tuple
            # 确保只在返回了 weights 的情况下提取
            if isinstance(module_output, (tuple, list)) and len(module_output) > 1:
                attn_weight = module_output[1] # 假设第二个是 weights
                if attn_weight is not None:
                    attn_maps[f"layer_{layer_id}"] = attn_weight.detach()
        return hook
    
    for layer_idx in range(layer_num):
        #block = infer_pipeline.fmt_model.aa_transformer.transformer_blocks[layer_idx]
        block = infer_pipeline.fmt_model.env_decoder[0].transformer_blocks[layer_idx]
        # 使用 get_attn_hook(layer_idx) 确保每一层都有唯一的标识
        block.register_forward_hook(get_attn_hook(layer_idx))

    
    output_folder = os.path.join(src, f"data/attn_map/{model_name}/{layer_name}")
    os.makedirs(output_folder, exist_ok=True)
    dataset = testset_acousticrooms_dataset(cfg, dataset_json)
    for batch_idx in tqdm(range(0, len(dataset), bsz)):
        # last batch
        if batch_idx + bsz >= len(dataset):
            present_bsz = len(dataset) - 1 - batch_idx
        else:
            present_bsz = bsz
        # assemble batch
        batch_seg_list = []
        packed_batch = dict()
        for i in range(present_bsz):
            # last batch
            if batch_idx + bsz >= len(dataset):
                present_bsz = len(dataset) - 1 - batch_idx
            else:
                present_bsz = bsz
            # assemble batch
            batch_seg_list = []
            packed_batch = dict()
            for i in range(present_bsz):
                segment = dataset.__getitem__(batch_idx + i)
                valid_src_num = segment["valid_src_num"]
                if segment is None or valid_src_num < dynamic_reference_count:
                    continue
                else:
                    batch_seg_list.append(segment)
            keys = batch_seg_list[0].keys()
            
            for key in keys:
                if key == "all_ir" or key == "all_cc_src_loc": #(N, 1, t)
                    sequences = [torch.from_numpy(item[key]).float()[-(dynamic_reference_count + 1):] for item in batch_seg_list]
                    packed_batch[key] = torch.stack(sequences, dim=0)#(b, N, ...)
                elif  key == "cc_depth_map":
                    sequences = [torch.from_numpy(item[key]).float() for item in batch_seg_list]
                    packed_batch[key] = torch.stack(sequences, dim=0)
            ref_rir = packed_batch["all_ir"][0, 0, :].cpu().numpy()
            tgt_rir = packed_batch["all_ir"][0, 1, :].cpu().numpy()
            fig, axs = plt.subplots(2, 1, figsize=(10, 10))
            plot_waveform(ref_rir, 16000, title="Ref", ax=axs[0])
            plot_waveform(tgt_rir, 16000, title="Tgt", ax=axs[1])
            plt.savefig(os.path.join(output_folder, f"ref_tgt.png"))
            plt.close()
            recon_wav, recon_env = infer_pipeline.inference_fm(
                batch=packed_batch,
                return_env=True
            )#(b, 1, t)
            print(f"Recon wav shape: {recon_wav.shape}")
            print(f"Recon env shape: {recon_env.shape}")
            gt_env = smooth_envelope(get_envelope(torch.from_numpy(tgt_rir).float())).cpu().numpy()
            print(f"Gt env shape: {gt_env.shape}")
            env_path = os.path.join(output_folder, f"env.png")
            fig, axs = plt.subplots(2, 1, figsize=(10, 10))
            plot_waveform(gt_env, 16000, title="gt Env", ax=axs[0])
            plot_waveform(recon_env[0], 16000, title="Recon Env", ax=axs[1])
            plt.savefig(env_path)
            plt.close()
            for key in attn_maps.keys():
                layer_attn_map = attn_maps[key]
                layer_attn_map = layer_attn_map.cpu().numpy()
                print(f"{key}: {layer_attn_map.shape}")
                #save the attn_maps to the output_folder
                npy_path = os.path.join(output_folder, f"{key}.npy")
                png_path = os.path.join(output_folder, f"{key}.png")
                np.save(npy_path, layer_attn_map)
                plot_attn(layer_attn_map, png_path=png_path)
            exit()
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference Script")
    args = parser.parse_args()
    args.device = "cuda"
    args.fmt_cfg = os.path.join(src, f"egs/rir/flow_matching_transformer/debug_EigeNet_v2_aa.json")
    args.fmt_ckpt = "/data/250010171/ckpts/EigeNet/discriminant/v2_aa_debug/checkpoint_backup/epoch-0010_step-0015000_loss-2.472556"
    args.bsz = 1
    args.dataset_json = "/data/250010171/code/EigeNet_discriminant/data/AcousticRooms_test_mini.jsonl"
    visualize_attn_map(args)