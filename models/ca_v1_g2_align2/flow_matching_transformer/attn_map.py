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
from models.aa_v1.flow_matching_transformer.evaluator import Evaluator
from models.aa_v1.flow_matching_transformer.test_infer_pipeline import InferencePipeline as RIR_InferencePipeline
from models.layers.utils import compute_metrics, plot_waveform
from models.aa_v1.flow_matching_transformer.test_infer_debug import testset_acousticrooms_dataset

import matplotlib.pyplot as plt
import seaborn as sns

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
    attn_type = args.attn_type
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
    
    output_folder = os.path.join(src, f"data/attn_map/{attn_type}")
    os.makedirs(output_folder, exist_ok=True)
    layer_num = cfg.model.flow_matching_transformer.av_transformer.arch.depth
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
        block = infer_pipeline.fmt_model.av_transformer.transformer_blocks[layer_idx]
        # 使用 get_attn_hook(layer_idx) 确保每一层都有唯一的标识
        block.register_forward_hook(get_attn_hook(layer_idx))

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
            recon_audio = infer_pipeline.inference_fm(
                batch=packed_batch
            )#(b, 1, t)
            for key in attn_maps.keys():
                #去最大的头
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
    args.fmt_cfg = os.path.join(src, f"egs/rir/flow_matching_transformer/debug_EigeNet_v1_aa.json")
    args.fmt_ckpt = "/data/250010171/ckpts/EigeNet/discriminant/v1_aa_debug/checkpoint_backup/epoch-0008_step-0030000_loss-2.652895"
    args.bsz = 1
    args.attn_type = "aa"
    args.dataset_json = "/data/250010171/code/EigeNet_discriminant/data/AcousticRooms_test_mini.jsonl"
    visualize_attn_map(args)