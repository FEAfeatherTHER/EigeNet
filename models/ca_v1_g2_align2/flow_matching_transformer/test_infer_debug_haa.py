import numpy as np
import os
import sys
import argparse
import json
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import glob
src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) # AnyTrainer
sys.path.insert(0, src) # AnyTrainer

import random
random.seed(42)
from einops import rearrange
from utils.util import load_config
from models.loss.evaluator import Evaluator
from models.aa_v1_g2_align2.flow_matching_transformer.test_infer_pipeline import InferencePipeline as RIR_InferencePipeline
from models.layers.utils import compute_metrics, plot_waveform
from models.dataset.haa_dataset_test import HAA_dataset_test
HAA_ROOT = '/data/share/amphion/data/noise-and-rirs/haa'

def main_debug(args):
    """主执行函数"""
    cfg = load_config(args.fmt_cfg)
    align_activate = cfg.model.flow_matching_transformer.aligner.activate
    # 初始化模型
    print("Initializing the inference pipeline...")
    inference_pipeline = RIR_InferencePipeline(
        fmt_cfg_path=args.fmt_cfg,
        fmt_ckpt_path=args.fmt_ckpt,
        align_activate=align_activate,
        device=args.device
    )
    
    device = args.device
    # classifier_free_guidance = args.classifier_free_guidance
    # print(f"Inference with Classifier free guidance: {classifier_free_guidance}")
    print(f"Pipeline initialized on device: {device}")
    
    # 准备阶段
    # 准备dataset超参
    duration = args.test_duration
    sample_rate = cfg.preprocess.sample_rate
    sample_length = int(duration * sample_rate)
    bsz = args.bsz

    # 准备 evaluator
    evaluator = Evaluator()

    # 创建生成结果目录
    fmt_ckpt = args.fmt_ckpt
    log_name = fmt_ckpt.split("/")[-4]
    exp_name = fmt_ckpt.split("/")[-3]
    model_log_exp_name = f"{log_name}_{exp_name}"
    train_step = fmt_ckpt.split("/")[-1].split("_")[1]
    haa_json_path = args.haa_jsonl_path
    testing_scene_names = args.testing_scene_names
    for testing_scene_name in testing_scene_names:
        print(f"inference on {testing_scene_name}")
        print()
        output_folder = os.path.join(src, f"data/output/{model_log_exp_name}/{train_step}/{testing_scene_name}")
        os.makedirs(output_folder, exist_ok=True)
        print(f"output_folder: {output_folder}")
        print()
        dataset = HAA_dataset_test(cfg, testing_scene_name)
        log = pd.DataFrame(columns = ['reference_count', 'edt_error', 'c50_error', 't60_error', 'outlier_count'])
        
        for dynamic_reference_count in args.reference_count_list:
            print(f"dynamic_reference_count: {dynamic_reference_count}")
            print()
        
            # 初始化指标
            edt_error_list = []
            c50_error_list = []
            t60_error_list = []
            total_count_outlier = 0

            # 生成管线
            print(f"start inference")
            print()
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
                    segment = dataset.__getitem__(batch_idx + i)
                    batch_seg_list.append(segment)
                keys = batch_seg_list[0].keys()
                
                for key in keys:
                    if key == "all_ir" or key == "all_cc_src_loc": #(N, 1, t)
                        sequences = [torch.from_numpy(item[key]).float()[-(dynamic_reference_count + 1):] for item in batch_seg_list]
                        packed_batch[key] = torch.stack(sequences, dim=0)#(b, N, ...)
                    elif  key == "cc_depth_map":
                        sequences = [torch.from_numpy(item[key]).float() for item in batch_seg_list]
                        packed_batch[key] = torch.stack(sequences, dim=0)
                
                all_ir = packed_batch["all_ir"]

                recon_audio = inference_pipeline.inference_fm(
                    batch=packed_batch,
                )#(b, 1, t)

                if recon_audio is not None:
                    # 测量metric
                    tgt_ir = packed_batch["all_ir"][:, -1].cpu().numpy() #(b, 1, t)
                    tgt_ir = tgt_ir[...,:sample_length]
                    recon_audio = recon_audio[...,:sample_length]
                    batch_edt_error_list, batch_c50_error_list, batch_t60_error_list, batch_count_outlier = compute_metrics(tgt_ir, recon_audio, evaluator)
                    edt_error_list.extend(batch_edt_error_list)
                    c50_error_list.extend(batch_c50_error_list)
                    t60_error_list.extend(batch_t60_error_list)
                    total_count_outlier += batch_count_outlier
                    plot_folder = os.path.join(output_folder, f"{dynamic_reference_count}")
                    os.makedirs(plot_folder, exist_ok=True)
                    
                    if hasattr(args, 'plot_interval'):
                        plot_interval = args.plot_interval
                        if batch_idx % plot_interval == 0:
                            #output_path = os.path.join(output_folder, f"{audio_name}.wav")
                            #sf.write(output_path, recon_audio, samplerate=16000)
                            gt_tgt_ir = tgt_ir[...,:sample_length][0]
                            pred_tgt_ir = recon_audio[...,:sample_length][0]
                            fig, axs = plt.subplots(2, 1, figsize=(10, 10))

                            plot_waveform(gt_tgt_ir, sample_rate, title="GT", ax=axs[0])
                            plot_waveform(pred_tgt_ir, sample_rate, title="Pred", ax=axs[1])
                            pdf_path = os.path.join(plot_folder, f"{batch_idx}.pdf")
                            plt.savefig(pdf_path)
                            plt.close()

                else:
                    print(f"Recon audio is None, skip this audio")
                    continue
            total_edt_error = round(np.mean(edt_error_list), 3)
            total_c50_error = round(np.mean(c50_error_list), 3)
            total_t60_error = round(np.mean(t60_error_list), 3)
            log.loc[len(log)] = [dynamic_reference_count ,total_edt_error, total_c50_error, total_t60_error, total_count_outlier]
            print(f"testing_scene_name: {testing_scene_name}")
            print(f"reference_count: {dynamic_reference_count}")
            print(f"edt_error: {total_edt_error}")
            print(f"c50_error: {total_c50_error}")
            print(f"t60_error: {total_t60_error}")
            print(f"outlier_count: {total_count_outlier}")
            print()
        
        metric_path = os.path.join(output_folder, f"{duration}_metrics_shot.csv")
        log.to_csv(metric_path, index = False)
        
if __name__ == "__main__":
    # fmt_cfg = os.path.join(src, f"egs/rir/flow_matching_transformer/debug_EigeNet_v1_g2_aa_noalign.json")
    # fmt_ckpt = "/data/250010171/ckpts/EigeNet/discriminant/base2_g2_noalign_debug/checkpoint/epoch-0009_step-0007400_loss-2.125936"
    fmt_cfg = os.path.join(src, f"egs/rir/flow_matching_transformer/debug_EigeNet_v1_g2_aa_align2_finetune_haa.json")
    fmt_ckpt = "/data/250010171/code/EigeNet_discriminant/ckpts/discriminant/base2_g2_align2_debug_layer6_finetune_haa/checkpoint/epoch-0007_step-0000640_loss-2.612913"
    
    parser = argparse.ArgumentParser(description="Inference Script")
    args = parser.parse_args()
    args.haa_jsonl_path = "/data/250010171/code/EigeNet_discriminant/data/haa_new.jsonl"
    args.device = "cuda"
    args.test_duration = 0.363
    args.fmt_cfg = fmt_cfg
    args.fmt_ckpt = fmt_ckpt
    args.bsz = 20
    args.reference_count_list = [8]
    args.plot_interval = 100
    #args.testing_scene_names = ["classroomBase"]
    args.testing_scene_names = ["classroomBase", "hallwayBase", "dampenedBase", "complexBase"]


    main_debug(args)
    
    
        