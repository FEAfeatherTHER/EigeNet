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
src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, src) 

import random
random.seed(42)
from einops import rearrange
from utils.util import load_config
from models.loss.evaluator import Evaluator
from models.EigeNet.src.infer_pipeline import InferencePipeline as EigeNet_InferencePipeline
from models.layers.utils import compute_metrics, plot_waveform
from models.dataset.haa_dataset_test import HAA_dataset_test

def main_debug(args):
    cfg = load_config(args.cfg)
    align_activate = cfg.model.eigenet_transformer.aligner.activate
    # initialize inference pipeline
    print("Initializing the inference pipeline...")
    device = args.device
    inference_pipeline = EigeNet_InferencePipeline(
        cfg_path=args.cfg,
        ckpt_path=args.ckpt,
        align_activate=align_activate,
        device=device
    )
    print(f"Pipeline initialized on device: {device}")
    
    # prepare stage
    # prepare dataset parameters
    duration = args.test_duration
    sample_rate = cfg.preprocess.sample_rate
    sample_length = int(duration * sample_rate)
    bsz = args.bsz

    # initialize evaluator
    evaluator = Evaluator()

    # create output folder
    ckpt = args.ckpt
    log_name = ckpt.split("/")[-4]
    exp_name = ckpt.split("/")[-3]
    model_log_exp_name = f"{log_name}_{exp_name}"
    #train_step = ckpt.split("/")[-1].split("_")[1]
    testing_scene_names = args.testing_scene_names
    log = pd.DataFrame(columns = ['scene_type', 'reference_count', 'edt_error', 'c50_error', 't60_error'])
    output_folder = os.path.join(src, f"data/output/{model_log_exp_name}/")
    os.makedirs(output_folder, exist_ok=True)
    print(f"output_folder: {output_folder}")
    print()
    for testing_scene_name in testing_scene_names:
        print(f"inference on {testing_scene_name}")
        print()
        scene_out_folder = os.path.join(output_folder, testing_scene_name)
        dataset = HAA_dataset_test(cfg, testing_scene_name)
        
        for dynamic_reference_count in args.reference_count_list:
            print(f"dynamic_reference_count: {dynamic_reference_count}")
            print()
        
            # initialize metrics
            edt_error_list = []
            c50_error_list = []
            t60_error_list = []


            # start inference
            print(f"start inference")
            print()
            for batch_idx in tqdm(range(0, len(dataset), bsz)):
                # last batch
                if batch_idx + bsz >= len(dataset):
                    present_bsz = len(dataset) - batch_idx
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
                    else:
                        continue
                
                all_ir = packed_batch["all_ir"]

                recon_audio,_ = inference_pipeline.inference(
                    batch=packed_batch,
                )#(b, 1, t)

                if recon_audio is not None:
                    # measure metrics
                    tgt_ir = packed_batch["all_ir"][:, -1].cpu().numpy() #(b, 1, t)
                    tgt_ir = tgt_ir[...,:sample_length]

                    recon_audio = recon_audio[...,:sample_length]
                    batch_edt_error_list, batch_c50_error_list, batch_t60_error_list, batch_count_outlier = compute_metrics(tgt_ir, recon_audio, evaluator)
                    edt_error_list.extend(batch_edt_error_list)
                    c50_error_list.extend(batch_c50_error_list)
                    t60_error_list.extend(batch_t60_error_list)
                    plot_folder = os.path.join(scene_out_folder, f"{dynamic_reference_count}")
                    os.makedirs(plot_folder, exist_ok=True)
                    
                    if hasattr(args, 'plot_interval') and args.plot_interval != 0:
                        plot_interval = args.plot_interval
                        if batch_idx % plot_interval == 0:
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
            log.loc[len(log)] = [testing_scene_name, dynamic_reference_count ,total_edt_error, total_c50_error, total_t60_error]
            print(f"testing_scene_name: {testing_scene_name}")
            print(f"reference_count: {dynamic_reference_count}")
            print(f"edt_error: {total_edt_error}")
            print(f"c50_error: {total_c50_error}")
            print(f"t60_error: {total_t60_error}")
            print()
        
    metric_path = os.path.join(output_folder, f"haa_metrics.csv")
    log.to_csv(metric_path, index = False)
        
if __name__ == "__main__":
    cfg = os.path.join(src, f"egs/rir/EigeNet/EigeNet_finetune_haa.json")
    ckpt = "path/to/checkpoint/eigenet_finetune_haa"
    
    parser = argparse.ArgumentParser(description="Inference Script")
    args = parser.parse_args()
    args.device = "cuda"
    args.test_duration = 0.363
    args.cfg = cfg
    args.ckpt = ckpt
    args.bsz = 1
    args.reference_count_list = [1, 4, 8]
    args.plot_interval = 0
    args.testing_scene_names = ["classroomBase", "hallwayBase", "dampenedBase", "complexBase"]
    main_debug(args)
    
    
        