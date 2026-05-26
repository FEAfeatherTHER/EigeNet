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
ROOT = '/data/share/amphion/data/noise-and-rirs/AcousticRooms/'
import random
random.seed(42)
from einops import rearrange
from utils.util import load_config
from models.dataset.acousticrooms_dataset import frame2mask, _load_and_cut_audio
from models.dataset.utils import get_3d_point_camera_coord, convert_equirect_to_camera_coord
from models.loss.evaluator import Evaluator
from models.aa_v1_g2_align2_toy.flow_matching_transformer.test_infer_pipeline import InferencePipeline as RIR_InferencePipeline
from models.layers.utils import compute_metrics, plot_waveform
import soundfile as sf

def prepare(args, cfg):
    align_activate = cfg.model.flow_matching_transformer.aligner.activate
    inference_pipeline = RIR_InferencePipeline(
        fmt_cfg_path=args.fmt_cfg,
        fmt_ckpt_path=args.fmt_ckpt,
        align_activate=align_activate,
        device=args.device
    )
    
    device = args.device
    print(f"Pipeline initialized on device: {device}")
    evaluator = Evaluator()
    return inference_pipeline, evaluator


def get_batch(data_json_path, reference_count = 8, duration = 0.5, sample_rate = 16000):
    data_json = json.load(open(data_json_path, 'r'))
    tgt_depth_path = data_json['tgt_depth_path']
    depth_map = torch.from_numpy(np.load(os.path.join(ROOT, tgt_depth_path))).float()
    cc_depth_map = convert_equirect_to_camera_coord(depth_map, 256, 512)
    cc_depth_map = rearrange(cc_depth_map, 'h w c -> c h w')
    tgt_ir = torch.from_numpy(_load_and_cut_audio(os.path.join(ROOT, data_json['tgt_rir_path']), 0, duration, sample_rate)[0]).float()
    tgt_src_loc = data_json['tgt_src_loc']
    tgt_rec_loc = data_json['tgt_rec_loc']
    references = data_json['references']
    if len(references) < reference_count:
        raise ValueError(f"Reference count is greater than the number of references: {len(references)}")
    random.shuffle(references)
    ref_src_loc_list = []
    ref_rir_list = []
    ref_ir_frames_list = []
    for reference in references[:reference_count]:
        ref_src_loc_list.append(reference['src_loc'])
        ref_rir_list.append(torch.from_numpy(_load_and_cut_audio(os.path.join(ROOT, reference['rir_path']), 0, duration, sample_rate)[0]).float())
    all_ir_list = ref_rir_list + [tgt_ir]
    all_src_loc_list = ref_src_loc_list + [tgt_src_loc]
    all_cc_src_loc_list = [get_3d_point_camera_coord(0, tgt_rec_loc, src_loc) for src_loc in all_src_loc_list]
    all_ir = torch.from_numpy(np.stack(all_ir_list, axis=0)).float()
    all_cc_src_loc = torch.from_numpy(np.stack(all_cc_src_loc_list, axis=0)).float()
    return {
        "all_ir": all_ir.unsqueeze(0), #(1, N, t)
        "all_cc_src_loc": all_cc_src_loc.unsqueeze(0), #(1, N, 3)
        "cc_depth_map": cc_depth_map.unsqueeze(0), #(1, 256, 512, 3)
    }

def main_debug(inference_pipeline, evaluator, cfg, batch):
    """主执行函数"""
    sample_length = int(cfg.preprocess.duration * cfg.preprocess.sample_rate)
    recon_audio, predict_align_feature = inference_pipeline.inference_fm(
        batch=batch,
    )#(b, 1, t)

    tgt_ir = batch["all_ir"][:, -1].cpu().numpy() #(b, 1, t)
    tgt_ir = tgt_ir[...,:sample_length]
    recon_audio = recon_audio[...,:sample_length]
    batch_edt_error_list, batch_c50_error_list, batch_t60_error_list, batch_count_outlier = compute_metrics(tgt_ir, recon_audio, evaluator)
    edt_error = batch_edt_error_list[0]
    c50_error = batch_c50_error_list[0]
    t60_error = batch_t60_error_list[0]
    min_length = min(tgt_ir.shape[-1], recon_audio.shape[-1])
    tgt_ir = tgt_ir[0, 0, :min_length]#(1, t)
    recon_audio = recon_audio[..., :min_length]
    if recon_audio.ndim == 3:
        recon_audio = recon_audio[0, 0]
    elif recon_audio.ndim == 2:
        recon_audio = recon_audio[0]
    return edt_error, c50_error, t60_error, tgt_ir, recon_audio, predict_align_feature
    
        
if __name__ == "__main__":
    # # no align
    # fmt_cfg = os.path.join(f'/data/250010171/code/EigeNet_discriminant/egs/rir/flow_matching_transformer/debug_EigeNet_v1_g2_aa_align2.json')
    # fmt_ckpt = "/data/250010171/code/EigeNet_discriminant/ckpts/discriminant/base2_aa_align_toy1.2/checkpoint_backup/epoch-0008_step-0006400_loss-2.312797"
    
    # align
    fmt_cfg = os.path.join(src, f"egs/rir/flow_matching_transformer/debug_EigeNet_v1_g2_aa_align2_toy.json")
    fmt_ckpt = "/data/250010171/code/EigeNet_discriminant/ckpts/discriminant/base2_aa_align_toy1.3/checkpoint_backup/epoch-0012_step-0009600_loss-2.107707"

    cfg = load_config(fmt_cfg)
    data_json_path = '/data/250010171/code/EigeNet_discriminant/visualization/visualization_data1.json'
    parser = argparse.ArgumentParser(description="Inference Script")
    args = parser.parse_args()
    args.device = "cuda"
    args.test_duration = 0.363
    args.fmt_cfg = fmt_cfg
    args.fmt_ckpt = fmt_ckpt
    out_dir = '/data/250010171/code/EigeNet_discriminant/data/debug/toy_1.3'
    os.makedirs(out_dir, exist_ok=True)

    #主程序
    #初始化模型
    inference_pipeline, evaluator = prepare(args, cfg)
    #确认reference数量
    reference_count = 8
    batch = get_batch(data_json_path, reference_count)
    #推理
    edt_error, c50_error, t60_error, gt_ir, pred_ir, predict_align_feature = main_debug(inference_pipeline, evaluator, cfg, batch)
    print(f"edt_error: {edt_error}")
    print(f"c50_error: {c50_error}")
    print(f"t60_error: {t60_error}")

    #gt_ir_path = os.path.join(out_dir, 'gt_ir.wav')
    pred_ir_path = os.path.join(out_dir, 'pred_ir_ours_wo_align.wav')
    #sf.write(gt_ir_path, gt_ir, samplerate=16000)
    sf.write(pred_ir_path, pred_ir, samplerate=16000)
    #print(f"save gt_ir to {gt_ir_path}")
    print(f"save pred_ir to {pred_ir_path}")

    print(f"predict_align_feature: {predict_align_feature}")
    pred_align_feature_path = os.path.join(out_dir, 'pred_align_feature.npy')
    np.save(pred_align_feature_path, predict_align_feature.numpy())
    print(f"save pred_align_feature to {pred_align_feature_path}")
    
    
    
        