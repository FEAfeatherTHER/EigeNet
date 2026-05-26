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
from models.aa_v1_g2_align2_modality_zero.flow_matching_transformer.test_infer_pipeline import InferencePipeline as RIR_InferencePipeline
from models.layers.utils import compute_metrics, plot_waveform
import soundfile as sf

class testset_acousticrooms_dataset(Dataset):
    def __init__(self, cfg, jsonl):
        self.cfg = cfg
        self.frame_rate = cfg.preprocess.frame_rate
        self.sample_rate = cfg.preprocess.sample_rate
        self.duration = cfg.preprocess.duration
        self.downsample_rate = self.sample_rate // self.frame_rate
        self.depth_map_root = cfg.dataset.depth_map_root
        self.rir_root = cfg.dataset.rir_root
        self.metadata_root = cfg.dataset.metadata_root
        self.reference_count = cfg.preprocess.max_reference_count
        self.segment_list = self._load_file_list(jsonl)

    def _load_file_list(self, rir_list):
        segment_list = []
        segment_list.extend([json.loads(line) for line in open(rir_list, 'r', encoding='utf-8')])
        return segment_list

    def __len__(self):
        return len(self.segment_list)

    def get_receiver_source_location(self, ir_path):
        scene_name = ir_path.split("/")[-3]
        scene_id = ir_path.split("/")[-2]
        ir_file_name = ir_path.split("/")[-1]
        src_node, rec_node = int(ir_file_name.split("_")[0][1:]), int(ir_file_name.split("_")[1][1:])
        json_file_name = "S00" + str(src_node) + "_R00" + str(rec_node) + ".json"
        metadata_file_path = os.path.join(self.metadata_root, scene_name, scene_id, json_file_name)
        with open(metadata_file_path, "r") as fin:
            meta_info = json.load(fin)
        src_loc = meta_info["src_loc"]
        rec_loc = meta_info["rec_loc"]
        return src_loc, rec_loc

    def get_ref_ir_info(self, tgt_ir_path, other_src_idx_list):
        valid_other_src_ir_paths = []
        dir_name = os.path.dirname(tgt_ir_path)
        tgt_ir_file = os.path.basename(tgt_ir_path)
        tgt_rec_idx = tgt_ir_file.split("_")[1]
        for other_src_idx in other_src_idx_list:
            other_src_ir_file = f"S00{other_src_idx}_{tgt_rec_idx}_hybrid_IR.wav"
            other_src_ir_path = os.path.join(dir_name, other_src_ir_file)
            if os.path.exists(other_src_ir_path):
                valid_other_src_ir_paths.append(other_src_ir_path)
            else:
                print(f"*"*20)
                print(f"other_src_ir_path not found")
                print(other_src_ir_path)
                print("--------------------------------")
        valid_src_num = len(valid_other_src_ir_paths)
        if valid_src_num <= self.reference_count and valid_src_num > 0:
            ref_ir_paths = valid_other_src_ir_paths
        elif valid_src_num > self.reference_count:
            ref_ir_paths = random.sample(valid_other_src_ir_paths, self.reference_count)
        elif valid_src_num == 0:
            return [], 0
        
        ref_ir_info = []
        
        for ref_ir_path in ref_ir_paths:
            ref_src_loc, _ = self.get_receiver_source_location(ref_ir_path)
            ref_ir, ref_ir_len = _load_and_cut_audio(ref_ir_path, 0, self.duration, self.sample_rate)
            ref_ir_frames = ref_ir_len // self.downsample_rate
            ref_ir_info.append([ref_src_loc, ref_ir, ref_ir_frames])
        
        return ref_ir_info, valid_src_num

    def get_batch(self, segment_info):
        scene_name = segment_info['scene_name']
        scene_id = segment_info['scene_id']
        src_idx = segment_info['src_idx']
        rec_idx = segment_info['rec_idx']
        tgt_src_loc = segment_info['src_loc']
        tgt_rec_loc = segment_info['rec_loc']
        tgt_ir_path = segment_info['rir_path']
        other_src_idx_list = segment_info['other_src_idx_list']
        tgt_ir_file = os.path.basename(tgt_ir_path)
        save_path = f"{scene_id}__{tgt_ir_file}"
        tgt_ir, tgt_ir_len = _load_and_cut_audio(tgt_ir_path, 0, self.duration, self.sample_rate)
        tgt_ir_frames = tgt_ir_len // self.downsample_rate

        depth_path = segment_info['depth_path']

        pano_depth_map = np.load(depth_path)
        #convert to camera coordinate
        pano_depth_map = torch.from_numpy(pano_depth_map).float()# (256, 512, 3)
        
        cc_depth_map = convert_equirect_to_camera_coord(pano_depth_map, 256, 512).numpy()
        #(256, 512, 3) -> (3, 256, 512)
        cc_depth_map = rearrange(cc_depth_map, 'h w c -> c h w')

        #get ref_ir_info
        ref_src_loc_list = []
        ref_ir_list = []
        ref_ir_frames_list = []
        #检查非空
        if len(other_src_idx_list) == 0:
            return None
        else:
            ref_ir_info_list, valid_src_num = self.get_ref_ir_info(tgt_ir_path, other_src_idx_list)
            if valid_src_num == 0:
                return None
            for ref_ir_info in ref_ir_info_list:
                ref_src_loc_list.append(ref_ir_info[0])
                ref_ir_list.append(ref_ir_info[1])
                ref_ir_frames_list.append(ref_ir_info[2])
            
            #get all_ir_info
            all_src_loc_list = ref_src_loc_list + [tgt_src_loc]
            all_ir_list = ref_ir_list + [tgt_ir]
            all_ir_frames_list = ref_ir_frames_list + [tgt_ir_frames]
            all_ir = np.stack(all_ir_list, axis=0)
            rotation = 0
            all_cc_src_loc_list = [get_3d_point_camera_coord(rotation, tgt_rec_loc, src_loc) for src_loc in all_src_loc_list]
            all_cc_src_loc = np.stack(all_cc_src_loc_list, axis=0)
            all_ir_frames = np.array(all_ir_frames_list)
            max_frame = int(self.duration * self.frame_rate)


            return {
                "all_ir": all_ir, #(N, t)
                "all_cc_src_loc": all_cc_src_loc, #(N, 3)
                "cc_depth_map": cc_depth_map, #(256, 512, 3)
                "valid_src_num": valid_src_num, #(1,)
                "save_path": save_path, #(1,)
            }

    def __getitem__(self, idx):
        # segment_info = self.segment_list[idx]
        # batch = self.get_batch(segment_info)

        while True:
            segment_info = self.segment_list[idx]
            try:
                batch = self.get_batch(segment_info)
                if batch is None:
                    idx = idx + 1
                    print(f"idx: {idx} load data with empty ref_ir_info")
                    continue
                
                break
            except Exception as e:
                print(f"Error loading batch at index {idx} : {e}")
                idx = random.randint(0, len(self) - 1)
                print(f"replacing with index {idx}")
                continue
        return batch

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
    test_jsonl_list = args.test_jsonl_list
    if args.save:
        save_root = '/mnt/data/jingchong/eigenet/output_modality_zero'
        os.makedirs(save_root, exist_ok=True)
        
    for test_jsonl in test_jsonl_list:
        jsonl_name = test_jsonl.split("/")[-1].split(".")[0]
        split_name = jsonl_name.split("_")[-1]
        print(f"inference on {split_name} split")
        print()
        
        output_folder = os.path.join(src, f"data/output_modality_zero/{model_log_exp_name}/{train_step}/{split_name}")
        os.makedirs(output_folder, exist_ok=True)
        print(f"\noutput_folder: {output_folder}\n")
        if args.save:
            audio_save_dir = os.path.join(save_root, f"{model_log_exp_name}/{train_step}/{split_name}")
            os.makedirs(audio_save_dir, exist_ok=True)
            print(f"\naudio saved to: {audio_save_dir}\n")

        dataset = testset_acousticrooms_dataset(cfg, test_jsonl)
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
                    valid_src_num = segment["valid_src_num"]
                    if segment is None or valid_src_num < dynamic_reference_count:
                        continue
                    else:
                        batch_seg_list.append(segment)
                if len(batch_seg_list) == 0:
                    continue
                keys = batch_seg_list[0].keys()
                save_path_list = []
                for key in keys:
                    if key == "all_ir" or key == "all_cc_src_loc": #(N, 1, t)
                        sequences = [torch.from_numpy(item[key]).float()[-(dynamic_reference_count + 1):] for item in batch_seg_list]
                        packed_batch[key] = torch.stack(sequences, dim=0)#(b, N, ...)
                    elif  key == "cc_depth_map":
                        sequences = [torch.from_numpy(item[key]).float() for item in batch_seg_list]
                        packed_batch[key] = torch.stack(sequences, dim=0)
                    elif key == "save_path":
                        sequences = [item[key] for item in batch_seg_list]
                        save_path_list.extend(sequences)
                
                
                recon_audio = inference_pipeline.inference_fm(
                    batch=packed_batch,
                )#(b, 1, t)
                B = recon_audio.shape[0]

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
                    if args.save:
                        current_ref_num_audio_save_dir = os.path.join(audio_save_dir, f"{dynamic_reference_count}")
                        os.makedirs(current_ref_num_audio_save_dir, exist_ok=True)
                        for b in range(B):
                            ir = recon_audio[b, :]
                            if ir.ndim == 1:
                                pass
                            elif ir.ndim == 2:
                                ir = ir[0, :]
                            ir_to = os.path.join(current_ref_num_audio_save_dir, save_path_list[b])
                            sf.write(ir_to, ir, samplerate=16000)
                            
                    if hasattr(args, 'plot_interval'):
                        plot_interval = args.plot_interval
                        if plot_interval != 0:
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
            print(f"split_name: {split_name}")
            print(f"reference_count: {dynamic_reference_count}")
            print(f"edt_error: {total_edt_error}")
            print(f"c50_error: {total_c50_error}")
            print(f"t60_error: {total_t60_error}")
            print(f"outlier_count: {total_count_outlier}")
            print()
        
        metric_path = os.path.join(output_folder, f"{duration}_metrics_keep_ref_ac_4.csv")
        log.to_csv(metric_path, index = False)
        
if __name__ == "__main__":
    # no align
    fmt_cfg = os.path.join(src, f"egs/rir/flow_matching_transformer/debug_EigeNet_v1_g2_aa_noalign.json")
    fmt_ckpt = "/data/250010171/code/EigeNet_discriminant/ckpts/discriminant/base2_g2_noalign_debug/checkpoint/epoch-0009_step-0007600_loss-1.965324"
    
    # # align
    # fmt_cfg = os.path.join(src, f"egs/rir/flow_matching_transformer/debug_EigeNet_v1_g2_aa_align2.json")
    # fmt_ckpt = "/data/250010171/code/EigeNet_discriminant/ckpts/discriminant/base2_g2_align2_debug_layer6/checkpoint/epoch-0009_step-0007400_loss-2.084283"

    parser = argparse.ArgumentParser(description="Inference Script")
    args = parser.parse_args()

    args.test_jsonl_list = ["/data/250010171/code/EigeNet_discriminant/data/AcousticRooms_test_unseen.jsonl"]

    # args.test_jsonl_list = ["/data/250010171/code/EigeNet_discriminant/data/AcousticRooms_test_unseen.jsonl",
    # "/data/250010171/code/EigeNet_discriminant/data/AcousticRooms_test_seen.jsonl"]
    args.device = "cuda"
    args.test_duration = 0.363
    args.fmt_cfg = fmt_cfg
    args.fmt_ckpt = fmt_ckpt
    args.bsz = 20
    args.reference_count_list = [8]
    args.plot_interval = 0
    args.save = False

    main_debug(args)
    
    
        