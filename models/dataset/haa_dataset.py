import numpy as np
import os
import sys
import json
import torch.nn.functional as F
from pathlib import Path
import soundfile as sf
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import glob
src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
sys.path.insert(0, src) 

import random
random.seed(42)
from einops import rearrange
from models.dataset.acousticrooms_dataset import frame2mask, _load_and_cut_audio
from models.dataset.utils import get_3d_point_camera_coord, convert_equirect_to_camera_coord
from utils.util import load_config
HAA_ROOT = 'path/to/haa'
SCENE_NAMES = ['classroomBase', 'complexBase', 'hallwayBase', 'dampenedBase']

class HAA_dataset(Dataset):
    def __init__(self, cfg, split = 'train'):
        self.cfg = cfg
        if split == 'train':
            self.jsonl_path = cfg.dataset.haa_train_path
        elif split == 'test':
            self.jsonl_path = cfg.dataset.haa_test_path
        self.frame_rate = cfg.preprocess.frame_rate
        self.sample_rate = cfg.preprocess.sample_rate
        self.duration = cfg.preprocess.duration
        self.downsample_rate = self.sample_rate // self.frame_rate
        self.reference_count = cfg.preprocess.max_reference_count
        self.segment_list = self._load_file_list(self.jsonl_path)
        
        #depth_path = os.path.join(HAA_ROOT, testing_scene_name, "depth.npy")
        #pano_depth_map = np.load(depth_path)
        #convert to camera coordinate
        #pano_depth_map = torch.from_numpy(pano_depth_map).float()# (256, 512, 3)
        #cc_depth_map = convert_equirect_to_camera_coord(pano_depth_map, 256, 512).numpy()
        #(256, 512, 3) -> (3, 256, 512)
        #self.cc_depth_map = rearrange(cc_depth_map, 'h w c -> c h w')
        self.metadata = {}
        for scene_name in SCENE_NAMES:
            metadata_path = os.path.join(HAA_ROOT, scene_name, "xyzs.npy")
            self.metadata[scene_name] = np.load(metadata_path)
        

    def _load_file_list(self, jsonl_path):
        segment_list = []
        segment_list.extend([json.loads(line) for line in open(jsonl_path, 'r', encoding='utf-8')])
        return segment_list

    def __len__(self):
        return len(self.segment_list)

    def get_receiver_source_location(self, rir_path):
        scene_name = Path(rir_path).stem.split('_')[-2]
        src_idx = int(Path(rir_path).stem.split('_')[-1])
        src_locs = self.metadata[scene_name][src_idx].copy()
        return src_locs
        

    def get_ref_ir_info(self, tgt_ir_path, tgt_src_idx):
        dir_name = os.path.dirname(tgt_ir_path)
        all_rir_files = set(glob.glob(os.path.join(dir_name, '*.wav')))
        all_other_rir_files = all_rir_files.difference(set([tgt_ir_path]))
        chosen_other_rir_files = random.sample(all_other_rir_files, self.reference_count)
        ref_ir_info = []
        
        for ref_ir_path in chosen_other_rir_files:
            ref_src_locs = self.get_receiver_source_location(ref_ir_path)
            ref_ir, ref_ir_len = _load_and_cut_audio(ref_ir_path, 0, self.duration, self.sample_rate)
            ref_ir_frames = ref_ir_len // self.downsample_rate
            ref_ir_info.append([ref_src_locs, ref_ir, ref_ir_frames])
        
        return ref_ir_info

    def get_batch(self, segment_info):
        scene_name = segment_info['scene_name']
        tgt_src_idx = segment_info['src_idx']
        tgt_src_loc = segment_info['src_loc']
        tgt_rec_loc = segment_info['rec_loc']
        tgt_ir_path = os.path.join(HAA_ROOT, segment_info['rir_path'])
        depth_path = os.path.join(HAA_ROOT, segment_info['depth_path'])
        tgt_ir, tgt_ir_len = _load_and_cut_audio(tgt_ir_path, 0, self.duration, self.sample_rate)
        tgt_ir_frames = tgt_ir_len // self.downsample_rate

        #get ref_ir_info
        ref_src_loc_list = []
        ref_ir_list = []
        ref_ir_frames_list = []

        ref_ir_info_list = self.get_ref_ir_info(tgt_ir_path, tgt_src_idx)
        for ref_ir_info in ref_ir_info_list:
            ref_src_loc_list.append(ref_ir_info[0])
            ref_ir_list.append(ref_ir_info[1])
            ref_ir_frames_list.append(ref_ir_info[2])

        #get panorama depth
        pano_depth_map = np.load(depth_path)
        #convert to camera coordinate
        pano_depth_map = torch.from_numpy(pano_depth_map).float()# (256, 512, 3)
        
        cc_depth_map = convert_equirect_to_camera_coord(pano_depth_map, 256, 512).numpy()
        #(256, 512, 3) -> (3, 256, 512)
        cc_depth_map = rearrange(cc_depth_map, 'h w c -> c h w')
        
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


class HAA_collator:
    """
    用于将Roll_Music_Dataset返回的样本批处理成张量。
    它处理长度不一的序列，通过填充使其具有相同的长度。
    """
    def __init__(self, cfg, split = 'train'):
        self.cfg = cfg
        self.split = split
        self.reference_count = cfg.preprocess.max_reference_count

    def __call__(self, batch):
        # 过滤掉可能因错误产生的None值（虽然__getitem__的逻辑避免了这种情况）
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        packed_batch = dict()
        keys = batch[0].keys()
        if self.split == 'test':
            dynamic_reference_count = self.reference_count
        elif self.split == 'train':
            dynamic_reference_count = random.randint(1, self.reference_count)
        
        for key in keys:
            if key == "all_ir" or key == "all_cc_src_loc": #(N, 1, t)
                sequences = [torch.from_numpy(item[key]).float()[-(dynamic_reference_count + 1):] for item in batch]
                packed_batch[key] = torch.stack(sequences, dim=0)#(b, N, ...)
            elif  key == "cc_depth_map":
                sequences = [torch.from_numpy(item[key]).float() for item in batch]
                packed_batch[key] = torch.stack(sequences, dim=0)
            # try:
            #     if key == "all_ir" or key == "all_cc_src_loc": #(N, 1, t)
            #         sequences = [torch.from_numpy(item[key]).float() for item in batch]
            #         packed_batch[key] = torch.stack(sequences, dim=0)#(b, N, ...)
            #         packed_batch[key] = packed_batch[key][:, -(dynamic_reference_count + 1):]
            #     elif  key == "cc_depth_map":
            #         sequences = [torch.from_numpy(item[key]).float() for item in batch]
            #         packed_batch[key] = torch.stack(sequences, dim=0)
            #     elif key == "all_code_mask":
            #         sequences = [torch.from_numpy(item[key]).bool() for item in batch]
            #         packed_batch[key] = torch.stack(sequences, dim=0)
            #         packed_batch[key] = packed_batch[key][:, -(dynamic_reference_count + 1):]
            # except Exception as e:
            #     print(f"Error packing batch for key {key}: {e}. Returning a random sample instead.")
            #     print([s.shape for s in sequences])
            #     exit(0)
            
            
        return packed_batch
