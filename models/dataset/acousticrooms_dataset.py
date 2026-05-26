import sys
import torch
import os
src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # AnyTrainer
#sys.path.append(src)
sys.path.insert(0, src)  # 使用 insert(0, ...) 确保优先级，并检查避免重复添加
#print(f"sys.path: {sys.path[0]}")
import librosa
import numpy as np
import random
import json
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils.util import load_config
import logging
from einops import rearrange
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from models.dataset.utils import get_3d_point_camera_coord, convert_equirect_to_camera_coord

# --- 辅助函数 ---

def frame2mask(frames, max_frame):
    frames = frames.reshape(-1)
    seq = np.arange(max_frame)
    return seq[None, :] < frames[:, None]

def _load_and_cut_audio(path, start_sec, duration_sec, sample_rate, mono = True):
    """
    高效加载并精确切割音频文件。
    """
    # librosa.load 能够高效地只加载需要的部分，而不会读取整个文件
    audio_clip, _ = librosa.load(path, sr=sample_rate, offset=start_sec, duration=duration_sec, mono = mono)
    audio_len = audio_clip.shape[-1]
    if audio_len < int(duration_sec * sample_rate):
        audio_clip = np.pad(audio_clip, (0, int(duration_sec * sample_rate) - audio_len), mode='constant')
    audio_clip = audio_clip[None, :]
    
    return audio_clip, audio_len#(1, t)

# --- 主数据集类 ---

class AcousticRooms_Dataset(Dataset):
    """
    用于声学的RIR数据集。
    从一个预计算的文件加载数据集元数据
    """
    def __init__(self, cfg=None, split = 'train'):
        self.cfg = cfg
        self.frame_rate = cfg.preprocess.frame_rate
        self.sample_rate = cfg.preprocess.sample_rate
        self.downsample_rate = self.sample_rate // self.frame_rate
        self.depth_map_root = cfg.dataset.depth_map_root
        self.rir_root = cfg.dataset.rir_root
        self.metadata_root = cfg.dataset.metadata_root
        self.duration = cfg.preprocess.duration
        self.reference_count = cfg.preprocess.max_reference_count
        # 从配置文件指定的路径加载片段列表
        if split == 'test':
            self.segment_list = self._load_file_list(self.cfg.dataset.test_rir_list)
            print(f"Valid_dataset loaded {len(self.segment_list)} test segments")
        else:
            self.segment_list = self._load_file_list(self.cfg.dataset.rir_list)
            print(f"Train_dataset loaded {len(self.segment_list)} train segments")

        if not self.segment_list:
            raise ValueError(f"No data loaded from {self.cfg.dataset.rir_list}. The file might be empty or in the wrong format.")
            
            
        print(f"Total duration: {round(sum([item['duration'] for item in self.segment_list]) / 3600, 2)} hours, Average duration: {np.mean([item['duration'] for item in self.segment_list])} seconds")


        self.wav_path_index2duration = {
            idx: item["duration"] for idx, item in enumerate(self.segment_list)
        }
        #for dynamic batch loading,获取每个时序信号的帧数
        self.index2num_frames = [
            int(item["duration"] * self.frame_rate) + 1
            for item in self.segment_list
        ]
        #for dynamic batch loading,获取按帧数排序的索引
        self.num_frame_indices = np.array(
            sorted(
                range(len(self.index2num_frames)),
                key=lambda k: self.index2num_frames[k],
            )
        )
        import random
        random.seed(cfg.train.random_seed)

    def _load_file_list(self, rir_list):
        """
        辅助函数，从 数据集 文件加载数据。
        """
        if not isinstance(rir_list, list):
            rir_list = [rir_list]
        segment_list = []
        for d in rir_list:
            if not isinstance(d, list):
                d = [d, 1]
            jsonl_path, num = d
            
            segment_list.extend([json.loads(line) for line in open(jsonl_path, 'r', encoding='utf-8')] * num)
        return segment_list

    def __len__(self):
        return len(self.segment_list)
    
    #for dynamic batch loading,获取其帧数
    def get_num_frames(self, index):
        # return self.wav_path_index2duration[index] * 50
        return self.wav_path_index2duration[index] * self.frame_rate
    
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
        # print(f"检查匹配ref ir info")
        # print(f"scene_name: {scene_name}")
        # print(f"scene_id: {scene_id}")
        # print(f"tgt_src_idx: {tgt_src_idx}")
        # print(f"tgt_rec_idx: {tgt_rec_idx}")
        # for ref_ir_path in ref_ir_paths:
        #     print(f"ref_ir_path: {ref_ir_path}")
        # print("--------------------------------")
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
            S = all_ir_frames.shape[0]

            return {
                "all_ir": all_ir, #(N, t)
                "all_cc_src_loc": all_cc_src_loc, #(N, 3)
                "cc_depth_map": cc_depth_map, #(256, 512, 3)
                "valid_src_num": valid_src_num, #(1,)
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

# --- Collator 类 ---

class AcousticRooms_Collator:
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
        batch_valid_src_num = [item["valid_src_num"] for item in batch]
        minimum_src_num = min(batch_valid_src_num)
        packed_batch = dict()
        keys = batch[0].keys()
        if self.split == 'test':
            dynamic_reference_count = min(minimum_src_num, self.reference_count)
        else:
            dynamic_reference_count = random.randint(1, min(minimum_src_num, self.reference_count))
        
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

if __name__ == "__main__":
    config_path = '/data/250010171/code/EigeNet/egs/rir/flow_matching_transformer/debug_EigeNet_v3.json'
    
    cfg = load_config(config_path)
    dataset = AcousticRooms_Dataset(cfg)
    #collator = PianoCollator(cfg)
    from tqdm import tqdm
    for data in tqdm(dataset):
        for key in data.keys():
            print(key)
            print(data[key].shape)
        exit(0)