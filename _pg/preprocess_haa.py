import pandas as pd
import os
from pathlib import Path
import glob
import soundfile as sf
from sympy import false
import librosa
import numpy as np
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import random
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
seed_num = 42
random.seed(seed_num)
np.random.seed(seed_num)
DATASET_ROOT = '/data/share/amphion/data/noise-and-rirs/haa'
DEPTH_ROOT = os.path.join(DATASET_ROOT, 'depth_map')
METADATA_ROOT = os.path.join(DATASET_ROOT, 'metadata')
RIR_ROOT = os.path.join(DATASET_ROOT, 'single_channel_ir')
output_jsonl = '/data/250010171/code/EigeNet_discriminant/data/haa.jsonl'
test_scenes = ['classroomBase', 'complexBase', 'dampenedBase', 'hallwayBase']

def preprocess_info(seg_info):
    rir_path, metadata_path, depth_path, scene_name, src_idx, duration = seg_info
    if scene_name == "classroomBase":
        rec_locs = np.array([3.5838, 5.7230, 1.2294])
    elif scene_name == "hallwayBase":
        rec_locs = np.array([0.6870, 10.2452, 0.5367])
    elif scene_name == "dampenedBase":
        rec_locs = np.array([2.4542, 2.4981, 1.2654])
    elif scene_name == "complexBase":
        rec_locs = np.array([2.8377, 10.1228, 1.1539])
    if not os.path.exists(metadata_path):
        print(f"cannot find metadata path {metadata_path}")
        return []
    if not os.path.exists(depth_path):
        print(f"cannot find depth path {depth_path}")
        return []
    src_locs = np.load(metadata_path)[src_idx].copy()
    
    return [{
        "scene_name": scene_name,
        "src_idx": src_idx,
        "src_loc": src_locs.tolist(),
        "rec_loc": rec_locs.tolist(),
        "rir_path": rir_path,
        "depth_path": depth_path,
        "duration": duration,
    }]

def seg_info_generator(all_rir_list):
    for rir_path in all_rir_list:
        scene_name = os.path.basename(os.path.dirname(os.path.dirname(rir_path)))
        src_idx = int(Path(rir_path).stem.split('_')[-1])
        duration = librosa.get_duration(filename=rir_path)
        metadata_path = os.path.join(DATASET_ROOT, scene_name, "xyzs.npy")
        depth_path = os.path.join(DATASET_ROOT, scene_name, "depth.npy")
        yield (rir_path, metadata_path, depth_path, scene_name, src_idx, duration)
    

def main():
    all_rir_list = []
    for test_scene in test_scenes:
        test_scene_dir = os.path.join(DATASET_ROOT, test_scene)
        rir_dir = os.path.join(test_scene_dir, 'single_channel_ir')
        all_rir_list.extend(glob.glob(os.path.join(rir_dir, '*.wav')))
    print(f"testset find {len(all_rir_list)} rir files")
    with Pool(processes=cpu_count()) as pool, open(output_jsonl, 'w', encoding='utf-8') as f_out:
        all_rir_iter = list(tqdm(pool.imap_unordered(preprocess_info, seg_info_generator(all_rir_list)), total=len(all_rir_list)))
        print(all_rir_iter[0])
        print(len(all_rir_iter))
        print('*'*20)
        print(f"Start loading segments to jsonl...")
        with tqdm(total=len(all_rir_list)) as pbar:
            for rir in all_rir_iter:
                if rir:
                    for rir_info in rir:
                        f_out.write(json.dumps(rir_info) + '\n')
                pbar.update(1)

if __name__ == "__main__":
    main()
           