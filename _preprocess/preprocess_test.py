import pandas as pd
import os
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
dataset_root = '/mnt/data1/jingchong/AcousticRooms'
DEPTH_ROOT = os.path.join(dataset_root, 'depth_map')
METADATA_ROOT = os.path.join(dataset_root, 'metadata')
RIR_ROOT = os.path.join(dataset_root, 'single_channel_ir')
output_jsonl = '/mnt/workspace/jingchong/mycode/RIR/EigeNet/data/AcousticRooms_test.jsonl'

test_scene_ids = [
            "Apartments_idx_50",
            "Apartments_idx_42",
            "Bathrooms_idx_18",
            "Bathrooms_idx_14",
            "Cafe_idx_1",
            "LivingRoomsWithHallway_idx_25",
            "LivingRoomsWithHallway_idx_30",
            "Office_idx_11",
            "Office_idx_10",
            "Auditorium_idx_1",
            "Bedrooms_idx_18",
            "Bedrooms_idx_33",
            "Listeningscene_idx_2",
            "Meetingscene_idx_20",
            "Meetingscene_idx_32",
            "Restaurants_idx_24",
            "Restaurants_idx_22",
            ]

as_df = pd.DataFrame(columns=['scene_id', 'src_id', 'rec_id', 'rir_path', 'depth_path', 'metadata_path', 'split', 'duration'])

def get_receiver_source_location(rir_file_path):
    scene_name = rir_file_path.split("/")[-3]
    scene_id = rir_file_path.split("/")[-2]
    ir_file_name = rir_file_path.split("/")[-1]
    src_node, rec_node = int(ir_file_name.split("_")[0][1:]), int(ir_file_name.split("_")[1][1:])
    json_file_name = "S00" + str(src_node) + "_R00" + str(rec_node) + ".json"
    metadata_file_path = os.path.join(METADATA_ROOT, scene_name, scene_id, json_file_name)
    with open(metadata_file_path, "r") as fin:
        meta_info = json.load(fin)
    src_loc = meta_info["src_loc"]
    rec_loc = meta_info["rec_loc"]
    return src_loc, rec_loc

def get_other_src_idx_list(rir_path):
    scene_name = rir_path.split("/")[-3]
    scene_id = rir_path.split("/")[-2]
    ir_file_name = rir_path.split("/")[-1]
    src_idx, rec_idx = int(ir_file_name.split("_")[0][1:]), int(ir_file_name.split("_")[1][1:])
    scene_folder = os.path.join(RIR_ROOT, scene_name, scene_id)
    all_src_idx_list = set([int(fn.split("_")[0][1:]) for fn in os.listdir(scene_folder)])
    remain_src_idx_list = list(all_src_idx_list.difference(set([src_idx])))
    valid_other_src_idx_list = []
    for node in remain_src_idx_list:
            rec_n = ir_file_name.split("_")[1]
            src_n = f"S00{node}"
            other_src_ir_path = os.path.join(scene_folder, f"{src_n}_{rec_n}_hybrid_IR.wav")
            if os.path.exists(other_src_ir_path):
                valid_other_src_idx_list.append(node)
    return valid_other_src_idx_list

def preprocess_info(seg_info):
    rir_path, metadata_path, depth_path, scene_name, scene_id, src_idx, rec_idx, duration = seg_info
    if not os.path.exists(metadata_path):
        return []
    if not os.path.exists(depth_path):
        return []
    if scene_id not in test_scene_ids:
        return []

    src_loc, rec_loc = get_receiver_source_location(rir_path)
    #other src shared the same rec
    other_src_idx_list = get_other_src_idx_list(rir_path)
    return [{
        "scene_name": scene_name,
        "scene_id": scene_id,
        "src_idx": src_idx,
        "rec_idx": rec_idx,
        "src_loc": src_loc,
        "rec_loc": rec_loc,
        "rir_path": rir_path,
        "depth_path": depth_path,
        "duration": duration,
        "other_src_idx_list": other_src_idx_list,
    }]

def seg_info_generator(all_rir_list):
    for rir_path in all_rir_list:
        scene_id = os.path.basename(os.path.dirname(rir_path))
        scene_name = os.path.basename(os.path.dirname(os.path.dirname(rir_path)))
        rir_file_name = os.path.splitext(os.path.basename(rir_path))[0]
        rec_idx, src_idx = int(rir_file_name.split("_")[1][1:]), int(rir_file_name.split("_")[0][1:])
        metadata_path = os.path.join(METADATA_ROOT, scene_name, scene_id, f"S00{src_idx}_R00{rec_idx}.json")
        depth_path = os.path.join(DEPTH_ROOT, scene_name, scene_id, f"{rec_idx}.npy")
        duration = librosa.get_duration(filename=rir_path)
        yield (rir_path, metadata_path, depth_path, scene_name, scene_id, src_idx, rec_idx, duration)
    

def main():
    all_rir_list = []
    for test_room_id in test_scene_ids:
        test_scene_name = test_room_id.split('_idx_')[0]
        test_dir = os.path.join(RIR_ROOT, test_scene_name, test_room_id)
        all_rir_list.extend(glob.glob(os.path.join(test_dir, '*.wav')))
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
           