import os
from pathlib import Path
import glob
import soundfile as sf
import librosa
import numpy as np
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import random
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

seed_num = 42
random.seed(seed_num)
np.random.seed(seed_num)

DATASET_ROOT = 'path/to/haa'
output_jsonl = 'path/to/haa.jsonl'
test_scenes = ['classroomBase', 'complexBase', 'dampenedBase', 'hallwayBase']

# Original HAA RIRs are stored at 48 kHz in RIRs.npy; export the first 1 s as wav.
HAA_RIR_SAMPLE_RATE = 48000
HAA_RIR_DURATION_SEC = 1.0
EXPORT_RIR = True
OVERWRITE_RIR = False


def export_single_channel_ir_for_scene(
    scene_name,
    dataset_root=DATASET_ROOT,
    sample_rate=HAA_RIR_SAMPLE_RATE,
    duration_sec=HAA_RIR_DURATION_SEC,
    overwrite=OVERWRITE_RIR,
):
    """Export monaural RIRs from RIRs.npy to single_channel_ir/{scene_name}_{src_idx}.wav."""
    scene_dir = os.path.join(dataset_root, scene_name)
    rirs_path = os.path.join(scene_dir, "RIRs.npy")
    output_dir = os.path.join(scene_dir, "single_channel_ir")

    if not os.path.exists(rirs_path):
        raise FileNotFoundError(f"RIRs.npy not found: {rirs_path}")

    os.makedirs(output_dir, exist_ok=True)
    rirs = np.load(rirs_path)
    num_samples = min(int(duration_sec * sample_rate), rirs.shape[1])

    for src_idx in tqdm(range(len(rirs)), desc=f"Exporting {scene_name}"):
        wav_path = os.path.join(output_dir, f"{scene_name}_{src_idx}.wav")
        if os.path.exists(wav_path) and not overwrite:
            continue
        rir = rirs[src_idx, :num_samples].astype(np.float32)
        sf.write(wav_path, rir, sample_rate)


def export_single_channel_ir_for_all_scenes(
    scenes=None,
    dataset_root=DATASET_ROOT,
    sample_rate=HAA_RIR_SAMPLE_RATE,
    duration_sec=HAA_RIR_DURATION_SEC,
    overwrite=OVERWRITE_RIR,
):
    scenes = scenes or test_scenes
    for scene_name in scenes:
        export_single_channel_ir_for_scene(
            scene_name,
            dataset_root=dataset_root,
            sample_rate=sample_rate,
            duration_sec=duration_sec,
            overwrite=overwrite,
        )


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
        duration = librosa.get_duration(path=rir_path)
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
        print('*' * 20)
        print("Start loading segments to jsonl...")
        with tqdm(total=len(all_rir_list)) as pbar:
            for rir in all_rir_iter:
                if rir:
                    for rir_info in rir:
                        f_out.write(json.dumps(rir_info) + '\n')
                pbar.update(1)


if __name__ == "__main__":
    if EXPORT_RIR:
        export_single_channel_ir_for_all_scenes()
    main()
