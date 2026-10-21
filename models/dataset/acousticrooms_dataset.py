import json
import os
import random

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from models.dataset.utils import convert_equirect_to_camera_coord


def frame2mask(frames, max_frame):
    frames = frames.reshape(-1)
    seq = np.arange(max_frame)
    return seq[None, :] < frames[:, None]


def _load_and_cut_audio(path, start_sec, duration_sec, sample_rate, mono=True):
    audio_clip, _ = librosa.load(
        path, sr=sample_rate, offset=start_sec, duration=duration_sec, mono=mono
    )
    audio_len = audio_clip.shape[-1]
    if audio_len < int(duration_sec * sample_rate):
        audio_clip = np.pad(
            audio_clip, (0, int(duration_sec * sample_rate) - audio_len), mode='constant'
        )
    return audio_clip[None, :], audio_len


class AcousticRooms_Dataset(Dataset):
    def __init__(self, cfg, split='train'):
        self.frame_rate = cfg.preprocess.frame_rate
        self.sample_rate = cfg.preprocess.sample_rate
        self.metadata_root = cfg.dataset.metadata_root
        self.duration = cfg.preprocess.duration
        self.reference_count = cfg.preprocess.max_reference_count
        self.depth_resolution = tuple(cfg.model.eigenet_transformer.depth_resol)
        if (len(self.depth_resolution) != 2
                or any(type(size) is not int or size <= 0 for size in self.depth_resolution)):
            raise ValueError('depth_resol must contain two positive integers')
        if self.reference_count < 1:
            raise ValueError('max_reference_count must be positive')
        manifests = cfg.dataset.test_rir_list if split == 'test' else cfg.dataset.rir_list
        self.segment_list = self._load_file_list(manifests)
        if not self.segment_list:
            raise ValueError(f'No data loaded from {manifests}')
        # Keep stable ordering for equal frame counts, as the dynamic sampler expects.
        self.num_frame_indices = np.array(sorted(
            range(len(self)),
            key=lambda index: int(self.get_num_frames(index)) + 1,
        ))

    def _load_file_list(self, rir_list):
        manifests = rir_list if isinstance(rir_list, list) else [rir_list]
        segments = []
        for entry in manifests:
            path, repeat = entry if isinstance(entry, list) else (entry, 1)
            with open(path, 'r', encoding='utf-8') as handle:
                records = [json.loads(line) for line in handle]
            segments.extend(records * repeat)
        return segments

    def __len__(self):
        return len(self.segment_list)

    def get_num_frames(self, index):
        return self.segment_list[index]['duration'] * self.frame_rate

    def get_receiver_source_location(self, ir_path):
        scene_name, scene_id, filename = ir_path.split('/')[-3:]
        source, receiver = filename.split('_')[:2]
        metadata_name = f'S00{int(source[1:])}_R00{int(receiver[1:])}.json'
        metadata_path = os.path.join(self.metadata_root, scene_name, scene_id, metadata_name)
        with open(metadata_path, 'r', encoding='utf-8') as handle:
            metadata = json.load(handle)
        return metadata['src_loc'], metadata['rec_loc']

    def _reference_paths(self, target_path, source_indices):
        directory = os.path.dirname(target_path)
        receiver = os.path.basename(target_path).split('_')[1]
        paths = [os.path.join(directory, f'S00{source}_{receiver}_hybrid_IR.wav')
                 for source in source_indices]
        paths = [path for path in paths if os.path.exists(path)]
        count = len(paths)
        if not count:
            raise ValueError(f'No valid reference RIR for {target_path}')
        if count > self.reference_count:
            paths = random.sample(paths, self.reference_count)
        return paths, count

    def get_batch(self, segment_info):
        target_path = segment_info['rir_path']
        reference_paths, valid_source_count = self._reference_paths(
            target_path, segment_info['other_src_idx_list']
        )
        depth_path = segment_info['depth_path']
        depth = np.load(depth_path)
        if depth.shape != self.depth_resolution:
            raise ValueError(
                f'depth shape {depth.shape} does not match configured resolution '
                f'{self.depth_resolution}: {depth_path}'
            )
        depth = torch.as_tensor(depth, dtype=torch.float32)
        camera_depth = convert_equirect_to_camera_coord(depth, *self.depth_resolution)

        sources = [self.get_receiver_source_location(path)[0] for path in reference_paths]
        sources.append(segment_info['src_loc'])
        # Match the old float64 camera transform before the model's float32 cast.
        locations = (np.asarray(sources, dtype=np.float64)
                     - np.asarray(segment_info['rec_loc'], dtype=np.float64))
        waveforms = [_load_and_cut_audio(path, 0, self.duration, self.sample_rate)[0]
                     for path in reference_paths + [target_path]]
        return {
            'all_ir': torch.as_tensor(np.stack(waveforms), dtype=torch.float32),
            'all_cc_src_loc': torch.as_tensor(locations, dtype=torch.float32),
            'cc_depth_map': camera_depth.permute(2, 0, 1),
            'valid_src_num': valid_source_count,
        }

    def __getitem__(self, idx):
        segment = self.segment_list[idx]
        try:
            return self.get_batch(segment)
        except Exception as error:
            raise RuntimeError(
                f"Failed to load sample {idx} ({segment.get('rir_path', '<missing rir_path>')}): {error}"
            ) from error


class AcousticRooms_Collator:
    def __init__(self, cfg, split='train'):
        self.split = split
        self.reference_count = cfg.preprocess.max_reference_count

    def __call__(self, batch):
        maximum = min(self.reference_count, min(item['valid_src_num'] for item in batch))
        count = maximum if self.split == 'test' else random.randint(1, maximum)
        return {
            'all_ir': torch.stack([item['all_ir'][-(count + 1):] for item in batch]),
            'all_cc_src_loc': torch.stack([item['all_cc_src_loc'][-(count + 1):] for item in batch]),
            'cc_depth_map': torch.stack([item['cc_depth_map'] for item in batch]),
        }
