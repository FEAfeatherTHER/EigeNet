"""Deterministic, complete-manifest evaluation without training-time resampling."""
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from models.dataset.acousticrooms_dataset import _load_and_cut_audio
from models.dataset.utils import convert_equirect_to_camera_coord, get_3d_point_camera_coord


class EvaluationDataset(Dataset):
    def __init__(self, cfg, manifests=None, max_references=8, seed=42):
        if cfg.preprocess.sample_rate != 16000 or cfg.preprocess.duration != 0.5:
            raise ValueError('Evaluation requires sample_rate=16000 and duration=0.5')
        if not isinstance(max_references, int) or max_references < 1:
            raise ValueError('max_references must be a positive integer')
        self.metadata_root = cfg.dataset.metadata_root
        self.max_references = max_references
        self.seed = seed
        self.segment_list = []
        manifests = cfg.dataset.test_rir_list if manifests is None else manifests
        if isinstance(manifests, (str, os.PathLike)):
            manifests = [manifests]
        # A direct [path, repeats] entry is accepted as well as a list of entries.
        if len(manifests) == 2 and isinstance(manifests[0], (str, os.PathLike)) and isinstance(manifests[1], int):
            manifests = [manifests]
        for entry in manifests:
            path, repeats = entry if isinstance(entry, (tuple, list)) else (entry, 1)
            if not isinstance(repeats, int) or repeats < 1:
                raise ValueError('Manifest repeats must be a positive integer')
            with open(path, encoding='utf-8') as stream:
                rows = [json.loads(line) for line in stream if line.strip()]
            self.segment_list.extend(rows * repeats)

    def __len__(self):
        return len(self.segment_list)

    def _reference_paths(self, record, index):
        target = os.path.abspath(record['rir_path'])
        directory, filename = os.path.split(target)
        receiver = filename.split('_')[1]
        paths = sorted({os.path.join(directory, f'S{int(source):03d}_{receiver}_hybrid_IR.wav')
                        for source in record['other_src_idx_list']})
        paths = [path for path in paths if os.path.abspath(path) != target and os.path.isfile(path)]
        if len(paths) > self.max_references:
            paths = random.Random(self.seed + index).sample(paths, self.max_references)
        return paths

    def get_receiver_source_location(self, ir_path):
        directory, filename = os.path.split(ir_path)
        scene_name, scene_id = directory.split(os.sep)[-2:]
        source, receiver = filename.split('_')[:2]
        metadata_name = f'S00{int(source[1:])}_R00{int(receiver[1:])}.json'
        with open(os.path.join(self.metadata_root, scene_name, scene_id, metadata_name), encoding='utf-8') as stream:
            metadata = json.load(stream)
        return metadata['src_loc'], metadata['rec_loc']

    def __getitem__(self, index):
        record = self.segment_list[index]
        try:
            references = self._reference_paths(record, index)
            if not references:
                return {'sample_id': index, 'skip_reason': 'no_reference'}
            target_ir, _ = _load_and_cut_audio(record['rir_path'], 0, 0.5, 16000)
            locations, audio = [], []
            for path in references:
                source_location, _ = self.get_receiver_source_location(path)
                impulse, _ = _load_and_cut_audio(path, 0, 0.5, 16000)
                locations.append(source_location)
                audio.append(impulse)
            locations.append(record['src_loc'])
            audio.append(target_ir)
            depth = torch.as_tensor(np.load(record['depth_path']), dtype=torch.float32)
            if depth.ndim != 2:
                raise ValueError(f'Expected a two-dimensional depth map, got {tuple(depth.shape)}')
            depth = convert_equirect_to_camera_coord(depth, *depth.shape).permute(2, 0, 1).numpy()
            impulses = np.stack(audio).astype(np.float32)
            if impulses.shape != (len(references) + 1, 1, 8000):
                raise ValueError(f'Unexpected impulse response shape {impulses.shape}')
            return {'sample_id': index, 'reference_count': len(references), 'all_ir': impulses,
                    'all_cc_src_loc': np.stack([get_3d_point_camera_coord(0, record['rec_loc'], location)
                                                for location in locations]).astype(np.float32),
                    'cc_depth_map': depth}
        except Exception as exc:
            raise RuntimeError(f"Evaluation sample {index} ({record.get('rir_path', '<unknown>')}): {exc}") from exc


def collate_evaluation(records):
    buckets = {}
    skipped = 0
    for record in records:
        if record.get('skip_reason') == 'no_reference':
            skipped += 1
        else:
            buckets.setdefault(record['reference_count'], []).append(record)
    groups = []
    for count in sorted(buckets):
        rows = buckets[count]
        group = {key: torch.stack([torch.as_tensor(row[key], dtype=torch.float32) for row in rows])
                 for key in ('all_ir', 'all_cc_src_loc', 'cc_depth_map')}
        group['sample_ids'] = [row['sample_id'] for row in rows]
        group['reference_counts'] = [count] * len(rows)
        groups.append(group)
    return {'groups': groups, 'skipped_no_reference': skipped}


class EvaluationShardSampler(Sampler):
    def __init__(self, dataset, rank=0, world_size=1):
        if not isinstance(world_size, int) or world_size < 1:
            raise ValueError('world_size must be a positive integer')
        if not isinstance(rank, int) or not 0 <= rank < world_size:
            raise ValueError('rank must satisfy 0 <= rank < world_size')
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        return len(range(self.rank, len(self.dataset), self.world_size))
