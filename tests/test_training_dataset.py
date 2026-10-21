import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from models.dataset.acousticrooms_dataset import (
    AcousticRooms_Collator,
    AcousticRooms_Dataset,
    _load_and_cut_audio,
    frame2mask,
)


class TrainingDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.room = self.root / 'scene' / 'room'
        self.room.mkdir(parents=True)
        self.receiver = [1000000.123456789, -2.23456789, 3.456789]
        self.sources = [[1000000.5 + i / 7, i / 3, -i / 11] for i in range(5)]
        for i in range(5):
            sf.write(self.room / f'S00{i}_R001_hybrid_IR.wav',
                     np.full(8, (i + 1) / 16, dtype=np.float32), 16000, subtype='FLOAT')
            (self.room / f'S00{i}_R001.json').write_text(json.dumps(
                dict(src_loc=self.sources[i], rec_loc=self.receiver)))
        self.depth_path = self.root / 'depth.npy'
        np.save(self.depth_path, np.ones((256, 512), dtype=np.float64))
        self.record = dict(scene_name='scene', scene_id='room', src_idx=0, rec_idx=1,
                           src_loc=self.sources[0], rec_loc=self.receiver,
                           rir_path=str(self.room / 'S000_R001_hybrid_IR.wav'),
                           depth_path=str(self.depth_path), other_src_idx_list=[4, 2, 1, 3],
                           duration=0.001)
        self.manifest = self.root / 'train.jsonl'
        self.manifest.write_text(json.dumps(self.record) + '\n')
        self.cfg = SimpleNamespace(
            preprocess=SimpleNamespace(sample_rate=16000, frame_rate=1000,
                                       duration=0.001, max_reference_count=3),
            dataset=SimpleNamespace(rir_list=str(self.manifest), test_rir_list=str(self.manifest),
                                    metadata_root=str(self.root), depth_map_root='', rir_root=''),
            model=SimpleNamespace(eigenet_transformer=SimpleNamespace(depth_resol=[256, 512])),
            train=SimpleNamespace(random_seed=42))

    def test_constructor_does_not_reset_global_rng(self):
        random.seed(817)
        state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        AcousticRooms_Dataset(self.cfg)
        self.assertEqual(random.getstate(), state)
        after_numpy = np.random.get_state()
        self.assertEqual(after_numpy[0], numpy_state[0])
        np.testing.assert_array_equal(after_numpy[1], numpy_state[1])
        self.assertEqual(after_numpy[2:], numpy_state[2:])
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))

    def test_outputs_are_float32_tensors_with_target_last(self):
        dataset = AcousticRooms_Dataset(self.cfg)
        random.seed(31)
        batch = dataset.get_batch(self.record)
        for key in ('all_ir', 'all_cc_src_loc', 'cc_depth_map'):
            self.assertIsInstance(batch[key], torch.Tensor)
            self.assertEqual(batch[key].dtype, torch.float32)
        self.assertEqual(batch['all_ir'].shape, (4, 1, 16))
        self.assertEqual(batch['valid_src_num'], 4)
        # Seed 31 chooses source indices [4, 2, 3]; the target is source 0.
        torch.testing.assert_close(batch['all_ir'][:, 0, 0],
                                   torch.tensor([5 / 16, 3 / 16, 4 / 16, 1 / 16]), rtol=0, atol=0)
        torch.testing.assert_close(batch['all_ir'][-1, 0, :8], torch.full((8,), 1 / 16), rtol=0, atol=0)
        torch.testing.assert_close(batch['all_ir'][-1, 0, 8:], torch.zeros(8), rtol=0, atol=0)
        # Double precision subtraction must happen before the float32 conversion.
        expected = torch.tensor([0.37654321105219424, 2.23456789, -3.456789])
        torch.testing.assert_close(batch['all_cc_src_loc'][-1], expected, rtol=0, atol=0)

    def test_depth_resolution_comes_from_model_and_mismatch_fails(self):
        self.cfg.model.eigenet_transformer.depth_resol = [2, 4]
        np.save(self.depth_path, np.ones((2, 4)))
        dataset = AcousticRooms_Dataset(self.cfg)
        batch = dataset.get_batch(self.record)
        self.assertEqual(batch['cc_depth_map'].shape, (3, 2, 4))
        np.save(self.depth_path, np.ones((4, 2)))
        with self.assertRaisesRegex(ValueError, 'depth.*shape|depth.*resolution'):
            dataset.get_batch(self.record)

    def test_no_references_raise(self):
        dataset = AcousticRooms_Dataset(self.cfg)
        for refs in ([], [99, 100]):
            with self.subTest(refs=refs):
                with self.assertRaisesRegex(ValueError, '[Nn]o.*reference'):
                    dataset.get_batch(dict(self.record, other_src_idx_list=refs))

    def test_load_error_is_contextual_and_does_not_replace_sample(self):
        dataset = AcousticRooms_Dataset(self.cfg)
        # The second return keeps the legacy retry loop finite during the red run.
        with patch.object(dataset, 'get_batch', side_effect=[OSError('broken wave'), {}]):
            with self.assertRaisesRegex(RuntimeError, '0.*S000.*broken wave') as error:
                dataset[0]
        self.assertIsInstance(error.exception.__cause__, OSError)

    def test_shortage_and_missing_candidates_preserve_reference_order(self):
        dataset = AcousticRooms_Dataset(self.cfg)
        record = dict(self.record, other_src_idx_list=[3, 99, 1])
        random.seed(52)
        state = random.getstate()
        batch = dataset.get_batch(record)
        self.assertEqual(random.getstate(), state)
        self.assertEqual(batch['valid_src_num'], 2)
        torch.testing.assert_close(batch['all_ir'][:, 0, 0],
                                   torch.tensor([4 / 16, 2 / 16, 1 / 16]), rtol=0, atol=0)

    def test_collator_uses_one_random_count_and_tail_for_entire_batch(self):
        dataset = AcousticRooms_Dataset(self.cfg)
        random.seed(31)
        first = dataset.get_batch(self.record)
        second = dataset.get_batch(dict(self.record, other_src_idx_list=[3, 2, 1]))
        # The next draw after sample([4, 2, 1, 3], 3) is K=2.
        batch = AcousticRooms_Collator(self.cfg)([first, second])
        self.assertEqual(batch['all_ir'].shape, (2, 3, 1, 16))
        torch.testing.assert_close(batch['all_ir'][:, :, 0, 0],
                                   torch.tensor([[3 / 16, 4 / 16, 1 / 16],
                                                 [3 / 16, 2 / 16, 1 / 16]]), rtol=0, atol=0)
        for index, sample in enumerate([first, second]):
            torch.testing.assert_close(batch['all_cc_src_loc'][index], sample['all_cc_src_loc'][-3:], rtol=0, atol=0)
            torch.testing.assert_close(batch['cc_depth_map'][index], sample['cc_depth_map'], rtol=0, atol=0)

    def test_test_split_uses_test_manifest_and_maximum_shared_reference_count(self):
        test_manifest = self.root / 'test.jsonl'
        test_manifest.write_text(json.dumps(dict(self.record, duration=0.012)) + '\n')
        self.cfg.dataset.test_rir_list = str(test_manifest)
        dataset = AcousticRooms_Dataset(self.cfg, split='test')
        self.assertEqual(dataset.get_num_frames(0), 12)
        first = dataset.get_batch(dict(self.record, other_src_idx_list=[4, 2, 1]))
        second = dataset.get_batch(dict(self.record, other_src_idx_list=[3]))
        state = random.getstate()
        batch = AcousticRooms_Collator(self.cfg, split='test')([first, second])
        self.assertEqual(random.getstate(), state)
        self.assertEqual(batch['all_ir'].shape, (2, 2, 1, 16))
        torch.testing.assert_close(batch['all_ir'][:, :, 0, 0],
                                   torch.tensor([[2 / 16, 1 / 16], [4 / 16, 1 / 16]]), rtol=0, atol=0)

    def test_manifest_repeats_and_dynamic_frame_order(self):
        records = [dict(self.record, duration=value) for value in [0.0035, 0.0011, 0.0012]]
        self.manifest.write_text(''.join(json.dumps(record) + '\n' for record in records))
        self.cfg.dataset.rir_list = [[str(self.manifest), 2], str(self.manifest)]
        dataset = AcousticRooms_Dataset(self.cfg)
        self.assertEqual(len(dataset), 9)
        self.assertEqual([dataset.get_num_frames(i) for i in range(9)], [3.5, 1.1, 1.2] * 3)
        np.testing.assert_array_equal(dataset.num_frame_indices, [1, 2, 4, 5, 7, 8, 0, 3, 6])
        np.testing.assert_array_equal(frame2mask(np.array([[0], [2], [4]]), 3),
                                      [[False, False, False], [True, True, False], [True, True, True]])

    def test_invalid_depth_configuration_is_rejected(self):
        for resolution in ([0, 8], [4], [2.5, 8], [True, 8]):
            with self.subTest(resolution=resolution):
                self.cfg.model.eigenet_transformer.depth_resol = resolution
                with self.assertRaises(ValueError):
                    AcousticRooms_Dataset(self.cfg)

    def test_real_io_errors_include_index_path_and_original_cause(self):
        dataset = AcousticRooms_Dataset(self.cfg)
        broken_depth = self.root / 'broken.npy'
        broken_depth.write_bytes(b'not a numpy array')
        failures = [dict(self.record, rir_path=str(self.room / 'S0099_R001_hybrid_IR.wav')),
                    dict(self.record, depth_path=str(self.root / 'absent.npy')),
                    dict(self.record, depth_path=str(broken_depth)),
                    dict(self.record, other_src_idx_list=[])]
        for record in failures:
            with self.subTest(record=record):
                dataset.segment_list[0] = record
                with self.assertRaises(RuntimeError) as error:
                    dataset[0]
                self.assertIn('sample 0', str(error.exception))
                self.assertIn(record['rir_path'], str(error.exception))
                self.assertIsNotNone(error.exception.__cause__)

    def test_audio_helper_retains_librosa_cut_padding_and_length(self):
        path = str(self.room / 'S001_R001_hybrid_IR.wav')
        audio, length = _load_and_cut_audio(path, 0.00025, 0.001, 16000)
        self.assertEqual(length, 4)
        np.testing.assert_array_equal(audio, [[0.125] * 4 + [0.] * 12])

    def test_data_loader_zero_and_two_workers_keep_shapes_and_targets(self):
        self.cfg.dataset.rir_list = [[str(self.manifest), 4]]
        dataset = AcousticRooms_Dataset(self.cfg)
        for workers in (0, 2):
            with self.subTest(workers=workers):
                loader = DataLoader(dataset, batch_size=2, num_workers=workers,
                                    collate_fn=AcousticRooms_Collator(self.cfg),
                                    generator=torch.Generator().manual_seed(23))
                batches = list(loader)
                self.assertEqual(len(batches), 2)
                for batch in batches:
                    self.assertEqual(batch['all_ir'].shape[0], 2)
                    self.assertIn(batch['all_ir'].shape[1], [2, 3, 4])
                    self.assertEqual(batch['cc_depth_map'].shape, (2, 3, 256, 512))
                    torch.testing.assert_close(batch['all_ir'][:, -1, 0, :8],
                                               torch.full((2, 8), 1 / 16), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
