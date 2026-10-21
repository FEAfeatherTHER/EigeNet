import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from models.dataset.evaluation_dataset import EvaluationDataset, EvaluationShardSampler, collate_evaluation


class EvaluationDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.room = self.root / 'scene' / 'room'
        self.room.mkdir(parents=True)
        for i in range(5):
            (self.room / f'S00{i}_R001_hybrid_IR.wav').touch()
        self.records = [dict(rir_path=str(self.room / 'S000_R001_hybrid_IR.wav'),
                             depth_path='synthetic.npy', src_loc=[0, 0, 0], rec_loc=[1, 0, 0],
                             other_src_idx_list=[4, 2, 1, 3, 0, 1])]
        self.manifest = self.root / 'test.jsonl'
        self.write_manifest()
        self.cfg = SimpleNamespace(preprocess=SimpleNamespace(sample_rate=16000, duration=0.5),
                                   dataset=SimpleNamespace(test_rir_list=str(self.manifest), metadata_root=str(self.root)))

    def write_manifest(self):
        self.manifest.write_text(''.join(json.dumps(r) + '\n' for r in self.records))

    def load(self, dataset, index=0):
        with patch('models.dataset.evaluation_dataset._load_and_cut_audio', return_value=(np.ones((1, 8000)), 8000)), \
             patch.object(dataset, 'get_receiver_source_location', return_value=([2, 0, 0], [1, 0, 0])), \
             patch('models.dataset.evaluation_dataset.np.load', return_value=np.ones((2, 4))):
            return dataset[index]

    def test_deterministic_local_rng_target_exclusion_and_shapes(self):
        state = random.getstate()
        dataset = EvaluationDataset(self.cfg, max_references=2)
        a = self.load(dataset)
        b = self.load(dataset)
        self.assertEqual(random.getstate(), state)
        self.assertEqual(a['reference_count'], 2)
        self.assertEqual(a['all_ir'].shape, (3, 1, 8000))
        self.assertEqual(a['all_cc_src_loc'].shape, (3, 3))
        self.assertEqual(a['cc_depth_map'].shape, (3, 2, 4))
        np.testing.assert_array_equal(a['all_ir'], b['all_ir'])
        self.assertNotIn(self.records[0]['rir_path'], dataset._reference_paths(self.records[0], 0))

    def test_reference_selection_is_order_independent_and_batch_independent(self):
        dataset = EvaluationDataset(self.cfg, manifests=[[str(self.manifest), 3]], max_references=2)
        record = dataset.segment_list[0]
        expected = dataset._reference_paths(record, 0)
        self.assertEqual(len(expected), 2)
        record['other_src_idx_list'].reverse()
        self.assertEqual(dataset._reference_paths(record, 0), expected)
        for index in [2, 1, 0]:
            paths = dataset._reference_paths(record, index)
            candidates = sorted(str(self.room / f'S00{i}_R001_hybrid_IR.wav') for i in range(1, 5))
            self.assertEqual(paths, random.Random(42 + index).sample(candidates, 2))
        single = collate_evaluation([self.load(dataset, 0)])['groups'][0]
        combined = collate_evaluation([self.load(dataset, 2), self.load(dataset, 0)])['groups'][0]
        np.testing.assert_array_equal(single['all_ir'][0], combined['all_ir'][1])
        self.assertEqual(combined['sample_ids'], [2, 0])

    def test_multi_digit_source_audio_and_metadata_conventions(self):
        target = self.room / 'S000_R010_hybrid_IR.wav'
        reference = self.room / 'S010_R010_hybrid_IR.wav'
        target.touch()
        reference.touch()
        self.records[0]['rir_path'] = str(target)
        self.records[0]['other_src_idx_list'] = [10]
        self.write_manifest()
        dataset = EvaluationDataset(self.cfg)
        self.assertEqual(dataset._reference_paths(dataset.segment_list[0], 0), [str(reference)])
        metadata = self.room / 'S0010_R0010.json'
        metadata.write_text(json.dumps({'src_loc': [10, 0, 0], 'rec_loc': [1, 0, 0]}))
        self.assertEqual(dataset.get_receiver_source_location(str(reference)), ([10, 0, 0], [1, 0, 0]))

    def test_shortage_and_no_reference_skip_before_io(self):
        self.records[0]['other_src_idx_list'] = [0, 1, 1, 99]
        self.write_manifest()
        dataset = EvaluationDataset(self.cfg)
        self.assertEqual(self.load(dataset)['reference_count'], 1)
        self.records[0]['other_src_idx_list'] = [0, 99]
        self.write_manifest()
        with patch('models.dataset.evaluation_dataset._load_and_cut_audio', side_effect=AssertionError('IO')):
            self.assertEqual(EvaluationDataset(self.cfg)[0], {'sample_id': 0, 'skip_reason': 'no_reference'})

    def test_errors_are_contextual_and_never_replace(self):
        dataset = EvaluationDataset(self.cfg)
        with patch('models.dataset.evaluation_dataset._load_and_cut_audio', side_effect=OSError('bad wave')) as loader:
            with self.assertRaisesRegex(RuntimeError, 'sample 0.*S000.*bad wave'):
                dataset[0]
            self.assertEqual(loader.call_count, 1)

    def test_collate_groups_and_skip_counts(self):
        a = self.load(EvaluationDataset(self.cfg, max_references=1))
        b = self.load(EvaluationDataset(self.cfg, max_references=3))
        b['sample_id'] = 7
        batch = collate_evaluation([b, a, a, {'sample_id': 9, 'skip_reason': 'no_reference'}])
        self.assertEqual(batch['skipped_no_reference'], 1)
        self.assertEqual([g['reference_counts'] for g in batch['groups']], [[1, 1], [3]])
        self.assertEqual(batch['groups'][0]['all_ir'].shape, (2, 2, 1, 8000))
        self.assertEqual(batch['groups'][1]['sample_ids'], [7])
        self.assertEqual(collate_evaluation([]), {'groups': [], 'skipped_no_reference': 0})

    def test_shards_complete_without_duplicates_even_empty(self):
        shards = [list(EvaluationShardSampler(range(7), rank=r, world_size=10)) for r in range(10)]
        self.assertEqual(sorted(x for s in shards for x in s), list(range(7)))
        self.assertEqual(shards[9], [])
        for r in range(10):
            self.assertEqual(len(EvaluationShardSampler(range(7), r, 10)), len(shards[r]))

    def test_manifest_order_repeats_and_config_validation(self):
        dataset = EvaluationDataset(self.cfg, manifests=[[str(self.manifest), 2]])
        self.assertEqual(len(dataset), 2)
        self.assertEqual(self.load(dataset, 1)['sample_id'], 1)
        self.cfg.preprocess.sample_rate = 8000
        with self.assertRaises(ValueError):
            EvaluationDataset(self.cfg)


if __name__ == '__main__':
    unittest.main()
