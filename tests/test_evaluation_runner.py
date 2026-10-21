import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from models.EigeNet.src.evaluate import (
    EigeNetPredictor, evaluate_loader, load_weights, main, scalar_log_values, write_summary,
)


class Codec(torch.nn.Module):
    def encode_z(self, audio):
        return audio

    def decode_z(self, z):
        return z


class ReferenceMean(torch.nn.Module):
    def forward(self, references, depth, coordinates):
        if coordinates.shape[1] != references.shape[1] + 1:
            raise AssertionError('target coordinates must be retained')
        return references.mean(dim=1), torch.zeros(1)


def group(scale=1.0, count=1, references=2):
    wave = torch.zeros(count, references + 1, 1, 8000)
    wave[:, :, :, 160] = 1
    wave[:, :-1] *= scale
    return {'all_ir': wave, 'all_cc_src_loc': torch.zeros(count, references + 1, 3),
            'cc_depth_map': torch.zeros(count, 3, 2, 2),
            'reference_counts': [references] * count, 'sample_ids': list(range(count))}


class EvaluationRunnerTests(unittest.TestCase):
    def test_runner_uses_detailed_metrics_at_evaluator_length(self):
        from models.loss.evaluator import Evaluator
        evaluator = Evaluator(signal_length=7992)
        batch = group(scale=0)
        summary = evaluate_loader(
            [{'groups': [batch], 'skipped_no_reference': 0}],
            EigeNetPredictor(ReferenceMean(), Codec(), 'cpu'), evaluator,
        ).summary()
        self.assertEqual(summary['protocol']['signal_length'], 7992)
        self.assertEqual(summary['metrics']['phase_error_deg']['invalid_count'], 1)
        self.assertEqual(summary['invalid_reasons']['phase_error_deg'], {'no_valid_phase_frequency': 1})
        self.assertEqual(summary['diagnostics']['phase_gt_energy_coverage']['mean'], 0)
        self.assertEqual(summary['diagnostics']['phase_missing_sample_count'], 1)
        logs = scalar_log_values(summary)
        self.assertEqual(logs['diagnostics/phase_missing_sample_count'], 1)
        self.assertNotIn('early_energy_db_mae/db', logs)
        self.assertNotIn('phase_error_deg/db', logs)
        empty = evaluate_loader([], lambda batch: None, evaluator).summary()
        self.assertEqual(empty['protocol']['signal_length'], 7992)

    def test_predictor_encodes_references_only_and_accepts_aligner_tuple(self):
        batch = group(scale=2)
        batch['all_ir'][:, -1, :, 160] = 99
        predictor = EigeNetPredictor(ReferenceMean(), Codec(), 'cpu')
        pred = predictor(batch)
        self.assertEqual(tuple(pred.shape), (1, 1, 8000))
        self.assertEqual(pred[0, 0, 160].item(), 2)
        self.assertFalse(pred.requires_grad)

    def test_runner_weights_samples_not_batches_and_counts_skips(self):
        predictor = EigeNetPredictor(ReferenceMean(), Codec(), 'cpu')
        loader = [{'groups': [group(1, 2)], 'skipped_no_reference': 1},
                  {'groups': [group(2, 1)], 'skipped_no_reference': 0}]
        summary = evaluate_loader(loader, predictor).summary()
        self.assertAlmostEqual(summary['metrics']['waveform_nmse']['value'], 1 / 3)
        self.assertEqual(summary['samples_evaluated'], 3)
        self.assertEqual(summary['skipped_no_reference'], 1)

    def test_empty_shard_has_defined_summary(self):
        summary = evaluate_loader([], lambda batch: None).summary()
        self.assertEqual(summary['samples_evaluated'], 0)
        self.assertIsNone(summary['metrics']['waveform_nmse']['value'])

    def test_prediction_error_is_not_silently_discarded(self):
        def broken(batch):
            raise ValueError('bad prediction')
        with self.assertRaisesRegex(RuntimeError, 'bad prediction'):
            evaluate_loader([{'groups': [group()], 'skipped_no_reference': 0}], broken)

    def test_current_checkpoint_weights_load_strictly(self):
        with tempfile.TemporaryDirectory() as root:
            source = torch.nn.Linear(2, 1)
            destination = torch.nn.Linear(2, 1)
            torch.save(source.state_dict(), Path(root) / 'pytorch_model.bin')
            load_weights(destination, root)
            torch.testing.assert_close(source.weight, destination.weight)
            with self.assertRaises(RuntimeError):
                load_weights(torch.nn.Linear(3, 1), root)

    def test_json_summary_is_strict_and_never_overwrites(self):
        summary = evaluate_loader([], lambda batch: None).summary()
        with tempfile.TemporaryDirectory() as root:
            path = write_summary(summary, root)
            self.assertEqual(json.loads(Path(path).read_text()), summary)
            with self.assertRaises(FileExistsError):
                write_summary(summary, root)

    def test_logs_only_finite_scalar_values(self):
        logs = scalar_log_values({'metrics': {
            'a': {'value': 0.0, 'db': None, 'valid_count': 2, 'invalid_count': 0},
            'b': {'value': None, 'valid_count': 0, 'invalid_count': 2}},
            'samples_evaluated': 2, 'skipped_no_reference': 0})
        self.assertEqual(logs['a'], 0)
        self.assertNotIn('b', logs)
        self.assertTrue(all(np.isfinite(v) for v in logs.values()))

    def test_trainer_validation_uses_shared_runner_and_restores_mode(self):
        from models.EigeNet.src.trainer import EigeNetTrainer
        trainer = EigeNetTrainer.__new__(EigeNetTrainer)
        trainer.model = ReferenceMean().train()
        trainer.audio_codec = Codec().eval()
        trainer.accelerator = SimpleNamespace(
            device=torch.device('cpu'), is_main_process=True,
            unwrap_model=lambda model, **kwargs: model)
        trainer.cfg = SimpleNamespace(evaluation=SimpleNamespace(seed=42, max_references=8))
        trainer.step = 1
        trainer.valid_dataloader = [{'groups': [group(2)], 'skipped_no_reference': 0}]
        with tempfile.TemporaryDirectory() as root:
            trainer.exp_dir = root
            metrics = trainer._valid()
            self.assertEqual(metrics['waveform_nmse'], 1)
            self.assertEqual(len(list(Path(root).glob('evaluation/*/summary.json'))), 1)
        self.assertTrue(trainer.model.training)
        self.assertFalse(trainer.audio_codec.training)

    def test_cli_real_synthetic_files_produce_only_aggregate_json(self):
        from scipy.io import wavfile
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            room = root / 'scene' / 'room'
            room.mkdir(parents=True)
            target = np.zeros(8000, dtype=np.float32)
            target[160] = 1
            wavfile.write(room / 'S000_R001_hybrid_IR.wav', 16000, target)
            wavfile.write(room / 'S001_R001_hybrid_IR.wav', 16000, 2 * target)
            (room / 'S001_R001.json').write_text(json.dumps(
                {'src_loc': [2, 0, 0], 'rec_loc': [0, 0, 0]}))
            np.save(root / 'depth.npy', np.ones((2, 4), dtype=np.float32))
            manifest = root / 'test.jsonl'
            manifest.write_text(json.dumps({
                'rir_path': str(room / 'S000_R001_hybrid_IR.wav'),
                'depth_path': str(root / 'depth.npy'), 'src_loc': [1, 0, 0],
                'rec_loc': [0, 0, 0], 'other_src_idx_list': [1]}) + '\n')
            config = root / 'config.json'
            config.write_text(json.dumps({
                'preprocess': {'sample_rate': 16000, 'duration': 0.5},
                'dataset': {'test_rir_list': [str(manifest)], 'metadata_root': str(root)},
                'valid': {'batch_size': 4, 'duration': 0.363}}))
            predictor = EigeNetPredictor(ReferenceMean().eval(), Codec().eval(), 'cpu')
            with patch('models.EigeNet.src.evaluate.build_predictor', return_value=predictor), \
                 redirect_stdout(io.StringIO()):
                summary = main(['--config', str(config), '--checkpoint', 'synthetic-model',
                                '--output-dir', str(root / 'results'), '--device', 'cpu'])
            self.assertEqual(summary['metrics']['waveform_nmse']['value'], 1)
            self.assertEqual(summary['samples_evaluated'], 1)
            self.assertEqual([p.name for p in (root / 'results').iterdir()], ['summary.json'])


if __name__ == '__main__':
    unittest.main()
