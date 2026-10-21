"""Real CPU/Gloo collectives: no CUDA allocation or model execution."""
import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp

from models.EigeNet.src.evaluate import merge_rank_results
from models.loss.evaluator import Evaluator, MetricAccumulator


def worker(rank, world_size, root):
    dist.init_process_group('gloo', init_method=Path(root, 'rendezvous').as_uri(),
                            rank=rank, world_size=world_size, timeout=timedelta(seconds=45))
    try:
        acc = MetricAccumulator(signal_length=7992)
        # One rank has no samples; another has a smaller final batch.
        factors = np.array({0: [1., 2.], 1: [], 2: [-1.]}[rank])
        gt = np.zeros((len(factors), 7992), dtype=np.float64)
        gt[:, 7900] = 1
        detailed = Evaluator(signal_length=7992).evaluate_detailed(gt, gt * factors[:, None])
        acc.update(**detailed)
        merged = merge_rank_results(acc).summary()
        try:
            merge_rank_results(acc, 'synthetic invalid waveform' if rank == 2 else None)
        except RuntimeError as exc:
            error = str(exc)
        else:
            error = ''
        Path(root, f'rank{rank}.json').write_text(json.dumps({'summary': merged, 'error': error}))
    finally:
        dist.destroy_process_group()


class DistributedEvaluationTests(unittest.TestCase):
    def test_nonoverlapping_shards_and_errors_are_combined_on_all_ranks(self):
        with tempfile.TemporaryDirectory() as root:
            mp.spawn(worker, args=(3, root), nprocs=3, join=True)
            for rank in range(3):
                result = json.loads(Path(root, f'rank{rank}.json').read_text())
                summary = result['summary']
                self.assertEqual(summary['samples_evaluated'], 3)
                self.assertAlmostEqual(summary['metrics']['waveform_nmse']['value'], 5 / 3)
                self.assertEqual(summary['protocol']['signal_length'], 7992)
                self.assertEqual(summary['metrics']['phase_error_deg']['valid_count'], 3)
                self.assertAlmostEqual(summary['metrics']['phase_error_deg']['value'], 60.)
                self.assertEqual(summary['diagnostics']['early_short_window_sample_count'], 3)
                self.assertEqual(summary['diagnostics']['early_window_samples']['min'], 92)
                self.assertEqual(summary['invalid_reasons']['edt_error'], {'ground_truth_undefined_ValueError': 3})
                self.assertIn('rank 2: synthetic invalid waveform', result['error'])


if __name__ == '__main__':
    unittest.main()
