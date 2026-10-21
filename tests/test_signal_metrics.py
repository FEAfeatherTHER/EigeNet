import copy
import json
import unittest
import numpy as np
from models.loss.evaluator import Evaluator, MetricAccumulator, METRIC_NAMES


class SignalMetricsTest(unittest.TestCase):
    def test_analytical(self):
        gt = np.zeros((1, 8000))
        gt[0, 100] = 1
        for factor, expected in [(1, 0), (2, 1), (-1, 4), (0, 1)]:
            r = Evaluator().evaluate_batch(gt, gt * factor)
            self.assertEqual(set(r), set(METRIC_NAMES))
            self.assertEqual(r['waveform_nmse'][0], expected)
            if factor:
                self.assertAlmostEqual(r['phase_error_deg'][0], 180 if factor < 0 else 0)
        result = Evaluator().evaluate_batch(gt, -gt)
        self.assertEqual(result['mr_log_stft_l1'][0], 0)
        self.assertEqual(result['early_energy_db_mae'][0], 0)

    def test_perfect_all_metrics(self):
        x = np.arange(8000)
        gt = (np.random.RandomState(12).randn(8000) * np.exp(-x / 600))[None]
        for value in Evaluator().evaluate_batch(gt, gt).values():
            np.testing.assert_array_equal(value, [0.])
            self.assertEqual(value.dtype, np.float64)

    def test_phase_wrap_and_single_sided_weighting(self):
        spectrum = np.zeros(4001, dtype=np.complex128)
        spectrum[11] = np.exp(1j * np.deg2rad(179))
        pred = spectrum.copy()
        pred[11] = np.exp(1j * np.deg2rad(-179))
        result = Evaluator().evaluate_new_metrics(np.fft.irfft(spectrum)[None], np.fft.irfft(pred)[None])
        self.assertAlmostEqual(result['metrics']['phase_error_deg'][0], 2, places=10)
        # Equal DC/Nyquist and interior energy must receive equal weights.
        gt = np.tile([2., 1., 0., 1.], 2000)[None]
        pred = np.tile([0., -1., -2., -1.], 2000)[None]
        result = Evaluator().evaluate_new_metrics(gt, pred)
        self.assertAlmostEqual(result['metrics']['phase_error_deg'][0], 144.)

    def test_constant_gain(self):
        gt = np.ones((1, 8000))
        r = Evaluator().evaluate_new_metrics(gt, 2 * gt)
        self.assertAlmostEqual(r['metrics']['phase_error_deg'][0], 0.)
        self.assertAlmostEqual(r['metrics']['early_energy_db_mae'][0], 10 * np.log10(4), places=12)

    def test_zero_and_partial_missing_phase(self):
        zero = np.zeros((1, 8000))
        r = Evaluator().evaluate_detailed(zero, zero)
        for name in ('waveform_nmse', 'phase_error_deg', 'early_energy_db_mae'):
            self.assertTrue(np.isnan(r['metrics'][name][0]))
            self.assertEqual(r['invalid_reasons'][name][0], 'ground_truth_energy_zero')
        self.assertTrue(np.isnan(r['diagnostics']['phase_gt_energy_coverage'][0]))
        self.assertEqual(r['diagnostics']['early_start_sample'][0], -1)
        self.assertEqual(r['diagnostics']['early_window_samples'][0], 0)
        gt = np.ones_like(zero)
        r = Evaluator().evaluate_new_metrics(gt, zero)
        self.assertTrue(np.isnan(r['metrics']['phase_error_deg'][0]))
        self.assertEqual(r['invalid_reasons']['phase_error_deg'][0], 'no_valid_phase_frequency')
        self.assertEqual(r['diagnostics']['phase_gt_energy_coverage'][0], 0)
        gt = np.tile([2., 0.], 4096)[None]
        r = Evaluator(signal_length=8192).evaluate_new_metrics(gt, np.ones_like(gt))
        self.assertEqual(r['metrics']['phase_error_deg'][0], 0)
        self.assertEqual(r['diagnostics']['phase_gt_energy_coverage'][0], .5)
        self.assertEqual(r['diagnostics']['phase_missing_frequency_count'][0], 1)
        self.assertEqual(r['diagnostics']['phase_valid_frequency_count'][0], 1)

    def test_early_shared_onset_window_and_floor(self):
        gt = np.zeros((1, 8000))
        gt[0, 1800] = 1
        gt[0, 5000] = 4
        pred = np.zeros_like(gt)
        pred[0, 1840] = 2
        result = Evaluator().evaluate_new_metrics(gt, pred)
        kernel = np.hanning(64) / np.hanning(64).sum()
        eg = np.convolve(gt[0] ** 2, kernel, 'full')[31:8031]
        ep = np.convolve(pred[0] ** 2, kernel, 'full')[31:8031]
        ref = eg.max()
        lg = 10 * np.log10(np.maximum(eg, ref * 1e-6) / ref)
        lp = 10 * np.log10(np.maximum(ep, ref * 1e-6) / ref)
        expected = np.abs(lg[1800:2600] - lp[1800:2600]).mean()
        self.assertAlmostEqual(result['metrics']['early_energy_db_mae'][0], expected, places=12)
        self.assertGreater(expected, 0)
        self.assertEqual(result['diagnostics']['early_start_sample'][0], 1800)
        self.assertEqual(result['diagnostics']['early_window_samples'][0], 800)
        zero = Evaluator().evaluate_new_metrics(np.ones_like(gt), np.zeros_like(gt))
        self.assertAlmostEqual(zero['metrics']['early_energy_db_mae'][0], 60 + np.mean(10 * np.log10(np.convolve(np.ones(8000), kernel, 'full')[31:8031][7:807])), places=12)

    def test_onset_threshold_and_truncated_window(self):
        gt = np.zeros((1, 8000))
        gt[0, 0] = .01
        gt[0, 7900] = 1
        r = Evaluator().evaluate_new_metrics(gt, gt)
        self.assertEqual(r['diagnostics']['early_start_sample'][0], 7900)
        self.assertEqual(r['diagnostics']['early_window_samples'][0], 100)
        self.assertEqual(r['metrics']['early_energy_db_mae'][0], 0)

    def test_invalid_and_lengths(self):
        z = np.zeros((1, 8000))
        for bad in [z[:, :7999], np.zeros((1, 2, 8000)), z + np.inf, z.astype(complex)]:
            with self.assertRaises(ValueError):
                Evaluator().evaluate_batch(bad, bad)
        with self.assertRaises(ValueError):
            Evaluator(sample_rate=8000)
        for length in (5807, 7992., True):
            with self.assertRaises(ValueError):
                Evaluator(signal_length=length)
        for length in (8000, 7992, 5808):
            ev = Evaluator(signal_length=length)
            x = np.ones((1, length + 20))
            a = ev.evaluate_batch(x[:, None], 2 * x[:, None])
            b = ev.evaluate_batch(x[:, :length], 2 * x[:, :length])
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
            self.assertEqual(a['waveform_nmse'][0], 1)
            self.assertEqual(MetricAccumulator(signal_length=length).summary()['protocol']['signal_length'], length)

    def test_near_zero_padding(self):
        gt = np.zeros((1, 8064))
        gt[0, 0] = 1e-100
        result = Evaluator().evaluate_batch(gt[:, None], 2 * gt[:, None])
        self.assertEqual(result['waveform_nmse'][0], 1)
        self.assertEqual(result['phase_error_deg'][0], 0)
        normal = Evaluator().evaluate_new_metrics(gt / 1e-100, 2 * gt / 1e-100)
        self.assertAlmostEqual(result['early_energy_db_mae'][0], normal['metrics']['early_energy_db_mae'][0])

    def test_stft_reference_and_acoustic_isolation(self):
        rng = np.random.RandomState(9)
        gt, pred = rng.randn(2, 8000), rng.randn(2, 8000)
        ev = Evaluator()
        ev.measure_edt = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError('degenerate'))
        detailed = ev.evaluate_detailed(gt, pred)
        result = detailed['metrics']
        self.assertTrue(np.isnan(result['edt_error']).all())
        self.assertEqual(detailed['invalid_reasons']['edt_error'], ['ground_truth_undefined_ValueError'] * 2)
        losses = []
        for n in (512, 1024, 2048):
            def spec(x):
                padded = np.pad(x, ((0, 0), (n // 2, n // 2)), mode='reflect')
                frames = np.lib.stride_tricks.sliding_window_view(padded, n, axis=1)[:, ::n // 4]
                window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)
                return np.log(np.maximum(np.abs(np.fft.rfft(frames * window, axis=-1)), 1e-4))
            losses.append(np.abs(spec(gt) - spec(pred)).mean((1, 2)))
        np.testing.assert_allclose(result['mr_log_stft_l1'], np.mean(losses, axis=0), rtol=1e-12, atol=1e-12)
        for value in result.values():
            self.assertEqual(value.dtype, np.float64)

    def test_prediction_acoustic_reason_and_t60_percentage(self):
        evaluator = Evaluator()
        values = iter([2.0, 3.0])
        evaluator.measure_rt60 = lambda *args, **kwargs: next(values)
        signal = np.ones((1, 8000))
        self.assertEqual(evaluator.evaluate_batch(signal, signal)['t60_error'][0], 50.)
        values = iter([2., np.inf])
        evaluator.measure_rt60 = lambda *args, **kwargs: next(values)
        self.assertEqual(evaluator.evaluate_detailed(signal, signal)['invalid_reasons']['t60_error'][0], 'prediction_measurement_nonfinite')

    def test_overflow_and_db_states(self):
        with self.assertRaises((ValueError, FloatingPointError)):
            Evaluator().evaluate_batch(np.full((1, 8000), 1e308), np.zeros((1, 8000)))
        acc = MetricAccumulator()
        acc.update({'waveform_nmse': np.array([0., np.nan]), 'phase_error_deg': np.array([180., 0.]), 'early_energy_db_mae': np.array([6., 0.])})
        summary = acc.summary()
        self.assertEqual(summary['metrics']['waveform_nmse']['db_status'], 'negative_infinity')
        for key in ('phase_error_deg', 'early_energy_db_mae'):
            self.assertNotIn('db', summary['metrics'][key])
        acc = MetricAccumulator().update({'waveform_nmse': np.array([np.nan])})
        self.assertEqual(acc.summary()['metrics']['waveform_nmse']['db_status'], 'undefined')
        json.dumps(acc.summary(), allow_nan=False)

    def test_underflow_is_not_misreported_as_zero_or_undefined(self):
        tiny = np.zeros((1, 8000))
        tiny[0, 160] = 1e-170
        with self.assertRaisesRegex(ValueError, 'underflow'):
            Evaluator().evaluate_batch(tiny, 2 * tiny)
        reference = np.zeros((1, 8000))
        reference[0, 160] = 1
        prediction = reference.copy()
        prediction[0, 170] = 1e-170
        with self.assertRaisesRegex(ValueError, 'underflow'):
            Evaluator().evaluate_batch(reference, prediction)

    def test_new_metrics_do_not_compute_old_metrics(self):
        ev = Evaluator()
        ev.measure_edt = lambda *a, **k: self.fail('acoustics must not run')
        from unittest.mock import patch
        with patch('torch.stft', side_effect=AssertionError('STFT must not run')):
            result = ev.evaluate_new_metrics(np.ones((1, 8000)), np.ones((1, 8000)))
        self.assertEqual(set(result['metrics']), {'phase_error_deg', 'early_energy_db_mae'})

    def test_empty_schema(self):
        summary = MetricAccumulator().summary()
        self.assertEqual(summary['samples_evaluated'], 0)
        self.assertEqual(len(summary['metrics']), 7)
        for metric in summary['metrics'].values():
            self.assertIsNone(metric['value'])
            self.assertEqual(metric['valid_count'], 0)
            self.assertEqual(metric['invalid_count'], 0)
        self.assertEqual(summary['metrics']['waveform_nmse']['db_status'], 'undefined')
        self.assertIsNone(summary['diagnostics']['phase_gt_energy_coverage']['mean'])
        json.dumps(summary, allow_nan=False)
        self.assertTrue(all(len(x) == 0 for x in Evaluator().evaluate_batch(np.empty((0, 8000)), np.empty((0, 8000))).values()))

    def test_partition_diagnostics_reasons_and_state(self):
        gt = np.zeros((3, 7992))
        gt[0] = np.tile([2., 0.], 3996)
        gt[2, 7900] = 1
        pred = gt.copy()
        pred[0] = 1
        result = Evaluator(signal_length=7992).evaluate_detailed(gt, pred)
        a = MetricAccumulator(signal_length=7992).update(**result, reference_counts=[1, 2, 2])
        a.skip(2)
        b = MetricAccumulator(signal_length=7992)
        for i in range(3):
            sliced = {group: {key: value[i:i+1] for key, value in entries.items()} for group, entries in result.items()}
            b.merge(MetricAccumulator(signal_length=7992).update(**sliced, reference_counts=[1 if i == 0 else 2]))
        b.merge(MetricAccumulator(signal_length=7992)).skip(2)
        self.assertEqual(a.summary(), b.summary())
        self.assertEqual(a.summary(), MetricAccumulator.from_state_dict(json.loads(json.dumps(a.state_dict(), allow_nan=False))).summary())
        summary = a.summary()
        self.assertEqual(summary['diagnostics']['phase_missing_frequency_total'], 1)
        self.assertEqual(summary['diagnostics']['phase_missing_sample_count'], 1)
        self.assertEqual(summary['diagnostics']['early_short_window_sample_count'], 1)
        self.assertEqual(summary['invalid_reasons']['phase_error_deg'], {'ground_truth_energy_zero': 1})
        self.assertEqual(summary['diagnostics']['early_start_sample']['valid_count'], 2)
        json.dumps(summary, allow_nan=False)

    def test_reject_mixed_protocol_unknown_metrics_and_old_state(self):
        a = MetricAccumulator()
        with self.assertRaisesRegex(ValueError, 'protocol'):
            a.merge(MetricAccumulator(signal_length=7992))
        state = a.state_dict()
        for field in ('protocol', 'signal_length'):
            bad = copy.deepcopy(state)
            bad.pop(field)
            with self.assertRaises(ValueError):
                MetricAccumulator.from_state_dict(bad)
        bad = copy.deepcopy(state)
        bad['sums']['retired_metric'] = 0
        with self.assertRaises(ValueError):
            MetricAccumulator.from_state_dict(bad)
        with self.assertRaises(ValueError):
            a.update({'retired_metric': np.zeros(1)})
        bad = copy.deepcopy(state)
        bad['protocol']['early_length'] = 900
        with self.assertRaises(ValueError):
            MetricAccumulator.from_state_dict(bad)


if __name__ == '__main__':
    unittest.main()
