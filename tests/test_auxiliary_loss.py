"""Auxiliary losses accumulate time and normalize each octave band separately."""
import math
import unittest

import torch

from models.EigeNet.src.trainer import EigeNetTrainer


def decay_and_constant_db():
    band_scale = torch.arange(1, 9, dtype=torch.float64).reshape(1, 1, 8)
    batch_scale = torch.tensor([1., 3.], dtype=torch.float64).reshape(2, 1, 1)
    target_power = torch.tensor([8., 4., 2.], dtype=torch.float64).reshape(1, 3, 1)
    target_power = target_power * band_scale * batch_scale
    prediction_power = torch.full_like(target_power, 8.) * band_scale * batch_scale
    return 10 * prediction_power.log10(), 10 * target_power.log10()


def expected_decay_edc():
    # Reverse-time sums: prediction [24, 16, 8], target [14, 6, 2].
    return sum(
        abs(10 * math.log10(pred / 24) - 10 * math.log10(target / 14))
        for pred, target in ((24, 14), (16, 6), (8, 2))
    ) / 3


class AuxiliaryLossTests(unittest.TestCase):
    def setUp(self):
        self.trainer = EigeNetTrainer.__new__(EigeNetTrainer)

    def test_decay_versus_constant_matches_hand_derived_time_edc(self):
        prediction, target = decay_and_constant_db()
        l1, edc = self.trainer.get_align_loss(prediction, target)
        self.assertAlmostEqual(edc.item(), expected_decay_edc(), places=8)
        self.assertAlmostEqual(l1.item(), 10 * math.log10(2), places=8)

    def test_different_band_profiles_have_independent_normalization(self):
        target_power = torch.tensor([
            [8., 2., 16., 20., 24., 40., 32., 80.],
            [4., 4., 8., 40., 12., 80., 16., 160.],
            [2., 8., 4., 80., 6., 160., 8., 320.],
        ], dtype=torch.float64).unsqueeze(0)
        prediction_power = torch.tensor(
            [8., 2., 16., 20., 24., 40., 32., 80.],
            dtype=torch.float64,
        ).reshape(1, 1, 8).expand_as(target_power)
        _, edc = self.trainer.get_align_loss(
            10 * prediction_power.log10(), 10 * target_power.log10(),
        )
        # Alternating bands have different prediction/target total ratios:
        # [24,16,8]/[14,6,2] versus [6,4,2]/[14,12,8], before band scaling.
        # Their first normalized samples are zero; average the other four.
        expected = (
            abs(10 * math.log10(16 / 24) - 10 * math.log10(6 / 14))
            + abs(10 * math.log10(8 / 24) - 10 * math.log10(2 / 14))
            + abs(10 * math.log10(4 / 6) - 10 * math.log10(12 / 14))
            + abs(10 * math.log10(2 / 6) - 10 * math.log10(8 / 14))
        ) / 6
        self.assertAlmostEqual(edc.item(), expected, places=8)

    def test_identical_features_have_zero_auxiliary_losses(self):
        _, target = decay_and_constant_db()
        l1, edc = self.trainer.get_align_loss(target, target)
        self.assertEqual(l1.item(), 0.)
        self.assertEqual(edc.item(), 0.)

    def test_uniform_power_multiplier_changes_l1_but_preserves_edc(self):
        _, target = decay_and_constant_db()
        prediction = target + 10 * math.log10(4)
        l1, edc = self.trainer.get_align_loss(prediction, target)
        self.assertAlmostEqual(l1.item(), 10 * math.log10(4), places=8)
        self.assertLess(edc.item(), 1e-9)

    def test_unequal_time_lengths_compare_only_the_shared_prefix(self):
        prediction, target = decay_and_constant_db()
        tail = torch.full((2, 2, 8), 90., dtype=torch.float64)
        for longer in ('prediction', 'target'):
            with self.subTest(longer=longer):
                x = torch.cat((prediction, tail), dim=1) if longer == 'prediction' else prediction
                y = torch.cat((target, -tail), dim=1) if longer == 'target' else target
                l1, edc = self.trainer.get_align_loss(x, y)
                self.assertAlmostEqual(l1.item(), 10 * math.log10(2), places=8)
                self.assertAlmostEqual(edc.item(), expected_decay_edc(), places=8)

    def test_edc_alone_has_finite_nonzero_prediction_gradients(self):
        prediction, target = decay_and_constant_db()
        prediction.requires_grad_()
        _, edc = self.trainer.get_align_loss(prediction, target)
        edc.backward()
        self.assertTrue(torch.isfinite(edc))
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(prediction.grad.abs().sum().item(), 1e-5)

    def test_combined_auxiliary_loss_has_finite_nonzero_gradients(self):
        prediction, target = decay_and_constant_db()
        prediction.requires_grad_()
        l1, edc = self.trainer.get_align_loss(prediction, target)
        combined = l1 + edc
        combined.backward()
        self.assertTrue(torch.isfinite(combined))
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(prediction.grad.abs().sum().item(), 1e-5)


if __name__ == '__main__':
    unittest.main()
