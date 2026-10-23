import importlib.util
from pathlib import Path
import unittest

import torch


_SPEC = importlib.util.spec_from_file_location(
    'waveform_loss_test',
    Path(__file__).resolve().parents[1] / 'models/loss/waveform_loss.py',
)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


class TestWaveformLoss(unittest.TestCase):
    def setUp(self):
        self.loss = _MOD.loss_fn(sr=16000, duration=0.064, device='cpu')
        time = torch.arange(1024, dtype=torch.float32) / 16000
        self.target = (
            torch.sin(2 * torch.pi * 440 * time)
            + 0.25 * torch.cos(2 * torch.pi * 1130 * time)
        ).reshape(1, 1, -1)

    def test_identical_nonzero_audio_has_zero_losses(self):
        mrstft, time_edc, envelope = self.loss(self.target, self.target)

        self.assertEqual(mrstft.item(), 0.0)
        self.assertEqual(time_edc.item(), 0.0)
        self.assertEqual(envelope, 0.0)

    def test_different_audio_has_finite_losses_and_gradients(self):
        time = torch.arange(1024, dtype=torch.float32) / 16000
        prediction = (
            0.7 * self.target + 0.2 * torch.sin(2 * torch.pi * 730 * time)
        ).detach().requires_grad_()

        mrstft, time_edc, envelope = self.loss(prediction, self.target)

        self.assertTrue(torch.isfinite(mrstft).item())
        self.assertTrue(torch.isfinite(time_edc).item())
        self.assertGreater(mrstft.item(), 0.0)
        self.assertGreater(time_edc.item(), 0.0)
        self.assertEqual(envelope, 0.0)
        for component in (mrstft, time_edc):
            gradient, = torch.autograd.grad(component, prediction, retain_graph=True)
            self.assertTrue(torch.isfinite(gradient).all().item())
            self.assertGreater(gradient.abs().sum().item(), 0.0)


if __name__ == '__main__':
    unittest.main()
