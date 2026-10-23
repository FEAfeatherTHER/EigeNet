import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock
import torch
from dac.model.dac import DAC as DACModel

_SPEC = importlib.util.spec_from_file_location('dac_wrapper_test', Path(__file__).resolve().parents[1] / 'models/layers/dac_codec.py')
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)

def _tiny_codec():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return DACModel(
            encoder_dim=2,
            encoder_rates=[2],
            decoder_dim=8,
            decoder_rates=[2],
            n_codebooks=1,
            codebook_size=8,
            codebook_dim=2,
            sample_rate=16000,
        )

class TestPrequant(unittest.TestCase):
    def test_loads_real_checkpoint_with_frozen_eval_codec(self):
        model = _tiny_codec()
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'tiny_dac.pth'
            model.save(str(checkpoint), package=False)
            wrapper = _MOD.DAC(str(checkpoint))

        self.assertIsInstance(wrapper.model, DACModel)
        self.assertFalse(wrapper.model.training)
        self.assertTrue(all(not module.training for module in wrapper.model.modules()))
        self.assertTrue(all(not parameter.requires_grad for parameter in wrapper.model.parameters()))
        torch.testing.assert_close(wrapper.model.state_dict(), model.state_dict())

    def test_decode_z_returns_real_audio_with_latent_gradients(self):
        wrapper = _MOD.DAC.__new__(_MOD.DAC)
        torch.nn.Module.__init__(wrapper)
        wrapper.model = _tiny_codec().eval().requires_grad_(False)
        z = torch.linspace(-1.0, 1.0, 96).reshape(2, 4, 12).requires_grad_()

        audio = wrapper.decode_z(z)

        self.assertIsInstance(audio, torch.Tensor)
        self.assertEqual(audio.shape, (2, 1, 24))
        torch.testing.assert_close(audio, wrapper.model.decode(z)['audio'])
        self.assertTrue(torch.isfinite(audio).all().item())
        audio.square().mean().backward()
        self.assertIsNotNone(z.grad)
        self.assertTrue(torch.isfinite(z.grad).all().item())
        self.assertGreater(z.grad.abs().sum().item(), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in wrapper.model.parameters()))

    def test_encoder_direct_without_quantizer_or_encode(self):
        wrapper = _MOD.DAC.__new__(_MOD.DAC)
        torch.nn.Module.__init__(wrapper)
        model = torch.nn.Module()
        model.encoder = torch.nn.Conv1d(1, 4, 3)
        model.encode = Mock(side_effect=AssertionError('model.encode must not run'))
        model.quantizer = Mock(side_effect=AssertionError('quantizer must not run'))
        wrapper.model = model
        x = torch.randn(2, 1, 10)
        torch.testing.assert_close(wrapper.encode_z(x), model.encoder(x))
        model.encode.assert_not_called()
        model.quantizer.assert_not_called()

if __name__ == '__main__':
    unittest.main()
