import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock
import torch

_SPEC = importlib.util.spec_from_file_location('dac_wrapper_test', Path(__file__).resolve().parents[1] / 'models/layers/dac_codec.py')
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)

class TestPrequant(unittest.TestCase):
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
