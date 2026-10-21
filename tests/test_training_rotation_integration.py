"""Training-only wiring and reproducible, rank-specific augmentation randomness."""
import random
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from models.EigeNet.src.trainer import EigeNetTrainer


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.9))

    def forward(self, references, depth, locations):
        self.geometry = (depth, locations)
        return references.mean(1) * self.weight


def trainer_for_test(rank=0):
    trainer = EigeNetTrainer.__new__(EigeNetTrainer)
    trainer.cfg = SimpleNamespace(
        train=SimpleNamespace(random_seed=114, gradient_accumulation_step=2,
                              auxiliary_loss_warmup_steps=0, alignment_loss_start_step=0,
                              alignment_loss_warmup_steps=0),
        loss=SimpleNamespace(mrstft_loss_weight=1., time_edc_loss_weight=0.,
                             env_loss_weight=0., align_l1_loss_weight=0.,
                             align_edc_loss_weight=0.))
    trainer.accelerator = SimpleNamespace(
        device=torch.device('cpu'), process_index=rank, num_processes=2,
        sync_gradients=True, backward=lambda loss: loss.backward(),
        clip_grad_norm_=torch.nn.utils.clip_grad_norm_)
    trainer.step, trainer.batch_count = 3, 6
    return trainer


class TrainingRotationTests(unittest.TestCase):
    def test_generator_reproducible_rank_and_microbatch_specific(self):
        trainer = trainer_for_test()
        draw = lambda t: torch.rand(16, generator=t._augmentation_generator())
        expected = draw(trainer)
        self.assertTrue(torch.equal(expected, draw(trainer)))
        self.assertFalse(torch.equal(expected, draw(trainer_for_test(rank=1))))
        trainer.batch_count += 1
        self.assertFalse(torch.equal(expected, draw(trainer)))
        trainer.batch_count = 0  # A new trainer resuming at an optimizer boundary.
        self.assertTrue(torch.equal(expected, draw(trainer)))
        trainer.step += 1
        self.assertFalse(torch.equal(expected, draw(trainer)))

    def test_generator_does_not_advance_global_random_states(self):
        py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        torch.rand(100, generator=trainer_for_test()._augmentation_generator())
        self.assertEqual(py_state, random.getstate())
        np.testing.assert_array_equal(np_state[1], np.random.get_state()[1])
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))

    def test_train_step_uses_augmented_geometry_and_original_audio(self):
        trainer = trainer_for_test()
        trainer.aligner_activate = False
        trainer.model = RecordingModel()
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
        trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.)
        trainer.encode_audio_to_z = lambda audio: audio.transpose(-1, -2)
        trainer.decode_z_to_audio = lambda latent: latent
        trainer.waveform_loss_fn = lambda pred, target: ((pred - target).square().mean(),
                                                       pred.sum() * 0, pred.sum() * 0)
        batch = {'all_ir': torch.ones(2, 3, 1, 16),
                 'cc_depth_map': torch.ones(2, 3, 4, 8),
                 'all_cc_src_loc': torch.ones(2, 3, 3)}
        original = {key: value.clone() for key, value in batch.items()}
        called = []

        def augment(depth, locations, generator):
            called.append(generator.initial_seed())
            return depth + 2, locations + 3

        trainer.rotation_augmentation = augment
        before = trainer.model.weight.detach().clone()
        loss, _, _ = trainer._train_step(batch)
        self.assertEqual(len(called), 1)
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(torch.equal(before, trainer.model.weight))
        torch.testing.assert_close(trainer.model.geometry[0], original['cc_depth_map'] + 2)
        torch.testing.assert_close(trainer.model.geometry[1], original['all_cc_src_loc'] + 3)
        for key in batch:
            self.assertTrue(torch.equal(batch[key], original[key]))


if __name__ == '__main__':
    unittest.main()
