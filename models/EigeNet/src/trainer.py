# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
#import torchaudio
#import torchvision
from librosa.feature import chroma_stft
import random
import numpy as np
import accelerate
import os
import math
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration, TorchDynamoPlugin

from transformers import Wav2Vec2BertModel
from transformers import SeamlessM4TFeatureExtractor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.base.base_trainer import BaseTrainer
from models.dataset.acousticrooms_dataset import AcousticRooms_Dataset, AcousticRooms_Collator
from models.dataset.rotation_augmentation import RotationAugmentation

from models.loss.waveform_loss import loss_fn as waveform_loss_fn
from models.loss.evaluator import Evaluator
from models.layers.utils import OctaveEnergyFeatures, spectrogram_edc_loss
from models.layers.dac_codec import DAC
from models.EigeNet.src.model import EigeNetTransformer
from models.EigeNet.src.evaluate import (
    EigeNetPredictor, build_evaluation_loader, evaluate_loader,
    scalar_log_values, write_summary,
)

import shutil
import json
import time
from datetime import timedelta

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))


class EigeNetTrainer(BaseTrainer):
    def __init__(self, args, cfg):
        augmentation = cfg.train.rotation_augmentation
        self.rotation_augmentation = RotationAugmentation(
            enabled=augmentation.enabled, probability=augmentation.probability,
        )
        super(EigeNetTrainer, self).__init__(args, cfg)
        self.cfg = cfg
        self.sample_rate = cfg.preprocess.sample_rate
        self.visualize_dir = os.path.join(self.exp_dir, "visualize")
        os.makedirs(self.visualize_dir, exist_ok=True)
        self.evaluator = Evaluator()
        self.aligner_activate = self.cfg.model.eigenet_transformer.aligner.activate
        self.waveform_loss_fn = waveform_loss_fn(
            sr=self.cfg.preprocess.sample_rate,
            duration=self.cfg.preprocess.duration,
            device=self.accelerator.device,
            env_duration=self.cfg.loss.env.duration,
            smooth=self.cfg.loss.env.smooth,
            env_window=self.cfg.loss.env.window,
            env_loss_alpha=self.cfg.loss.env.alpha,
        )
        self._build_output_model()
        if self.aligner_activate:
            # Fixed windows and frequency boundaries are reused across steps;
            # target extraction remains outside the compiled model.
            self.alignment_features = OctaveEnergyFeatures(
                sr=self.sample_rate, frame_rate=self.cfg.preprocess.frame_rate,
            ).to(self.accelerator.device)

    def _init_accelerator(self):
        self.exp_dir = os.path.join(
            os.path.abspath(self.cfg.log_dir), self.args.exp_name
        )
        project_config = ProjectConfiguration(
            project_dir=self.exp_dir,
            #logging_dir=os.path.join(self.exp_dir, "log"),
        )
        from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
        ddp_kwargs = [InitProcessGroupKwargs(timeout=timedelta(seconds=3*3600)), DistributedDataParallelKwargs(find_unused_parameters=True)]
        self.accelerator = accelerate.Accelerator(
            gradient_accumulation_steps=self.cfg.train.gradient_accumulation_step,
            log_with=self.cfg.tracker.type,
            project_config=project_config,
            kwargs_handlers=ddp_kwargs,
            # K varies between batches although every RIR is 0.5 s long.
            # Accelerate applies compilation after distributed model wrapping.
            dynamo_plugin=TorchDynamoPlugin(
                backend="inductor", mode="default", dynamic=True, fullgraph=False,
                # PyTorch 2.8 splits RMSNorm's weight-gradient reduction using
                # a shape-dependent factor, generating fractional Triton indices
                # for variable reference counts. Keep the reduction unsplit;
                # dynamic shapes, FlexAttention and distributed training remain enabled.
                options={"split_reductions": False},
            ),
        )
        if self.accelerator.is_main_process:
            os.makedirs(project_config.project_dir, exist_ok=True)
            #os.makedirs(project_config.logging_dir, exist_ok=True)
        with self.accelerator.main_process_first():
            if self.cfg.tracker.dummy_run:
                pass
            else:
                self.accelerator.init_trackers(
                    project_name=self.cfg.tracker.project_name,
                    config=self.cfg.tracker.config.to_dict(),
                    init_kwargs={self.cfg.tracker.type: {"experiment_name": self.args.exp_name}})

    def loss_weight_warmup_factor(self, type = 'auxiliary'):
        """calculate the linear weight factor for the loss weights"""
        if type == 'auxiliary':
            if self.step < self.cfg.train.auxiliary_loss_warmup_steps:
                return self.step / float(self.cfg.train.lr_warmup_steps)
            return 1.0
        if type == 'alignment':
            start_step = self.cfg.train.alignment_loss_start_step
            if self.step < start_step:
                return 0.0 
            elif self.step < self.cfg.train.alignment_loss_warmup_steps:
                return self.step / float(self.cfg.train.lr_warmup_steps-start_step)
            return 1.0

    def _build_model(self):
        model = EigeNetTransformer(
            duration=self.cfg.preprocess.duration,
            frame_rate=self.cfg.preprocess.frame_rate,
            cfg=self.cfg.model.eigenet_transformer)
        return model

    def _build_dataset(self):
        return AcousticRooms_Dataset, AcousticRooms_Collator

    def _build_validation_dataloader(self):
        # Evaluation uses independent local RNGs and non-overlapping shards.
        # Do not accelerator.prepare this loader: padding would duplicate targets.
        settings = self.cfg.evaluation
        return build_evaluation_loader(
            self.cfg, max_references=settings.max_references, seed=settings.seed,
            rank=self.accelerator.process_index,
            world_size=self.accelerator.num_processes,
            num_workers=self.cfg.train.dataloader.num_worker,
        )

    def _build_output_model(self):
        self.audio_codec = DAC(self.cfg.model.DAC.path)
        self.audio_codec.eval()
        self.audio_codec.to(self.accelerator.device)
    
    @torch.no_grad()
    def encode_audio_to_z(self, audio): # 
        
        bsz, irs_num, channel, audio_len = audio.shape
        audio = audio.reshape(-1, channel, audio_len)

        z = self.audio_codec.encode_z(audio)#(B*N, 1024, t)

        dim, frames = z.shape[-2], z.shape[-1]
        z = z.reshape(bsz, irs_num, dim, frames) #(B, N, 1024, t)
        z = z.transpose(-1, -2) #(B, N, t, 1024)
        return z
    
    def get_align_features(self, audio): # 
        """
        args:
        audio:(B, 1, T)
        return:
        align_features:(B, t, 8)
        """
        return self.alignment_features(audio)

    def get_align_loss(self, x, y):
        """
        Args:
        x:(B, t, 8)
        y:(B, t, 8)
        Returns:
        align_loss:(1,)
        """
        min_length = min(x.shape[1], y.shape[1])
        x = x[:, :min_length]
        y = y[:, :min_length]
        alignment_l1_loss = F.l1_loss(x, y)
        alignment_edc_loss = spectrogram_edc_loss(x, y)
        return alignment_l1_loss, alignment_edc_loss
    
    def decode_z_to_audio(self, z):
        pred_audio = self.audio_codec.decode_z(z)
        return pred_audio

    def _augmentation_generator(self):
        """Separate augmentation RNG, reproducible at optimizer-step boundaries."""
        accumulation = self.cfg.train.gradient_accumulation_step
        microstep = self.step * accumulation + self.batch_count % accumulation
        seed = (self.cfg.train.random_seed + self.accelerator.process_index
                + self.accelerator.num_processes * microstep) % (2**63 - 1)
        return torch.Generator(device=self.accelerator.device).manual_seed(seed)

    def _train_step(self, batch):
        train_losses = {}
        total_loss = 0
        train_stats = {}
        all_ir = batch["all_ir"].to(self.accelerator.device)  # [B, N, 1, T]
        gt_tgt_ir = all_ir[:,-1]
        all_cc_src_loc = batch["all_cc_src_loc"].to(self.accelerator.device) # [B, N, 3]
        cc_depth_map = batch["cc_depth_map"].to(self.accelerator.device) # [B, 3, H, W]
        cc_depth_map, all_cc_src_loc = self.rotation_augmentation(
            cc_depth_map, all_cc_src_loc, generator=self._augmentation_generator(),
        )
        
        # The model creates target tokens from time and geometry, so encode
        # references only. Target audio is still used by all active losses.
        ref_ir_z = self.encode_audio_to_z(all_ir[:, :-1])
        align_l1_loss = all_ir.new_zeros(())
        align_edc_loss = all_ir.new_zeros(())
        if not self.aligner_activate:
            pred_tgt_ir_z = self.model(ref_ir_z, cc_depth_map, all_cc_src_loc)
        else:
            pred_tgt_ir_z, pred_align_features = self.model(ref_ir_z, cc_depth_map, all_cc_src_loc)
            gt_align_features = self.get_align_features(gt_tgt_ir) 
            align_l1_loss, align_edc_loss = self.get_align_loss(pred_align_features, gt_align_features)
        pred_tgt_ir = self.decode_z_to_audio(pred_tgt_ir_z.transpose(-1, -2))
        if torch.isnan(pred_tgt_ir).any() or torch.isinf(pred_tgt_ir).any():
            print(f"pred_tgt_ir is nan or inf")
            exit()
        
        mrstft_loss, time_edc_loss, env_loss = self.waveform_loss_fn(pred_tgt_ir, gt_tgt_ir)
        
        main_loss = self.cfg.loss.mrstft_loss_weight * mrstft_loss
        auxiliary_loss = self.cfg.loss.time_edc_loss_weight * time_edc_loss \
            + self.cfg.loss.env_loss_weight * env_loss
        alignment_loss_weight_scale = self.loss_weight_warmup_factor(type = 'alignment')
        alignment_loss = self.cfg.loss.align_l1_loss_weight * align_l1_loss\
             + self.cfg.loss.align_edc_loss_weight * align_edc_loss
        auxiliary_loss_weight_scale = self.loss_weight_warmup_factor(type = 'auxiliary')
        batch_total_loss = main_loss\
            + alignment_loss_weight_scale * alignment_loss \
            + auxiliary_loss_weight_scale * auxiliary_loss

        total_loss += batch_total_loss
        train_losses["batch_total_loss"] = batch_total_loss
        train_losses["mrstft_loss"] = mrstft_loss
        train_losses["time_edc_loss"] = time_edc_loss
        train_losses["align_l1_loss"] = align_l1_loss
        train_losses["align_edc_loss"] = align_edc_loss
        #train_losses["env_loss"] = env_loss
        
        self.optimizer.zero_grad()
        self.accelerator.backward(total_loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()), 0.2
            )
        self.optimizer.step()
        self.scheduler.step()

        # One device-to-host transfer replaces a synchronization per metric.
        loss_values = torch.stack([loss.detach().reshape(()) for loss in train_losses.values()]).cpu().tolist()
        train_losses = dict(zip(train_losses, loss_values))
        self.current_loss = train_losses["batch_total_loss"]

        train_losses["batch_size"] = ref_ir_z.shape[0]
        train_losses["learning_rate"] = self.optimizer.param_groups[0]["lr"]

        return (self.current_loss, train_losses, train_stats)

    def _train_epoch(self):
        r"""Training epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
        if isinstance(self.model, dict):
            for key in self.model.keys():
                self.model[key].train()
        else:
            self.model.train()

        epoch_sum_loss: float = 0.0
        epoch_losses: dict = {}
        epoch_step: int = 0
        ema_loss = None

        for batch in self.train_dataloader:
            # Put the data to cuda device
            device = self.accelerator.device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
        
        # B = 16
        # T1 = 100
        # T = 480000
        # T_mel = 1000 # 480 hop size
        
        # self.fake_prefix = torch.randint(0, 100, (B, T1)).to(self.accelerator.device)  # [B, T1], int
        # # self.fake_prefix_mask = torch.randint(0, 2, (B, T1)).to(self.accelerator.device)  # [B, T1], 1/0
        # self.fake_prefix_mask = torch.ones(B, T1).to(self.accelerator.device)  # [B, T1], 1/0
        # self.fake_x_wav = torch.randn(B, T).to(self.accelerator.device)  # [B, T], float
        # self.fake_x_mel = torch.randn(B, T_mel, 128).to(self.accelerator.device)  # [B, T//480, 128], float
        # # self.fake_x_mask = torch.randint(0, 2, (B, T_mel)).to(self.accelerator.device)  # [B, T//480]
        # self.fake_x_mask = torch.ones(B, T_mel).to(self.accelerator.device)  # [B, T//480]
        
        # for i in range(100000):
        #     batch = None

            # Do training step and BP
            with self.accelerator.accumulate(self.model):
                total_loss, train_losses, training_stats = self._train_step(batch)
            self.batch_count += 1
            ema_loss = (
                0.98 * ema_loss + 0.02 * self.current_loss
                if ema_loss is not None
                else self.current_loss
            )
            # Update info for each step
            # TODO: step means BP counts or batch counts?
            if self.batch_count % self.cfg.train.gradient_accumulation_step == 0:
                epoch_sum_loss = total_loss
                for key, value in train_losses.items():
                    epoch_losses[key] = value

                if isinstance(train_losses, dict):
                    for key, loss in train_losses.items():
                        self.accelerator.log(
                            {"Epoch/Train {} Loss".format(key): loss},
                            step=self.step,
                        )

                if (
                    self.accelerator.is_main_process
                    and self.batch_count
                    % (10 * self.cfg.train.gradient_accumulation_step)
                    == 0
                ):
                    self.echo_log(train_losses, mode="Training")

                self.step += 1
                epoch_step += 1

                if self.step % self.cfg.train.save_checkpoints_steps == 0:
                    if hasattr(self.cfg, "valid") and self.run_eval:
                        if self.accelerator.is_main_process:
                            print(f"Starting validation at step {self.step}...")
                        # 执行验证遍历
                        val_metrics = self._valid()
                        
                        # 使用 SwanLab 记录 (通过 accelerate.log)
                        formatted_metrics = {f"Validation/{k}": v for k, v in val_metrics.items()}
                        if self.accelerator.is_main_process:
                            for k, v in formatted_metrics.items():
                                print(f"Validation/{k}: {v}")
                        self.accelerator.log(formatted_metrics, step=self.step)
                    self.save_checkpoint()

                if self.accelerator.is_main_process:
                    if self.step % 100 == 0:
                        print(f"EMA Loss: {ema_loss:.6f}")

        self.accelerator.wait_for_everyone()
        
        # if self.accelerator.is_main_process:
        #     # renew the dataloader
        #     if not hasattr(self.cfg.dataset, "return_speech"):
        #         self.train_dataloader, self.valid_dataloader = self._build_dataloader()
        
        # self.accelerator.wait_for_everyone()
        
        # if not self.accelerator.is_main_process:
        #     if not hasattr(self.cfg.dataset, "return_speech"):
        #         self.train_dataloader, self.valid_dataloader = self._build_dataloader()

        return epoch_sum_loss, epoch_losses
    
    def _valid(self):
        # DDP forwards may synchronize buffers. Evaluation shards can have
        # different lengths and reference counts, so use the shared underlying
        # weights without the DDP wrapper and aggregate only after every shard.
        model_was_training = self.model.training
        codec_was_training = self.audio_codec.training
        self.model.eval()
        self.audio_codec.eval()
        try:
            if not hasattr(self, "_evaluation_predictor"):
                raw_model = self.accelerator.unwrap_model(
                    self.model, keep_torch_compile=False,
                )
                if self.accelerator.device.type == "cuda":
                    # A separate cached forward graph shares parameters with the
                    # training model; it does not copy stale checkpoint weights.
                    raw_model = torch.compile(
                        raw_model, backend="inductor", dynamic=True,
                        options={"split_reductions": False},
                    )
                self._evaluation_predictor = EigeNetPredictor(
                    raw_model, self.audio_codec, self.accelerator.device,
                )
            summary = evaluate_loader(
                self.valid_dataloader, self._evaluation_predictor,
            ).summary()
            summary["evaluation"] = {
                "step": self.step,
                "seed": self.cfg.evaluation.seed,
                "max_references": self.cfg.evaluation.max_references,
                "weights": "current_training_parameters",
            }
            if self.accelerator.is_main_process:
                # Each invocation gets a new directory, including a repeated
                # step after resume; existing evaluation results are preserved.
                import tempfile
                output_root = os.path.join(self.exp_dir, "evaluation")
                os.makedirs(output_root, exist_ok=True)
                output_dir = tempfile.mkdtemp(
                    prefix=f"step-{self.step:07d}-", dir=output_root,
                )
                write_summary(summary, output_dir)
            return scalar_log_values(summary)
        finally:
            self.model.train(model_was_training)
            self.audio_codec.train(codec_was_training)
