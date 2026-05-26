# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time
import math
import torch
import torch.nn.functional as F
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
from accelerate.utils import ProjectConfiguration
import torch.nn as nn
from transformers import Wav2Vec2BertModel
from transformers import SeamlessM4TFeatureExtractor
import matplotlib.pyplot as plt


from models.base.base_trainer import BaseTrainer
#from models.dataset.acousticrooms_dataset import AcousticRooms_Dataset, AcousticRooms_Collator
from models.dataset.haa_dataset import HAA_dataset, HAA_collator
from models.loss.waveform_loss import loss_fn as waveform_loss_fn
from models.loss.evaluator import Evaluator
from models.layers.utils import compute_metrics, plot_waveform, convert_stft_db_to_octave_db, extract_stft_energy_db, spectrogram_edc_loss
from models.layers.dac_codec import DAC
from models.aa_v1_g2_align2_toy3.flow_matching_transformer.fmt_model import FlowMatchingTransformer


import shutil
import json
import time
from datetime import timedelta

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))


class FMTTrainer(BaseTrainer):
    def __init__(self, args, cfg):
        super(FMTTrainer, self).__init__(args, cfg)
        self.cfg = cfg
        self.sample_rate = cfg.preprocess.sample_rate
        self.visualize_dir = os.path.join(self.exp_dir, "visualize")
        os.makedirs(self.visualize_dir, exist_ok=True)
        self.evaluator = Evaluator()
        self.aligner_activate = self.cfg.model.flow_matching_transformer.aligner.activate

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
        model = FlowMatchingTransformer(
            duration=self.cfg.preprocess.duration,
            frame_rate=self.cfg.preprocess.frame_rate,
            cfg=self.cfg.model.flow_matching_transformer)
        return model

    def _build_dataset(self):
        return HAA_dataset, HAA_collator

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
        if audio.dim() == 3:
            audio = audio[:, 0]
        elif audio.dim() == 2:
            pass
        else:
            raise ValueError(f"The dimension of audio must be 2 or 3, but got {audio.dim()}")
        align_features = extract_stft_energy_db(audio, sr=self.sample_rate, frame_rate=self.cfg.preprocess.frame_rate)
        align_features = convert_stft_db_to_octave_db(align_features).transpose(1, 2)
        return align_features

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
        alignment_l1_loss = nn.L1Loss()(x, y)
        alignment_edc_loss = spectrogram_edc_loss(x, y)
        return alignment_l1_loss, alignment_edc_loss
    
    def decode_z_to_audio(self, z):
        pred_audio = self.audio_codec.decode_z(z)
        return pred_audio

    def _train_step(self, batch):
        train_losses = {}
        total_loss = 0
        train_stats = {}
        all_ir = batch["all_ir"].to(self.accelerator.device)  # [B, N, 1, T]
        gt_tgt_ir = all_ir[:,-1]
        #inter_ir = rearrange(all_ir, "b n 1 t -> (b n) 1 t").squeeze(-2) #(B*N, T)
        #gt_env = self.waveform_loss_fn.smooth_envelope_gaussian(self.waveform_loss_fn.get_envelope(inter_ir)).unsqueeze(1)#(B*N, 1, T)
        all_cc_src_loc = batch["all_cc_src_loc"].to(self.accelerator.device) # [B, N, 3]
        cc_depth_map = batch["cc_depth_map"].to(self.accelerator.device) # [B, 256, 512, 3]
        
        all_ir_z = self.encode_audio_to_z(all_ir) #(B, N, T, 1024)
        gt_tgt_ir = all_ir[:,-1]
        align_l1_loss = torch.zeros(1).to(self.accelerator.device)
        align_edc_loss = torch.zeros(1).to(self.accelerator.device)
        if not self.aligner_activate:
            pred_tgt_ir_z = self.model(all_ir_z, cc_depth_map, all_cc_src_loc)
        else:
            pred_tgt_ir_z, pred_align_features = self.model(all_ir_z, cc_depth_map, all_cc_src_loc)
            gt_align_features = self.get_align_features(gt_tgt_ir) #(B*N, D)
            align_l1_loss, align_edc_loss = self.get_align_loss(pred_align_features, gt_align_features)
        pred_tgt_ir = self.decode_z_to_audio(pred_tgt_ir_z.transpose(-1, -2))
        if torch.isnan(pred_tgt_ir).any() or torch.isinf(pred_tgt_ir).any():
            print(f"pred_tgt_ir is nan or inf")
            exit()
        
        mrstft_loss, time_edc_loss, spect_edc_loss, env_loss = self.waveform_loss_fn(pred_tgt_ir, gt_tgt_ir)
        
        main_loss = self.cfg.loss.mrstft_loss_weight * mrstft_loss
        auxiliary_loss = self.cfg.loss.time_edc_loss_weight * time_edc_loss \
            + self.cfg.loss.spect_edc_loss_weight * spect_edc_loss \
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
        train_losses["spect_edc_loss"] = spect_edc_loss
        train_losses["align_l1_loss"] = align_l1_loss
        train_losses["align_edc_loss"] = align_edc_loss
        #train_losses["env_loss"] = env_loss
        
        # print(f"*"*20)
        # print(f"检查loss")
        # print(f"batch_total_loss: {batch_total_loss.item()}")
        # print(f"mrstft_loss: {mrstft_loss.item()}")
        # print(f"time_edc_loss: {time_edc_loss.item()}")
        # print(f"spect_edc_loss: {spect_edc_loss.item()}")
        # print(f"env_loss: {env_loss.item()}")
        # print("--------------------------------")
        # exit()
        
        self.optimizer.zero_grad()
        self.accelerator.backward(total_loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()), 0.2
            )
        self.optimizer.step()
        self.scheduler.step()

        for item in train_losses:
            train_losses[item] = train_losses[item].item()

        self.current_loss = total_loss.item()

        train_losses["batch_size"] = all_ir_z.shape[0]
        train_losses["learning_rate"] = self.optimizer.param_groups[0]["lr"]

        return (total_loss.item(), train_losses, train_stats)

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
                    if hasattr(self.cfg, "valid"):
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
    
    @torch.no_grad()
    def _valid_step(self, batch):
        valid_metrics = {}

        all_ir = batch["all_ir"].to(self.accelerator.device)  # [B, N, 1, t]
        gt_tgt_ir = all_ir[:,-1]
        all_cc_src_loc = batch["all_cc_src_loc"].to(self.accelerator.device) # [B, N, 3]
        cc_depth_map = batch["cc_depth_map"].to(self.accelerator.device) # [B, 256, 512, 3]
        
        all_ir_z = self.encode_audio_to_z(all_ir) #(B, N, t, 1024)
        if not self.aligner_activate:
            pred_tgt_ir_z = self.model(all_ir_z, cc_depth_map, all_cc_src_loc)
        else:
            pred_tgt_ir_z, pred_align_features = self.model(all_ir_z, cc_depth_map, all_cc_src_loc)
            del pred_align_features
            
        pred_tgt_ir = self.decode_z_to_audio(pred_tgt_ir_z.transpose(-1, -2))
        valid_length = int(self.cfg.valid.duration * self.sample_rate)
        pred_tgt_ir = pred_tgt_ir[...,:valid_length].cpu().numpy()
        gt_tgt_ir = gt_tgt_ir[...,:valid_length].cpu().numpy()
        batch_edt_error_list, batch_c50_error_list, batch_t60_error_list, batch_count_outlier = compute_metrics(gt_tgt_ir, pred_tgt_ir, self.evaluator)
        
        valid_metrics["edt_error"] = round(np.mean(batch_edt_error_list), 4)
        valid_metrics["c50_error"] = round(np.mean(batch_c50_error_list), 4)
        valid_metrics["t60_error"] = round(np.mean(batch_t60_error_list), 4)
        valid_metrics["count_outlier"] = int(batch_count_outlier)

        return valid_metrics, gt_tgt_ir, pred_tgt_ir


    def _valid(self):
        r"""遍历整个验证集并计算平均 metrics"""
        if isinstance(self.model, dict):
            for key in self.model.keys():
                self.model[key].eval()
        else:
            self.model.eval()

        val_metrics_sum = {}
        val_count = 0

        for batch in self.valid_dataloader:
            # 将数据移动到设备
            device = self.accelerator.device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            # 执行验证步（需要在子类实现 _valid_step 的逻辑，或者通用的评估逻辑）
            valid_metrics, gt_tgt_ir, pred_tgt_ir = self._valid_step(batch)
            if self.cfg.valid.visualize_interval != 0:
                if val_count % self.cfg.valid.visualize_interval == 0:
                    #visualize the waveform
                    vis_gt_tgt_ir = gt_tgt_ir[0]
                    vis_pred_tgt_ir = pred_tgt_ir[0]
                    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

                    plot_waveform(vis_gt_tgt_ir, self.sample_rate, title="GT", ax=axs[0])
                    plot_waveform(vis_pred_tgt_ir, self.sample_rate, title="Pred", ax=axs[1])
                    cur_visualize_dir = os.path.join(self.visualize_dir, str(self.step))
                    os.makedirs(cur_visualize_dir, exist_ok=True)
                    png_path = os.path.join(cur_visualize_dir, f"{val_count}.png")
                    plt.savefig(png_path)
                    plt.close()
            
            # 累加 Loss
            for k, v in valid_metrics.items():
                if k not in val_metrics_sum:
                    val_metrics_sum[k] = 0.0
                val_metrics_sum[k] += v
            val_count += 1

        # 计算所有进程的平均值 (Gather from all processes)
        # 注意：accelerator.log 内部通常会处理同步，但手动平均更准确
        avg_metrics = {k: v / val_count for k, v in val_metrics_sum.items()}
        
        # 切回训练模式
        if isinstance(self.model, dict):
            for key in self.model.keys():
                self.model[key].train()
        else:
            self.model.train()
            
        return avg_metrics
