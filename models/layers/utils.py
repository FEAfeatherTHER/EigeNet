from __future__ import annotations

import os
from pickle import FALSE
import random
random.seed(42)
from collections import defaultdict
from importlib.resources import files
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import librosa
import matplotlib.pyplot as plt
import torch.nn.functional as F

def is_package_available(package_name: str) -> bool:
    try:
        import importlib

        package_exists = importlib.util.find_spec(package_name) is not None
        return package_exists
    except Exception:
        return False

def Midi_To_Seqence(midi, duration: float = None, unison: int = 8, fs: int = 25) -> torch.Tensor:
    if duration is None:
        clip_duration_sec = midi.get_end_time()
    else:
        clip_duration_sec = duration
    frame_length = int(clip_duration_sec * fs)+1
    inst = midi.instruments[0]
    pitch_seqence = [[] for _ in range(frame_length)]
    event_sequence = [[] for _ in range(frame_length)]

    for note in inst.notes:
        start_frame = int(note.start * fs)
        end_frame = int(note.end * fs)
        event_sequence[start_frame].append(note.velocity/127)
        for frame in range(start_frame, end_frame+1):
            present_frame_unison = len(pitch_seqence[frame])
            if present_frame_unison < unison:
                pitch_seqence[frame].append(note.pitch + 1)
            else:
                continue
    #zero padding
    for frame in range(frame_length):
        present_pitch_unison = len(pitch_seqence[frame])
        if present_pitch_unison < unison:
            pitch_seqence[frame].extend([0] * (unison - present_pitch_unison))
        elif present_pitch_unison >= unison:
            pitch_seqence[frame] = pitch_seqence[frame][:unison]
        present_event_unison = len(event_sequence[frame])
        if present_event_unison < unison:
            event_sequence[frame].extend([0] * (unison - present_event_unison))
        elif present_event_unison >= unison:
            event_sequence[frame] = event_sequence[frame][:unison]
    #transform to np.array
    pitch_seqence = np.array(pitch_seqence)
    event_sequence = np.array(event_sequence)
    midi_sequence = np.stack([pitch_seqence, event_sequence], axis=0)
    return midi_sequence

def load_wav(sample_rate, wav_path, device, mono = True):
    speech = librosa.load(wav_path, sr=sample_rate, mono = mono)[0]
    if mono == True:
        pass
    elif mono == False:
        if speech.ndim == 1:
            speech = np.stack([speech, speech], axis=0)#(2, T)
        elif speech.ndim == 2 and speech.shape[0] == 2:
            pass
    speech_tensor = torch.tensor(speech).to(device)
    return speech_tensor

def vad(audio_file, sample_rate, start, end, silence_ratio=0.9, silence_threshold=0.01):
    """
    检测音频是否80%都是空白的（无声部分）。
    
    :param audio_file: 输入音频文件路径
    :param silence_threshold: 静音部分的阈值(归一化后振幅)，低于该值视为静音
    :param silence_ratio: 认为音频空白的比例，如果超过这个比例则认为是空白
    :return: 如果音频超过80%是静音，则返回True，否则返回False
    """
    # 加载音频文件
    duration = end - start
    audio = librosa.load(audio_file, sr=sample_rate, offset=start, duration=duration)[0]
    maxamplitude = np.max(np.abs(audio))
    if maxamplitude <= 0.001:
        return 1.0, False
    # 计算音频的绝对幅度
    normalized_amplitude = np.abs(audio) / maxamplitude
    
    # 计算静音部分的数量，静音是指振幅小于阈值的部分
    silence = normalized_amplitude < silence_threshold
    
    # 计算静音比例
    silence_percentage = np.sum(silence) / len(audio)
    
    # 判断是否超过静音比例的阈值
    if silence_percentage > silence_ratio:
        return silence_percentage, False  # 音频大部分是静音,没通过vad测试
    else:
        return silence_percentage, True  # 音频有足够的声音,通过vad测试

def plot_waveform(waveform, sr, title="Waveform", ax=None):

        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / sr

        if ax is None:
            _, ax = plt.subplots(num_channels, 1)
        ax.plot(time_axis, waveform[0], linewidth=1)
        ax.grid(True)
        ax.set_xlim([0, time_axis[-1]])
        ax.set_title(title)

def compute_metrics(batch_gt_ir, batch_pred_ir, evaluator):
    """Return named per-sample metrics; accumulation belongs to the evaluator.

    This replaces the former four-tuple/batch-average interface. All signals
    reach the evaluator before its separate signal and acoustic crops.
    """
    return evaluator.evaluate_batch(batch_gt_ir, batch_pred_ir)


class StreamingMean:
    def __init__(self):
        self.mean = None
        self.n = 0
        
    def update(self, x: np.ndarray):
        """传入新的 np.array 来更新均值"""
        self.n += 1
        if self.mean is None:
            # 第一笔数据直接作为初始均值
            self.mean = x.copy()
        else:
            # 动态更新公式
            self.mean += (x - self.mean) / self.n

class OctaveEnergyFeatures(torch.nn.Module):
    """Fixed STFT setup for alignment targets, with reusable window/band slices."""

    def __init__(self, sr=16000, frame_rate=50, eps=1e-10):
        super().__init__()
        self.hop_length = sr // frame_rate
        self.n_fft = 1024
        self.eps = eps
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)
        bands = torch.tensor([
            (31.25, 62.5), (62.5, 125), (125, 250), (250, 500),
            (500, 1000), (1000, 2000), (2000, 4000), (4000, 7999.0),
        ])
        # The original masks used low <= frequency < high. Resolve those fixed
        # boundaries once on CPU, avoiding GPU boolean indexing and .any() waits.
        freqs = torch.linspace(0, sr / 2, self.n_fft // 2 + 1)
        self.band_slices = tuple(tuple(pair) for pair in torch.searchsorted(freqs, bands).tolist())

    def forward(self, audio):
        if audio.ndim == 3:
            audio = audio[:, 0]
        elif audio.ndim != 2:
            raise ValueError("Alignment audio must have shape (B, T) or (B, 1, T)")
        spectrum = torch.stft(
            audio, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.n_fft, window=self.window, center=True, return_complex=True,
        )
        # Preserve the original dB round-trip and epsilon placement exactly.
        energy_db = 10 * torch.log10(torch.abs(spectrum) ** 2 + self.eps)
        power = 10 ** (energy_db / 10.0)
        octave_power = torch.stack(
            [power[:, low:high, :].sum(dim=1) for low, high in self.band_slices],
            dim=1,
        )
        return (10 * torch.log10(octave_power + self.eps)).transpose(1, 2)


def spectrogram_edc_loss(pred_db, gt_db, eps=1e-10):
    """
    计算频域/倍频程的 EDC (能量衰减曲线) Loss
    假设输入形状为 (B, F_bins, T)
    """
    # 1. 退回线性功率域 (因为 EDC 积分必须在能量域进行)
    pred_power = 10 ** (pred_db / 10.0)
    gt_power = 10 ** (gt_db / 10.0)
    
    # 2. 施罗德反向积分 (从后往前累加能量)
    # torch.flip 两次是为了实现时间维度上的逆向 cumsum
    pred_edc = torch.flip(torch.cumsum(torch.flip(pred_power, dims=[-1]), dim=-1), dims=[-1])
    gt_edc = torch.flip(torch.cumsum(torch.flip(gt_power, dims=[-1]), dim=-1), dims=[-1])
    
    # 3. 转回 dB 域，计算衰减曲线的 MSE
    pred_edc_db = 10*torch.log10(pred_edc + eps)
    gt_edc_db = 10*torch.log10(gt_edc + eps)
    
    # 算 MSE，并且可以通过减去第一帧的能量来进行归一化 (只关注斜率)
    pred_edc_norm = pred_edc_db - pred_edc_db[..., :1]
    gt_edc_norm = gt_edc_db - gt_edc_db[..., :1]
    
    return F.l1_loss(pred_edc_norm, gt_edc_norm) # EDC 用 L1 往往比 MSE 更稳

if __name__ == '__main__':
    midi_path = '/data/250010171/code/AnyTrainer-midi2audio/data/test/乐器编辑5s_demo/恰似你的温柔_19.76_24.76.mid'
    midi = pretty_midi.PrettyMIDI(midi_path)
    midi_seq = Midi_To_Seqence(midi)
    print(midi_seq.shape)




        
            
   