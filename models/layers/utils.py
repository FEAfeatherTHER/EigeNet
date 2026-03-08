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
    """
        gt_ir : (b, 1, t)
        pred_ir : (b, 1, t)
    """
    if batch_gt_ir.ndim == 3:
        batch_gt_ir = batch_gt_ir[:,0]#(b, t)
    if batch_pred_ir.ndim == 3:
        batch_pred_ir = batch_pred_ir[:,0]#(b, t)
    
    count_outlier = False
    bsz = batch_gt_ir.shape[0]
    batch_edt_error_list = []
    batch_c50_error_list = []
    batch_t60_relative_error_list = []
    batch_t60_absolute_error_list = []
    batch_count_outlier = 0
    for i in range(bsz):
        gt_ir = batch_gt_ir[i]
        pred_ir = batch_pred_ir[i]

        gt_edt = evaluator.measure_edt(gt_ir)
        pred_edt = evaluator.measure_edt(pred_ir)
        edt_error = np.abs(gt_edt - pred_edt)
        if edt_error != np.inf and edt_error != -np.inf and gt_edt != -np.inf and pred_edt != -np.inf:
            batch_edt_error_list.append(edt_error)
        else:
            count_outlier = True
        gt_c50 = evaluator.measure_clarity(gt_ir)
        pred_c50 = evaluator.measure_clarity(pred_ir)
        c50_error = np.abs(gt_c50 - pred_c50)
        if c50_error != np.inf and c50_error != -np.inf and gt_c50 != -np.inf and pred_c50 != -np.inf:
            batch_c50_error_list.append(c50_error)
        else:
            count_outlier = True
        gt_t60 = evaluator.measure_rt60(gt_ir)
        pred_t60 = evaluator.measure_rt60(pred_ir)
        t60_relative_error = np.abs(gt_t60 - pred_t60) / gt_t60 * 100.
        t60_absolute_error = (pred_t60 - gt_t60) * 1000. #ms
        if np.abs(t60_relative_error) != np.inf and np.abs(t60_absolute_error) != np.inf:
            batch_t60_relative_error_list.append(t60_relative_error)
            batch_t60_absolute_error_list.append(t60_absolute_error)
        else:
            count_outlier = True

            # print(f"*"*20)
            # print(f"gt_t60: {gt_t60}")
            # print(f"pred_t60: {pred_t60}")
            # print(f"*"*20)
            # sample_length = int(0.363 * 16000)
            # gt_tgt_ir = gt_ir[None, :sample_length]
            # pred_tgt_ir = pred_ir[None, :sample_length]
            # fig, axs = plt.subplots(2, 1, figsize=(10, 10))

            # plot_waveform(gt_tgt_ir, 16000, title="GT", ax=axs[0])
            # plot_waveform(pred_tgt_ir, 16000, title="Pred", ax=axs[1])
            # pdf_path = os.path.join('/data/250010171/code/EigeNet/data/debug', f"debug.pdf")
            # plt.savefig(pdf_path)
            # plt.close()
            # print(f"save pdf to {pdf_path}")
            # exit()

        
        batch_count_outlier += count_outlier
    
    return batch_edt_error_list, batch_c50_error_list, batch_t60_relative_error_list, batch_count_outlier




if __name__ == '__main__':
    midi_path = '/data/250010171/code/AnyTrainer-midi2audio/data/test/乐器编辑5s_demo/恰似你的温柔_19.76_24.76.mid'
    midi = pretty_midi.PrettyMIDI(midi_path)
    midi_seq = Midi_To_Seqence(midi)
    print(midi_seq.shape)




        
            
   