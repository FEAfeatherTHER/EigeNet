import os
import torch
import torch.nn.functional as F
import librosa
import auraloss 
import torch.nn as nn


class Envelope_Loss(nn.Module):
    def __init__(self, alpha=0.5, eps=1e-6):
        """
        pearson correlation + magnitude similarity
        :param alpha: weight coefficient. 
        :param eps:
        """
        super(Envelope_Loss, self).__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, pred, target):
        # --- Energy Loss ---
        pred_db = 10 * torch.log10(pred + self.eps)
        target_db = 10 * torch.log10(target + self.eps)
        loss_energy = torch.mean(torch.abs(pred_db - target_db))
        
        loss_energy
        return loss_energy


class loss_fn(nn.Module):
    def __init__(self, sr, 
        duration, 
        device, 
        env_duration = 0.4, 
        smooth = "gaussian", 
        env_window = 512, 
        env_loss_alpha = 0.5,
        ):
        super(loss_fn, self).__init__()
        self.sr = sr
        self.duration = duration
        self.sample_length = int(duration * sr)
        self.device = device
        self.smooth = smooth
        self.env_duration = env_duration
        self.env_window = env_window
        self.env_loss_alpha = env_loss_alpha
        self.env_loss = Envelope_Loss(alpha=env_loss_alpha, eps=1e-6)
        self.mrstft_high_q_fn = auraloss.freq.MultiResolutionSTFTLoss(
                fft_sizes = [32, 64, 128, 256, 512],
                hop_sizes = [16, 32, 64, 128, 256],
                win_lengths = [32, 64, 128, 256, 512],
                scale=None,
                n_bins=None,
                sample_rate=self.sr,
                perceptual_weighting=False,
                device=self.device,
            )

    def forward(self, pred_ir, tgt_ir, pred_env = None, env = False):
        #unify the length of the irs
        pred_ir = self._cut_or_zero_padding(pred_ir, self.sample_length)
        tgt_ir = self._cut_or_zero_padding(tgt_ir, self.sample_length)

        # """waveform loss"""
        # wave_loss = F.mse_loss(pred_ir, tgt_ir)
        """mrstft loss"""
        mrstft_loss = self.mrstft_high_q_fn(pred_ir, tgt_ir)

        """time Energy decay curveloss"""
        #my own energy decay loss
        pred_energy_decay_curve = self._time_energy_decay_curve(pred_ir, self.sr, 1024, 256)
        tgt_energy_decay_curve = self._time_energy_decay_curve(tgt_ir, self.sr, 1024, 256)
        time_edc_loss = F.l1_loss(pred_energy_decay_curve, tgt_energy_decay_curve)

        """envelope loss"""
        env_loss = 0.0
        if env:
            gt_env = self.get_envelope(tgt_ir[:,0])
            if pred_env is None:
                pred_env = self.get_envelope(pred_ir[:,0])
            else:
                pred_env = pred_env[:,0]
            env_loss = self.get_env_loss(pred_env, gt_env)
        
        return mrstft_loss, time_edc_loss, env_loss

    def _cut_or_zero_padding(self, ir, end):
        if ir.shape[2] > end:
            ir = ir[:,:,:end]
        elif ir.shape[2] < end:
            zeros = torch.zeros((ir.shape[0], ir.shape[1], end - ir.shape[2]), device=ir.device, dtype=ir.dtype)
            ir = torch.cat([ir, zeros], dim=2)
        return ir

    def _time_energy_decay_curve(self, ir, sr, nfft, hop_length):
        ir = ir.squeeze(1)
        power = ir ** 2
        energy = torch.flip(torch.cumsum(torch.flip(power, [-1]), dim=-1), [-1])
        energy_db = 10 * torch.log10(energy + 1.0e-7)
        energy_db = energy_db - energy_db[:,:1]
        # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        # ax.plot(energy_db[0,:])
        # plt.savefig(f"../../assets/energy_decay_curve.pdf")
        # plt.close()
        return energy_db
    
    def get_envelope(self, signal):
        """
        get the energy envelope of the signal 
        
        Args:
        - signal: Tensor, shape of (batch, samples) or (samples,)
        """
        if not signal.is_cuda:
            signal = signal.to("cuda")
        
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)  # (1, samples)

        # --- (Hilbert Transform) ---
        N = signal.shape[-1]
        Xf = torch.fft.fft(signal)
        h = torch.zeros(N, device=signal.device)
        
        if N % 2 == 0:
            h[0] = h[N // 2] = 1
            h[1:N // 2] = 2
        else:
            h[0] = 1
            h[1:(N + 1) // 2] = 2
            
        Xf = Xf * h
        z = torch.fft.ifft(Xf)
        envelope = z.real[..., :N]**2 + z.imag[..., :N]**2
        return envelope  

    def smooth_envelope_mv(self, envelope, window_size=512):
        # --- (Moving Average) ---
        N = envelope.shape[-1]
        kernel = torch.ones((1, 1, window_size), device=envelope.device) / window_size
        pad_size = window_size // 2
        envelope_padded = F.pad(envelope.unsqueeze(1), (pad_size, pad_size), mode='constant', value=0)
        smoothed_envelope = F.conv1d(envelope_padded, kernel)

        return smoothed_envelope[:, 0, :N]
    
    def smooth_envelope_gaussian(self, envelope, window_size=512, sigma=None):
        # sigma = window_size / 6 by default
        if sigma is None:
            sigma = window_size / 6.0
            
        device = envelope.device
        N = envelope.shape[-1]
        
        # --- gaussian smoothing ---
        # create a sequence of coordinates from -(window_size-1)/2 to (window_size-1)/2
        x = torch.arange(window_size, device=device).float() - (window_size - 1) / 2.0
        kernel = torch.exp(-x.pow(2) / (2 * sigma**2))
        kernel = kernel / kernel.sum() 
        kernel = kernel.view(1, 1, -1)
        pad_size = window_size // 2
        envelope_padded = F.pad(envelope.unsqueeze(1), (pad_size, pad_size), mode='reflect')
        
        smoothed_envelope = F.conv1d(envelope_padded, kernel)
        smoothed_envelope = smoothed_envelope[:, 0, :N] 


        return smoothed_envelope

    def get_env_loss(self, pred_env, gt_env):
        if self.smooth == "gaussian":
            gt_env = self.smooth_envelope_gaussian(gt_env, 
                window_size=self.env_window)
        elif self.smooth == "mv":
            gt_env = self.smooth_envelope_mv(gt_env, 
                window_size=self.env_window)
        elif self.smooth == None:
            pass
        else:
            raise ValueError(f"Invalid smooth method: {self.smooth}")
        env_length = int(self.env_duration * self.sr)
        pred_env = pred_env[:, :env_length]
        gt_env = gt_env[:, :env_length]
        env_loss = self.env_loss(pred_env, gt_env)
        return env_loss

if __name__ == "__main__":
    ws = 256
    wave_loss = loss_fn(sr=16000, duration=0.4, device="cuda", smooth = "gaussian", env_window = ws, env_loss_alpha = 0.99)
    gt_ir_path = '/data/250010171/code/EigeNet_discriminant/data/debug/train_debug_gt_ir.wav'
    pred_ir_path = '/data/250010171/code/EigeNet_discriminant/data/debug/train_debug_pred_ir.wav'
    pred_env_path = '/data/250010171/code/EigeNet_discriminant/data/debug/train_debug_pred_env.wav'
    gt_ir = librosa.load(gt_ir_path, sr=16000)[0][:8000]
    pred_ir = librosa.load(pred_ir_path, sr=16000)[0][:8000]
    pred_env = librosa.load(pred_env_path, sr=16000)[0][:8000]
    gt_ir = torch.from_numpy(gt_ir).unsqueeze(0).cuda().float()
    pred_ir = torch.from_numpy(pred_ir).unsqueeze(0).cuda().float()
    pred_env = torch.from_numpy(pred_env).unsqueeze(0).cuda().float()
    gt_env = wave_loss.smooth_envelope_gaussian(wave_loss.get_envelope(gt_ir), window_size=ws).cpu().numpy()
    gt_ir = gt_ir.unsqueeze(1)
    print(f"gt_ir: {gt_ir.shape}")

    env_root = '/data/250010171/code/EigeNet_discriminant/data/debug'
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(10, 10))
    plt.plot(gt_env[0], linewidth=3)
    plt.axis('off')
    plt.savefig(os.path.join(env_root, 'env_visualization_3.pdf'))
    plt.close()
