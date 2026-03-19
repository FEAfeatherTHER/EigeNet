import torch
import torch.nn.functional as F
import librosa
import auraloss 
import torch.nn as nn
import numpy as np
def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (Tensor): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    window = window.to(x.device)
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=True, pad_mode='constant')
    real = torch.real(x_stft)
    imag = torch.imag(x_stft)

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)

def convert_ir_to_spec(ir):
    tgt_spec = []
    for i in range(ir.shape[0]):
        tgt_spec.append(stft(
            ir[i],
            fft_size=124,
            hop_size=31,
            win_length=62,
            window=torch.hann_window(62)
        ).permute(0, 2, 1).unsqueeze(0))
    return torch.vstack(tgt_spec)

import torch
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

        # --- Pearson Correlation Loss ---
        # centering
        pred_mean = torch.mean(pred, dim=1, keepdim=True)
        target_mean = torch.mean(target, dim=1, keepdim=True)
        pred_centered = pred - pred_mean
        target_centered = target - target_mean

        # cosine similarity
        num = torch.sum(pred_centered * target_centered, dim=1)
        denom = torch.sqrt(torch.sum(pred_centered**2, dim=1) * torch.sum(target_centered**2, dim=1) + self.eps)
        pearson_corr = num / denom
        loss_trend = torch.mean(1 - pearson_corr)

        # --- Log-MSE Loss ---
        # in log space, handle magnitude difference
        log_pred = torch.log(pred + self.eps)
        log_target = torch.log(target + self.eps)
        loss_magnitude = torch.mean((log_pred - log_target) ** 2)
        
        # --- weighted loss ---
        total_loss = self.alpha * loss_trend + (1 - self.alpha) * loss_magnitude
        return total_loss


class loss_fn(nn.Module):
    def __init__(self, sr, 
        duration, 
        device, 
        env_duration = 0.4, 
        smooth = "gaussian", 
        env_window = 512, 
        env_normalized = False,
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
        self.env_normalized = env_normalized
        self.env_loss_alpha = env_loss_alpha
        self.env_loss = Envelope_Loss(env_loss_alpha)

    def forward(self, pred_ir, tgt_ir, pred_env = None):
        #unify the length of the irs
        pred_ir = self._cut_or_zero_padding(pred_ir, self.sample_length)
        tgt_ir = self._cut_or_zero_padding(tgt_ir, self.sample_length)
        # """waveform loss"""
        # wave_loss = F.mse_loss(pred_ir, tgt_ir)
        """mrstft loss"""
        #hi_q_temporal
        mrstft_high_q_fn = auraloss.freq.MultiResolutionSTFTLoss(
                fft_sizes = [32, 64, 128, 256, 512],
                hop_sizes = [16, 32, 64, 128, 256],
                win_lengths = [32, 64, 128, 256, 512],
                scale=None,
                n_bins=None,
                sample_rate=self.sr,
                perceptual_weighting=False,
                device=self.device,
            )
        #not hi_q_temporal
        mrstft_low_q_fn = auraloss.freq.MultiResolutionSTFTLoss(
                fft_sizes = [256, 512, 1024], 
                hop_sizes = [64, 128, 256],
                win_lengths = [256, 512, 1024],
                scale="mel",
                n_bins=36,
                sample_rate=self.sr,
                perceptual_weighting=True,
                device=self.device,
            )
        mrstft_loss = 1*mrstft_high_q_fn(pred_ir, tgt_ir)
        + 0*mrstft_low_q_fn(pred_ir, tgt_ir)

        """time Energy decay curveloss"""
        #my own energy decay loss
        energy_loss_fn = nn.L1Loss()
        pred_energy_decay_curve = self._time_energy_decay_curve(pred_ir, self.sr, 1024, 256)
        tgt_energy_decay_curve = self._time_energy_decay_curve(tgt_ir, self.sr, 1024, 256)
        time_edc_loss = energy_loss_fn(pred_energy_decay_curve, tgt_energy_decay_curve)

        """spectrogram Energy decay curveloss"""
        #xRIR edc loss
        pred_spect = convert_ir_to_spec(pred_ir).permute(0, 2, 3, 1)
        gt_spect = convert_ir_to_spec(tgt_ir).permute(0, 2, 3, 1)
        spect_edc_loss = self.compute_spect_energy_decay_losses(gts=torch.exp(gt_spect) - 1e-8, preds=torch.exp(pred_spect) - 1e-8)
        
        """envelope loss"""
        if pred_env is not None:
            if pred_env.shape[1] == 1:
                pred_env = pred_env.squeeze(1)
            pred_env = torch.abs(pred_env) / (torch.max(torch.abs(pred_env), dim=-1, keepdim=True)[0] + 1e-7)
        else:
            pred_env = self.get_envelope(pred_ir[:,0])
        gt_env = self.get_envelope(tgt_ir[:,0])
        env_loss = self.get_env_loss(pred_env, gt_env)
        
        return mrstft_loss, time_edc_loss, spect_edc_loss, env_loss

    def _cut_or_zero_padding(self, ir, end):
        if ir.shape[2] > end:
            ir = ir[:,:,:end]
        else:
            zeros = torch.zeros(ir.shape[0], ir.shape[1], end - ir.shape[2])
            zeros = zeros.to(ir.device)
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
    
    def compute_spect_energy_decay_losses(self,
                        loss_type="l1_loss",
                        #loss_weight=1.0e-2,
                        gts=None,
                        preds=None,
                        mask=None,
                        slice_till_direct_signal=False,
                        direct_signal_len_in_ms=50,
                        dont_collapse_across_freq_dim=False,
                        sr=16000,
                        hop_length=62,
                        win_length=248,
                        ):
        """
        compute energy decay loss
        :param loss_type: loss type
        :param loss_weight: loss weight
        :param gts: gt IRs
        :param preds: estimated IRs
        :param mask: mask to mark valid entries in batch
        :param slice_till_direct_signal: remove direct signal part of IR
        :param direct_signal_len_in_ms: direct signal length in milliseconds
        :param dont_collapse_across_freq_dim: collapse along frequency dimension of spectrogram (spect.)
        :param sr: sampling rate
        :param hop_length: hop length to compute spect.
        :param win_length: length of temporal window to compute spect.
        :return: energy decay loss
        """
        assert len(gts.size()) == len(preds.size()) == 4
        assert gts.size(-1) in [1, 2]
        assert preds.size(-1) in [1, 2]
        #print(gts.shape, preds.shape)

        slice_idx = None
        if slice_till_direct_signal:
            if direct_signal_len_in_ms == 50:
                if (sr == 16000) and (hop_length == 62) and (win_length == 248):
                    # (62 * 11 + 248 / 2) / 16000 = 0.050375 (50 ms)
                    # so has to use the 12th window (idx = 11).. so slice idx should be 12
                    slice_idx = 12
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

        if slice_till_direct_signal:
            if dont_collapse_across_freq_dim:
                gts_fullBandAmpEnv = gts[:slice_idx]
            else:
                gts_fullBandAmpEnv = torch.sum(gts[:slice_idx], dim=-3)
        else:
            if dont_collapse_across_freq_dim:
                gts_fullBandAmpEnv = gts
            else:
                gts_fullBandAmpEnv = torch.sum(gts**2, dim=-3)
        power_gts_fullBandAmpEnv = gts_fullBandAmpEnv #** 2
        energy_gts_fullBandAmpEnv = torch.flip(torch.cumsum(torch.flip(power_gts_fullBandAmpEnv, [-2]), -2), [-2])
        valid_loss_idxs = ((energy_gts_fullBandAmpEnv != 0.).type(energy_gts_fullBandAmpEnv.dtype))[..., 1:, :]

        db_gts_fullBandAmpEnv = 10 * torch.log10(energy_gts_fullBandAmpEnv + 1.0e-13)
        norm_db_gts_fullBandAmpEnv = db_gts_fullBandAmpEnv - db_gts_fullBandAmpEnv[..., :1, :]
        norm_db_gts_fullBandAmpEnv = norm_db_gts_fullBandAmpEnv[..., 1:, :]
        if slice_till_direct_signal:
            weighted_norm_db_gts_fullBandAmpEnv = norm_db_gts_fullBandAmpEnv
        else:
            weighted_norm_db_gts_fullBandAmpEnv = norm_db_gts_fullBandAmpEnv * valid_loss_idxs

        if slice_till_direct_signal:
            if dont_collapse_across_freq_dim:
                preds_fullBandAmpEnv = preds[:slice_idx]
            else:
                preds_fullBandAmpEnv = torch.sum(preds[:slice_idx], dim=-3)
        else:
            if dont_collapse_across_freq_dim:
                preds_fullBandAmpEnv = preds
            else:
                preds_fullBandAmpEnv = torch.sum(preds**2, dim=-3)
        power_preds_fullBandAmpEnv = preds_fullBandAmpEnv #** 2
        energy_preds_fullBandAmpEnv = torch.flip(torch.cumsum(torch.flip(power_preds_fullBandAmpEnv, [-2]), -2), [-2])
        db_preds_fullBandAmpEnv = 10 * torch.log10(energy_preds_fullBandAmpEnv + 1.0e-13)
        norm_db_preds_fullBandAmpEnv = db_preds_fullBandAmpEnv - db_preds_fullBandAmpEnv[..., :1, :]
        norm_db_preds_fullBandAmpEnv = norm_db_preds_fullBandAmpEnv[..., 1:, :]
        if slice_till_direct_signal:
            weighted_norm_db_preds_fullBandAmpEnv = norm_db_preds_fullBandAmpEnv
        else:
            weighted_norm_db_preds_fullBandAmpEnv = norm_db_preds_fullBandAmpEnv * valid_loss_idxs

        if loss_type == "l1_loss":
            if mask is None:
                loss = F.l1_loss(weighted_norm_db_preds_fullBandAmpEnv, weighted_norm_db_gts_fullBandAmpEnv)
            else:
                # not counting the contribution from masked out locations in the batch
                assert torch.sum(mask) == mask.size(0)
                loss = torch.sum(torch.abs(weighted_norm_db_preds_fullBandAmpEnv - weighted_norm_db_gts_fullBandAmpEnv)) /\
                    (torch.sum(mask) * np.prod(list(weighted_norm_db_preds_fullBandAmpEnv.size())[1:]))
        else:
            raise NotImplementedError

        #loss = loss * loss_weight

        return loss

    def get_envelope(self, signal):
        """
        get the envelope of the signal 
        
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
        envelope = torch.abs(z)[...,:N]
        return envelope  

    def smooth_envelope_mv(self, envelope, window_size=512, normalized = True):
        # --- (Moving Average) ---
        N = envelope.shape[-1]
        kernel = torch.ones((1, 1, window_size), device=envelope.device) / window_size
        pad_size = window_size // 2
        envelope_padded = F.pad(envelope.unsqueeze(1), (pad_size, pad_size), mode='constant', value=0)
        smoothed_envelope = F.conv1d(envelope_padded, kernel)
        
        if normalized:
            env_max_val = torch.max(smoothed_envelope, dim=-1, keepdim=True)[0]
            smoothed_envelope = smoothed_envelope / (env_max_val + 1e-7)

        return smoothed_envelope[:, 0, :N]
    def smooth_envelope_gaussian(self, envelope, window_size=512, sigma=None, normalized=True):
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

        if normalized:
            env_max_val = torch.max(smoothed_envelope, dim=-1, keepdim=True)[0]
            smoothed_envelope = smoothed_envelope / (env_max_val + 1e-7)

        return smoothed_envelope

    def get_env_loss(self, pred_env, gt_env):
        if self.smooth == "gaussian":
            pred_env = self.smooth_envelope_gaussian(pred_env, 
                window_size=self.env_window, 
                normalized=self.env_normalized)
            gt_env = self.smooth_envelope_gaussian(gt_env, 
                window_size=self.env_window, 
                normalized=self.env_normalized)
        elif self.smooth == "mv":
            pred_env = self.smooth_envelope_mv(pred_env, 
                window_size=self.env_window, 
                normalized=self.env_normalized)
            gt_env = self.smooth_envelope_mv(gt_env, 
                window_size=self.env_window, 
                normalized=self.env_normalized)
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
    wave_loss = loss_fn(sr=16000, duration=0.4, device="cuda", smooth = "gaussian", env_window = 32, env_normalized = False)
    gt_ir_path = '/data/250010171/code/EigeNet_discriminant/data/debug/S005_R070_hybrid_IR_16000hz.wav'
    pred_ir_path = '/data/250010171/code/EigeNet_discriminant/data/debug/S005_R070_hybrid_IR_16000hz_recon.wav'
    gt_ir = librosa.load(gt_ir_path, sr=16000)[0][:8000]
    pred_ir = librosa.load(pred_ir_path, sr=16000)[0][:8000]
    gt_ir = torch.from_numpy(gt_ir).unsqueeze(0).unsqueeze(0).cuda()
    pred_ir = torch.from_numpy(pred_ir).unsqueeze(0).unsqueeze(0).cuda()
    mrstft_loss, time_edc_loss, spect_edc_loss, env_loss = wave_loss(pred_ir, gt_ir)
    print(f"mrstft_loss: {mrstft_loss}")
    print(f"time_edc_loss: {time_edc_loss}")
    print(f"spect_edc_loss: {spect_edc_loss}")
    print(f"env_loss: {env_loss}")