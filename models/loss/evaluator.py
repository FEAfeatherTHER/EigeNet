import numpy as np
from scipy.interpolate import interp1d
import torchaudio
from scipy.signal import hilbert
import torch

def apply_delay(signal, delay_tensor):
    """
    Apply integer delay to each sample in the batch with zero-padding.
    
    Args:
    - signal: Tensor of shape [batch_size, sequence_length, channels] or [batch_size, sequence_length].
    - delay_tensor: Tensor of shape [batch_size] with integer delay values for each sample.
    
    Returns:
    - Delayed signal with the same shape as input signal.
    """
    batch_size, sequence_length = signal.shape[:2]
    
    # Output tensor initialized with zeros (same shape as input signal)
    delayed_signal = torch.zeros_like(signal)
    
    for i in range(batch_size):
        delay = delay_tensor[i].item()  # Get the delay value for this batch item
        
        if delay > 0:
            # Positive delay: shift right and pad with zeros at the beginning
            delayed_signal[i, delay:] = signal[i, :-delay]
        elif delay < 0:
            # Negative delay: shift left and pad with zeros at the end
            delayed_signal[i, :delay] = signal[i, -delay:]
        else:
            # No delay, just copy the signal
            delayed_signal[i] = signal[i]
    
    return delayed_signal

def griffin_lim(spec, n_fft=124, win_length=62, hop_length=31, power=1, n_iter=32):
    transform = torchaudio.transforms.GriffinLim(
        n_fft=n_fft, n_iter=n_iter,win_length=win_length,hop_length=hop_length,power=power
    )
    return transform(spec)

def normalize(audio, norm='peak'):
    """
    normalize IR
    :param audio: IR
    :param norm: normalization mode
    :return: normalized IR
    """
    if norm == 'peak':
        peak = abs(audio).max()
        if peak != 0:
            return audio / peak
        else:
            return audio
    elif norm == 'rms':
        if torch.is_tensor(audio):
            audio = audio.numpy()
        audio_without_padding = np.trim_zeros(audio, trim='b')
        rms = np.sqrt(np.mean(np.square(audio_without_padding))) * 100
        if rms != 0:
            return audio / rms
        else:
            return audio
    else:
        raise NotImplementedError

def measure_drr_energy_ratio(y, cutoff_time=0.003, fs=16000):
    """
    get direct to reverberant energy ratio (DRR)
    :param y: IR
    :param cutoff_time: cutoff time to compute DRR
    :param fs: sampling frequency
    :return: DRR
    """
    direct_sound_idx = int(cutoff_time * fs)

    # removing leading silence
    y = normalize(y)
    y = np.trim_zeros(y, trim='fb')

    # everything up to the given idx is summed up and treated as direct sound energy
    y = np.power(y, 2)
    direct = sum(y[:direct_sound_idx + 1])
    reverberant = sum(y[direct_sound_idx + 1:])
    if direct == 0 or reverberant == 0:
        drr = 1
    else:
        drr = 10 * np.log10(direct / reverberant)

    return drr

def calculate_drr_diff(gt, est, cutoff_time=0.003, fs=16000, compute_relative_diff=False, get_diff_val=True,
                       get_gt_val=False, get_pred_val=False,):
    """
    get difference in DRR, DRR for gt or DRR for estimated IR
    :param gt: gt IR
    :param est: estimated IR
    :param cutoff_time: cutoff time to compute DRR
    :param fs: sampling frequency
    :param compute_relative_diff: flag to compute relative difference
    :param get_diff_val: flag to get difference in DRR
    :param get_gt_val: flag to get DRR of gt IR
    :param get_pred_val: flag to get DRR of estimated IR
    :return: difference in DRR, DRR of gt IR or DRR of estimated IR
    """
    drr_gt = measure_drr_energy_ratio(gt, cutoff_time=cutoff_time, fs=fs)
    drr_est = measure_drr_energy_ratio(est, cutoff_time=cutoff_time, fs=fs)
    diff = abs(drr_gt - drr_est)
    if compute_relative_diff:
        diff = abs(diff / drr_gt)

    if get_diff_val:
        return diff
    elif get_gt_val:
        return drr_gt
    elif get_pred_val:
        return drr_est

class Evaluator:
    def __init__(self, cfg=None, seq_name=None, sample_rate=16000, signal_length=8000):
        if sample_rate != 16000:
            raise ValueError("The signal metric protocol requires 16000 Hz")
        self.protocol = _protocol(signal_length)
        self.signal_length = self.protocol['signal_length']
        self.mse = []
        self.psnr = []
        self.ssim = []
        self.cfg = cfg
        self.seq_name = seq_name
        self.t60_error = []
        self.clarity_error = []
        self.edt_error = []
        self.spec_mse = []

        self.invalid = 0

    def evaluate_batch(self, gt, pred):
        return self.evaluate_detailed(gt, pred)['metrics']

    def evaluate_new_metrics(self, gt, pred):
        return _evaluate_new_metrics(self, gt, pred)

    def evaluate_detailed(self, gt, pred):
        return _evaluate_detailed(self, gt, pred)

    def psnr_metric(self, img_pred, img_gt):
        mse = np.mean((img_pred - img_gt)**2)
        psnr = -10 * np.log(mse) / np.log(10)
        return psnr

    def stft_loss(self, img_pred, img_gt):
        mse = np.mean((img_pred - img_gt)**2)
        return mse

    def env_loss(self, pred_wav, gt_wav):
        pred_env = np.abs(hilbert(pred_wav))
        gt_env = np.abs(hilbert(gt_wav))
        envelope_distance = np.mean(np.abs(gt_env - pred_env) / np.max(gt_env)) * 100.
        return float(envelope_distance)
    
    def t60_impulse(self, energy, rt='t20', fs=16000, trans=False, trans1=False):
        rt = rt.lower()
        if rt == 't30':
            init = -5.0
            end = -35.0
            factor = 2.0
        elif rt == 't20':
            init = -5.0
            end = -25.0
            factor = 3.0
        elif rt == 't10':
            init = -5.0
            end = -15.0
            factor = 6.0
        elif rt == 'edt':
            init = 0.0
            end = -10.0
            factor = 6.0

        if trans:
            result = energy[0][..., None]
            new_energy = []
            for band in range(result.shape[1]):
                # Filtering signal
                filtered_signal = result[:, band]
                abs_signal = np.abs(filtered_signal)
                # Schroeder integration
                sch = abs_signal**2
                new_energy.append(sch)
            energy = np.array(new_energy)
        elif trans1:
            abs_signal = np.abs(energy[0])
            energy = np.array(abs_signal**2).reshape(1, -1, 22).mean(-1)
            factor *= 1
            trans = False
        else:
            factor *= 1

        t60 = np.zeros(energy.shape[0])
        c50 = np.zeros(energy.shape[0])
        schdb_list = []
        for band in range(energy.shape[0]):
            if trans:
                pow_energy = energy[band]
            else:
                if not trans1:
                    pow_energy = np.power(10, energy[band])
                else:
                    pow_energy = energy[band]
                x_dense = np.arange(0, pow_energy.shape[-1]*22)
                x_bin = np.arange(0, pow_energy.shape[-1]*22, 22)
                f = interp1d(x_bin, pow_energy, kind = 'slinear')
                pow_energy = f(x_dense[:len(x_bin)*22-22])
            sch = np.cumsum(pow_energy[::-1])[::-1]
            sch_db = 10.0 * np.log10(sch / np.max(sch))
            sch_db -= sch_db[0]
            if band == 0:
                out_sch_db = sch_db
            schdb_list.append(sch_db)

            init_sample = np.min(np.where(-5 - sch_db > 0)[0])
            if len(np.where(-35 - sch_db> 0)[0]) > 0:
                end_sample = np.min(np.where(-35 - sch_db> 0)[0])
            else:
                end_sample = len(sch_db) - 1
            t60[band] = factor * (end_sample / fs - init_sample / fs)
            if trans1 or not trans:
                t = int((50 / 1000.0) * fs + 1)
            else:
                t = int((50 / 1000.0) * fs + 1)
            c50[band] = 10.0 * np.log10((np.sum(pow_energy[:t]) / np.sum(pow_energy[t:])))
        return t60, out_sch_db, c50, init_sample
    
    def measure_edt(self, h, fs=16000, decay_db=10):
        h = np.array(h)
        fs = float(fs)

        # The power of the impulse response in dB
        power = h ** 2
        energy = np.cumsum(power[::-1])[::-1]  # Integration according to Schroeder

        # remove the possibly all zero tail
        i_nz = np.max(np.where(energy > 0)[0])
        energy = energy[:i_nz]
        energy_db = 10 * np.log10(energy)
        energy_db -= energy_db[0]

        try:
            i_decay = np.min(np.where(- decay_db - energy_db > 0)[0])
        except ValueError:
            return np.inf
        t_decay = i_decay / fs
        decay_time = t_decay
        est_edt = (60 / decay_db) * decay_time
        return est_edt

    def measure_rt60(self, h, fs=16000, decay_db=20, plot=False, rt60_tgt=None):
        """
        Analyze the RT60 of an impulse response. Optionaly plots some useful information.
        Parameters
        ----------
        h: array_like
            The impulse response.
        fs: float or int, optional
            The sampling frequency of h (default to 1, i.e., samples).
        decay_db: float or int, optional
            The decay in decibels for which we actually estimate the time. Although
            we would like to estimate the RT60, it might not be practical. Instead,
            we measure the RT20 or RT30 and extrapolate to RT60.
        plot: bool, optional
            If set to ``True``, the power decay and different estimated values will
            be plotted (default False).
        rt60_tgt: float
            This parameter can be used to indicate a target RT60 to which we want
            to compare the estimated value.
        """

        h = np.array(h)
        fs = float(fs)

        # The power of the impulse response in dB
        power = h ** 2
        energy = np.cumsum(power[::-1])[::-1]  # Integration according to Schroeder

        # remove the possibly all zero tail
        i_nz = np.max(np.where(energy > 0)[0])
        energy = energy[:i_nz]
        energy_db = 10 * np.log10(energy)
        energy_db -= energy_db[0]
        # -5 dB headroom
        try:
            i_5db = np.min(np.where(-5 - energy_db > 0)[0])
        except ValueError:
            return np.inf
        e_5db = energy_db[i_5db]
        t_5db = i_5db / fs

        try:
            i_decay = np.min(np.where(-5 - decay_db - energy_db > 0)[0])
        except ValueError:
            return np.inf
        
        t_decay = i_decay / fs

        # compute the decay time
        decay_time = t_decay - t_5db
        est_rt60 = (60 / decay_db) * decay_time

        return est_rt60
    
    def measure_clarity(self, signal, time=50, fs=16000):
        h2 = signal**2
        t = int((time/1000)*fs + 1)
        return 10*np.log10(np.sum(h2[:t])/np.sum(h2[t:]))
    
    def compute_energy_db(self, h):
        h = np.array(h)
        # The power of the impulse response in dB
        power = h ** 2
        energy = np.cumsum(power[::-1])[::-1]  # Integration according to Schroeder

        # remove the possibly all zero tail
        i_nz = np.max(np.where(energy > 0)[0])
        energy = energy[:i_nz]
        energy_db = 10 * np.log10(energy)
        energy_db -= energy_db[0]
        return  energy_db
    
    def evaluate_edt(self, pred_ir, gt_ir):
        np_pred_ir = pred_ir
        np_gt_ir = gt_ir
        pred_edt = self.measure_edt(np_pred_ir)
        gt_edt = self.measure_edt(np_gt_ir)
        edt_error = abs(pred_edt - gt_edt)
        self.edt_error.append(edt_error)

    def evaluate_clarity(self, pred_ir, gt_ir):
        np_pred_ir = pred_ir
        np_gt_ir = gt_ir
        pred_clarity = self.measure_clarity(np_pred_ir)
        gt_clarity = self.measure_clarity(np_gt_ir)
        clarity_error = abs(pred_clarity - gt_clarity)
        self.clarity_error.append(clarity_error)

    def evaluate_t60(self, pred_ir, gt_ir):
        np_pred_ir = pred_ir
        np_gt_ir = gt_ir
        mse = np.mean((np_pred_ir - np_gt_ir) ** 2)
        self.mse.append(mse)
        psnr = self.psnr_metric(np_pred_ir, np_gt_ir)
        self.psnr.append(psnr)
        try:
            pred_t60 = self.measure_rt60(np_pred_ir)
            gt_t60 = self.measure_rt60(np_gt_ir)
            t60_error = abs(pred_t60 - gt_t60) / gt_t60
            self.t60_error.append(t60_error)
        except:
            self.invalid += 1
    
    def evaluate_energy_db(self, pred_ir, gt_ir):
        pred_db = self.compute_energy_db(pred_ir)
        gt_db = self.compute_energy_db(gt_ir)
        return pred_db, gt_db
    
    def evaluate_spec_mse(self, pred_ir_spec, gt_ir_spec):
        self.spec_mse.append(np.mean(pred_ir_spec - gt_ir_spec)**2)
   


METRIC_NAMES = (
    'waveform_nmse', 'mr_log_stft_l1', 'edt_error', 'c50_error', 't60_error',
    'phase_error_deg', 'early_energy_db_mae',
)
NEW_METRIC_NAMES = ('phase_error_deg', 'early_energy_db_mae')
DIAGNOSTIC_NAMES = (
    'phase_gt_energy_coverage', 'phase_missing_frequency_count',
    'phase_valid_frequency_count', 'early_start_sample', 'early_window_samples',
)
METRIC_PROTOCOL = {
    'version': 'eigenet_seven_metrics_v2',
    'sample_rate': 16000,
    'signal_length': 8000,
    'acoustic_length': 5808,
    'metrics': list(METRIC_NAMES),
    'stft_fft_sizes': [512, 1024, 2048],
    'stft_hop_fraction': 0.25,
    'stft_window': 'periodic_hann',
    'stft_center': True,
    'stft_pad_mode': 'reflect',
    'stft_normalized': False,
    'stft_onesided': True,
    'stft_log': 'natural',
    'stft_magnitude_floor': 0.0001,
    'waveform_definition': 'Squared error sum / ground-truth energy sum over signal_length',
    'phase_fft': 'numpy.fft.rfft, full signal_length, no window or padding',
    'phase_valid_frequency': 'Both ground-truth and prediction magnitude nonzero',
    'phase_weight': 'Ground-truth single-sided spectral energy; no interior-bin doubling',
    'phase_difference': 'abs(atan2(sin(pred_phase-gt_phase), cos(pred_phase-gt_phase)))',
    'phase_unit': 'degrees',
    'early_window': 'numpy.hanning(64), sum normalized',
    'early_slice_start': 31,
    'early_length': 800,
    'early_onset_fraction': 0.001,
    'early_onset': 'First original GT squared-energy cumulative sample >= 0.001 of total',
    'early_smoothing': 'Square original waveform, full convolution, slice [31:31+N]',
    'early_reference': 'Maximum smoothed GT energy over full signal',
    'early_floor_fraction': 1e-6,
    'early_definition': 'Mean absolute dB energy difference in shared GT-onset window',
    'early_unit': 'dB',
    'aggregation': 'Equal-weight valid samples for every metric; float64 sums',
    'acoustic_units': {'edt_error': 'seconds', 'c50_error': 'dB', 't60_error': 'percent'},
}


def _protocol(signal_length):
    from copy import deepcopy
    if (isinstance(signal_length, (bool, np.bool_)) or
            not isinstance(signal_length, (int, np.integer)) or signal_length < 5808):
        raise ValueError('signal_length must be an integer >= 5808')
    result = deepcopy(METRIC_PROTOCOL)
    result['signal_length'] = int(signal_length)
    return result


def _convert_waveforms(value, signal_length):
    if torch.is_tensor(value):
        if value.device.type != 'cpu':
            raise ValueError('Inputs must be CPU tensors')
        value = value.detach().numpy()
    value = np.asarray(value)
    if np.iscomplexobj(value):
        raise ValueError('Inputs must be real waveforms')
    value = np.asarray(value, dtype=np.float64)
    if value.ndim == 3 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 2 or value.shape[1] < signal_length:
        raise ValueError(f'Expected mono [B,T] or [B,1,T] with T >= {signal_length}')
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite waveform input')
    return value[:, :signal_length]


def _checked_pair(self, gt, pred):
    gt = _convert_waveforms(gt, self.signal_length)
    pred = _convert_waveforms(pred, self.signal_length)
    if gt.shape != pred.shape:
        raise ValueError('Batch shapes must match')
    return gt, pred


def _checked_ratio(error, reference):
    numerator = np.sum(np.abs(error) ** 2, axis=-1, dtype=np.float64)
    denominator = np.sum(np.abs(reference) ** 2, axis=-1, dtype=np.float64)
    result = np.full(denominator.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result


def _new_metrics(gt, pred):
    """Inputs have already been checked/cropped; caller guards arithmetic."""
    batch_size, length = gt.shape
    metrics = {name: np.full(batch_size, np.nan, dtype=np.float64) for name in NEW_METRIC_NAMES}
    reasons = {name: [''] * batch_size for name in NEW_METRIC_NAMES}
    diagnostics = {
        'phase_gt_energy_coverage': np.full(batch_size, np.nan, dtype=np.float64),
        'phase_missing_frequency_count': np.zeros(batch_size, dtype=np.int64),
        'phase_valid_frequency_count': np.zeros(batch_size, dtype=np.int64),
        'early_start_sample': np.full(batch_size, -1, dtype=np.int64),
        'early_window_samples': np.zeros(batch_size, dtype=np.int64),
    }
    gt_fft, pred_fft = np.fft.rfft(gt, axis=-1), np.fft.rfft(pred, axis=-1)
    gt_magnitude, pred_magnitude = np.abs(gt_fft), np.abs(pred_fft)
    weights = gt_magnitude ** 2
    kernel = np.hanning(64)
    kernel /= kernel.sum(dtype=np.float64)
    for i in range(batch_size):
        gt_nonzero = gt_magnitude[i] != 0
        valid = gt_nonzero & (pred_magnitude[i] != 0)
        diagnostics['phase_valid_frequency_count'][i] = np.count_nonzero(valid)
        diagnostics['phase_missing_frequency_count'][i] = np.count_nonzero(gt_nonzero & ~valid)
        total = weights[i].sum(dtype=np.float64)
        valid_energy = weights[i, valid].sum(dtype=np.float64)
        if total == 0:
            reasons['phase_error_deg'][i] = 'ground_truth_energy_zero'
        else:
            diagnostics['phase_gt_energy_coverage'][i] = valid_energy / total
            if valid_energy == 0:
                reasons['phase_error_deg'][i] = 'no_valid_phase_frequency'
            else:
                # angle(0) is never used as a phase observation.
                difference = np.angle(pred_fft[i, valid]) - np.angle(gt_fft[i, valid])
                difference = np.abs(np.arctan2(np.sin(difference), np.cos(difference)))
                metrics['phase_error_deg'][i] = (
                    np.sum(weights[i, valid] * difference, dtype=np.float64) / valid_energy * 180 / np.pi
                )
        gt_energy, pred_energy = gt[i] ** 2, pred[i] ** 2
        cumulative = np.cumsum(gt_energy, dtype=np.float64)
        if cumulative[-1] == 0:
            reasons['early_energy_db_mae'][i] = 'ground_truth_energy_zero'
            continue
        onset = int(np.searchsorted(cumulative, 0.001 * cumulative[-1], side='left'))
        stop = min(onset + 800, length)
        diagnostics['early_start_sample'][i] = onset
        diagnostics['early_window_samples'][i] = stop - onset
        smooth_gt = np.convolve(gt_energy, kernel, 'full')[31:31 + length]
        smooth_pred = np.convolve(pred_energy, kernel, 'full')[31:31 + length]
        reference = np.max(smooth_gt)
        floor = 1e-6 * reference
        log_gt = 10 * np.log10(np.maximum(smooth_gt[onset:stop], floor) / reference)
        log_pred = 10 * np.log10(np.maximum(smooth_pred[onset:stop], floor) / reference)
        metrics['early_energy_db_mae'][i] = np.mean(np.abs(log_pred - log_gt), dtype=np.float64)
    return {'metrics': metrics, 'invalid_reasons': reasons, 'diagnostics': diagnostics}


def _evaluate_new_metrics(self, gt, pred):
    gt, pred = _checked_pair(self, gt, pred)
    try:
        with np.errstate(over='raise', under='raise', invalid='raise', divide='raise'):
            return _new_metrics(gt, pred)
    except FloatingPointError as exc:
        raise ValueError('Signal metric arithmetic overflow, underflow or invalid operation') from exc


def _evaluate_detailed(self, gt, pred):
    gt, pred = _checked_pair(self, gt, pred)
    try:
        with np.errstate(over='raise', under='raise', invalid='raise', divide='raise'):
            detailed = _new_metrics(gt, pred)
            result, reasons = detailed['metrics'], detailed['invalid_reasons']
            result['waveform_nmse'] = _checked_ratio(pred - gt, gt)
            reasons['waveform_nmse'] = [
                'ground_truth_energy_zero' if np.isnan(value) else ''
                for value in result['waveform_nmse']
            ]
            losses = []
            for size in (512, 1024, 2048):
                window = torch.hann_window(size, periodic=True, dtype=torch.float64)

                def spectrum(batch):
                    if not len(batch):
                        return torch.empty((0, size // 2 + 1, 1), dtype=torch.float64)
                    return torch.stft(
                        torch.from_numpy(batch.copy()), n_fft=size, hop_length=size // 4,
                        win_length=size, window=window, center=True, pad_mode='reflect',
                        normalized=False, onesided=True, return_complex=True,
                    ).abs().clamp_min(0.0001).log()
                loss = (spectrum(pred) - spectrum(gt)).abs().mean(dim=(1, 2)).numpy()
                if not np.isfinite(loss).all():
                    raise FloatingPointError('Nonfinite STFT arithmetic')
                losses.append(loss)
            result['mr_log_stft_l1'] = np.mean(losses, axis=0, dtype=np.float64)
            reasons['mr_log_stft_l1'] = [''] * len(gt)
    except FloatingPointError as exc:
        raise ValueError('Signal metric arithmetic overflow, underflow or invalid operation') from exc
    for name, function in (
        ('edt_error', self.measure_edt), ('c50_error', self.measure_clarity),
        ('t60_error', self.measure_rt60),
    ):
        errors = np.full(len(gt), np.nan, dtype=np.float64)
        reasons[name] = [''] * len(gt)
        for i, (reference, estimate) in enumerate(zip(gt, pred)):
            values = []
            for role, waveform in (('ground_truth', reference), ('prediction', estimate)):
                try:
                    with np.errstate(all='ignore'):
                        value = function(waveform[:5808], fs=16000)
                except (ValueError, IndexError, ZeroDivisionError) as exc:
                    reasons[name][i] = f'{role}_undefined_{type(exc).__name__}'
                    break
                if not np.isfinite(value):
                    reasons[name][i] = f'{role}_measurement_nonfinite'
                    break
                values.append(value)
            if len(values) != 2:
                continue
            reference_value, estimate_value = values
            if name == 't60_error' and reference_value <= 0:
                reasons[name][i] = 'ground_truth_t60_nonpositive'
                continue
            try:
                with np.errstate(over='raise', under='raise', invalid='raise', divide='raise'):
                    error = abs(estimate_value - reference_value)
                    if name == 't60_error':
                        error = error / reference_value * 100.0
                if not np.isfinite(error):
                    raise FloatingPointError('Nonfinite acoustic error')
            except FloatingPointError as exc:
                raise ValueError('Acoustic error arithmetic overflow, underflow or invalid operation') from exc
            errors[i] = error
        result[name] = errors
    detailed['metrics'] = {name: result[name] for name in METRIC_NAMES}
    detailed['invalid_reasons'] = {name: reasons[name] for name in METRIC_NAMES}
    return detailed


class MetricAccumulator:
    """Mergeable v2 scalar metrics; historical protocols are intentionally rejected."""
    def __init__(self, signal_length=8000):
        self.protocol = _protocol(signal_length)
        self.signal_length = self.protocol['signal_length']
        self.samples_evaluated = 0
        self.skipped_no_reference = 0
        self.reference_count_histogram = {}
        self.sums = {name: np.zeros((), dtype=np.float64) for name in METRIC_NAMES}
        self.counts = {name: np.zeros((), dtype=np.int64) for name in METRIC_NAMES}
        self.invalid = {name: np.zeros((), dtype=np.int64) for name in METRIC_NAMES}
        self.invalid_reasons = {name: {} for name in METRIC_NAMES}
        self.diagnostic_sums = {name: np.float64(0) for name in DIAGNOSTIC_NAMES}
        self.diagnostic_counts = {name: 0 for name in DIAGNOSTIC_NAMES}
        self.diagnostic_min = {name: None for name in DIAGNOSTIC_NAMES}
        self.phase_missing_frequency_total = 0
        self.phase_missing_sample_count = 0
        self.early_short_window_sample_count = 0

    def update(self, metrics, reference_counts=None, diagnostics=None, invalid_reasons=None):
        if set(metrics) - set(METRIC_NAMES):
            raise ValueError('Unknown metric in v2 protocol')
        arrays = {key: np.asarray(value, dtype=np.float64) for key, value in metrics.items()}
        if any(value.ndim != 1 for value in arrays.values()):
            raise ValueError('Expected scalar metric arrays [B]')
        sizes = {len(value) for value in arrays.values()}
        if len(sizes) > 1:
            raise ValueError('Metric batch sizes differ')
        size = next(iter(sizes), 0)
        if any(np.isinf(value).any() for value in arrays.values()):
            raise ValueError('Infinite metric input')
        if reference_counts is not None:
            if len(reference_counts) != size:
                raise ValueError('Reference counts do not match batch')
            if any(int(count) != count or count < 0 for count in reference_counts):
                raise ValueError('Invalid reference count')
        reasons = {} if invalid_reasons is None else invalid_reasons
        if set(reasons) - set(arrays):
            raise ValueError('Invalid reason metric names')
        for key, values in reasons.items():
            if len(values) != size or any(not isinstance(value, str) for value in values):
                raise ValueError('Invalid reasons must be strings matching batch size')
            if any(bool(reason) != bool(np.isnan(value)) for reason, value in zip(values, arrays[key])):
                raise ValueError('Invalid reasons must match undefined metrics')
        diagnostic_arrays = {} if diagnostics is None else {
            key: np.asarray(value, dtype=np.float64) for key, value in diagnostics.items()
        }
        if set(diagnostic_arrays) - set(DIAGNOSTIC_NAMES):
            raise ValueError('Unknown diagnostic')
        for key, value in diagnostic_arrays.items():
            if value.shape != (size,) or np.isinf(value).any():
                raise ValueError('Invalid diagnostic shape or infinity')
            finite = value[np.isfinite(value)]
            if key == 'phase_gt_energy_coverage':
                if np.any((finite < 0) | (finite > 1 + 1e-12)):
                    raise ValueError('Invalid phase coverage')
            elif np.isnan(value).any() or np.any(finite != np.floor(finite)):
                raise ValueError('Diagnostic counts and sample positions must be integers')
            elif np.any(finite < (-1 if key == 'early_start_sample' else 0)):
                raise ValueError('Invalid diagnostic count or sample position')
        with np.errstate(over='raise', invalid='raise'):
            for key, value in arrays.items():
                for i, row in enumerate(value):
                    if np.isfinite(row):
                        self.sums[key] += row
                        self.counts[key] += 1
                    else:
                        self.invalid[key] += 1
                        reason = reasons[key][i] if key in reasons else 'undefined_unspecified'
                        self.invalid_reasons[key][reason] = self.invalid_reasons[key].get(reason, 0) + 1
            for key, value in diagnostic_arrays.items():
                valid = np.isfinite(value)
                if key == 'early_start_sample':
                    valid &= value >= 0
                elif key == 'early_window_samples':
                    valid &= value > 0
                for row in value[valid]:
                    self.diagnostic_sums[key] += row
                    self.diagnostic_counts[key] += 1
                    current_min = self.diagnostic_min[key]
                    self.diagnostic_min[key] = float(row) if current_min is None else min(current_min, float(row))
        missing = diagnostic_arrays.get('phase_missing_frequency_count', np.empty(0))
        self.phase_missing_frequency_total += sum(int(value) for value in missing)
        self.phase_missing_sample_count += int(np.count_nonzero(missing > 0))
        windows = diagnostic_arrays.get('early_window_samples', np.empty(0))
        self.early_short_window_sample_count += int(np.count_nonzero((windows > 0) & (windows < 800)))
        self.samples_evaluated += size
        if reference_counts is not None:
            for count in reference_counts:
                key = str(int(count))
                self.reference_count_histogram[key] = self.reference_count_histogram.get(key, 0) + 1
        return self

    def skip(self, count):
        if int(count) != count or count < 0:
            raise ValueError('Invalid skipped count')
        self.skipped_no_reference += int(count)
        return self

    def merge(self, other):
        if self.protocol != other.protocol or self.signal_length != other.signal_length:
            raise ValueError('Cannot merge different metric protocols or signal lengths')
        with np.errstate(over='raise', invalid='raise'):
            for key in METRIC_NAMES:
                self.sums[key] += other.sums[key]
                self.counts[key] += other.counts[key]
                self.invalid[key] += other.invalid[key]
                for reason, count in other.invalid_reasons[key].items():
                    self.invalid_reasons[key][reason] = self.invalid_reasons[key].get(reason, 0) + count
            for key in DIAGNOSTIC_NAMES:
                self.diagnostic_sums[key] += other.diagnostic_sums[key]
                self.diagnostic_counts[key] += other.diagnostic_counts[key]
                other_min = other.diagnostic_min[key]
                if other_min is not None:
                    current_min = self.diagnostic_min[key]
                    self.diagnostic_min[key] = other_min if current_min is None else min(current_min, other_min)
        for attr in ('samples_evaluated', 'skipped_no_reference', 'phase_missing_frequency_total',
                     'phase_missing_sample_count', 'early_short_window_sample_count'):
            setattr(self, attr, getattr(self, attr) + getattr(other, attr))
        for key, value in other.reference_count_histogram.items():
            self.reference_count_histogram[key] = self.reference_count_histogram.get(key, 0) + value
        return self

    def state_dict(self):
        from copy import deepcopy
        return deepcopy({
            'protocol': self.protocol, 'signal_length': self.signal_length,
            'samples_evaluated': self.samples_evaluated,
            'skipped_no_reference': self.skipped_no_reference,
            'reference_count_histogram': self.reference_count_histogram,
            'sums': {k: float(v) for k, v in self.sums.items()},
            'counts': {k: int(v) for k, v in self.counts.items()},
            'invalid': {k: int(v) for k, v in self.invalid.items()},
            'invalid_reasons': self.invalid_reasons,
            'diagnostic_sums': {k: float(v) for k, v in self.diagnostic_sums.items()},
            'diagnostic_counts': self.diagnostic_counts,
            'diagnostic_min': self.diagnostic_min,
            'phase_missing_frequency_total': self.phase_missing_frequency_total,
            'phase_missing_sample_count': self.phase_missing_sample_count,
            'early_short_window_sample_count': self.early_short_window_sample_count,
        })

    @classmethod
    def from_state_dict(cls, state):
        from copy import deepcopy
        if 'protocol' not in state or 'signal_length' not in state:
            raise ValueError('Missing v2 protocol metadata; historical states are unsupported')
        obj = cls(signal_length=state['signal_length'])
        if state['protocol'] != obj.protocol:
            raise ValueError('Unsupported metric protocol')
        expected = obj.state_dict()
        if set(state) != set(expected):
            raise ValueError('Invalid v2 accumulator state fields')
        for attr in ('sums', 'counts', 'invalid', 'invalid_reasons'):
            if set(state[attr]) != set(METRIC_NAMES):
                raise ValueError('Unknown or missing metric in accumulator state')
        for attr in ('diagnostic_sums', 'diagnostic_counts', 'diagnostic_min'):
            if set(state[attr]) != set(DIAGNOSTIC_NAMES):
                raise ValueError('Unknown or missing diagnostic in accumulator state')
        for attr, dtype in (('sums', np.float64), ('counts', np.int64), ('invalid', np.int64)):
            values = {}
            for key, value in state[attr].items():
                array = np.asarray(value)
                if array.shape != () or not np.isfinite(array) or array < 0:
                    raise ValueError('Invalid metric accumulator scalar')
                if dtype == np.int64 and int(value) != value:
                    raise ValueError('Invalid accumulator count')
                values[key] = np.asarray(value, dtype=dtype)
            setattr(obj, attr, values)
        for attr in ('samples_evaluated', 'skipped_no_reference', 'phase_missing_frequency_total',
                     'phase_missing_sample_count', 'early_short_window_sample_count'):
            value = state[attr]
            if not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError('Invalid accumulator sample count')
            setattr(obj, attr, int(value))
        obj.reference_count_histogram = deepcopy(state['reference_count_histogram'])
        obj.invalid_reasons = deepcopy(state['invalid_reasons'])
        obj.diagnostic_sums = {k: np.float64(v) for k, v in state['diagnostic_sums'].items()}
        obj.diagnostic_counts = deepcopy(state['diagnostic_counts'])
        obj.diagnostic_min = deepcopy(state['diagnostic_min'])
        return obj

    def summary(self):
        from copy import deepcopy
        metrics = {}
        for key in METRIC_NAMES:
            count = int(self.counts[key])
            value = float(self.sums[key] / count) if count else None
            entry = {'value': value, 'valid_count': count, 'invalid_count': int(self.invalid[key])}
            if key == 'waveform_nmse':
                entry['db'] = 10 * float(np.log10(value)) if value is not None and value > 0 else None
                entry['db_status'] = 'undefined' if value is None else 'negative_infinity' if value == 0 else 'finite'
            metrics[key] = entry
        diagnostics = {}
        for key in DIAGNOSTIC_NAMES:
            count = self.diagnostic_counts[key]
            diagnostics[key] = {
                'mean': float(self.diagnostic_sums[key] / count) if count else None,
                'min': self.diagnostic_min[key], 'valid_count': count,
            }
        for attr in ('phase_missing_frequency_total', 'phase_missing_sample_count',
                     'early_short_window_sample_count'):
            diagnostics[attr] = getattr(self, attr)
        return {
            'protocol': deepcopy(self.protocol), 'metrics': metrics,
            'samples_evaluated': self.samples_evaluated,
            'skipped_no_reference': self.skipped_no_reference,
            'reference_count_histogram': dict(self.reference_count_histogram),
            'invalid_reasons': deepcopy(self.invalid_reasons), 'diagnostics': diagnostics,
        }
