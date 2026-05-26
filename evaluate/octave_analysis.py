import numpy as np
from scipy.signal import butter, filtfilt
import pandas as pd
from tqdm import tqdm
import os
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader

def get_valid_bands(fs):
    """根据 IEC 61260 / ISO 266 生成 7 个核心倍频程"""
    import math
    sqrt2 = math.sqrt(2)
    
    # 7个核心倍频程中心频率
    center_freqs = {
        "oct1 (63Hz)": 63.0,
        "oct2 (125Hz)": 125.0,
        "oct3 (250Hz)": 250.0,
        "oct4 (500Hz)": 500.0,
        "oct5 (1kHz)": 1000.0,
        "oct6 (2kHz)": 2000.0,
        "oct7 (4kHz)": 4000.0,
    }
    
    valid_bands = {}
    nyquist = fs / 2.0
    
    for name, fc in center_freqs.items():
        low = fc / sqrt2
        high = fc * sqrt2
        
        if low >= nyquist: continue
        if high >= nyquist: high = nyquist - 0.1
        valid_bands[name] = (low, high)
        
    return valid_bands

# ==========================================
# 提取自 evaluator.py 的声学评测函数
# ==========================================
def measure_clarity(signal, time=50, fs=16000):
    h2 = signal**2
    t = int((time/1000)*fs + 1)
    late_energy = np.sum(h2[t:])
    if late_energy == 0: late_energy = 1e-12 # 防除零报错
    early_energy = np.sum(h2[:t])
    if early_energy == 0: early_energy = 1e-12
    return 10*np.log10(early_energy/late_energy)

def measure_edt(h, fs=16000, decay_db=10):
    h = np.array(h)
    fs = float(fs)
    power = h ** 2
    energy = np.cumsum(power[::-1])[::-1] 
    
    if np.max(energy) <= 0: return np.inf
    i_nz = np.max(np.where(energy > 0)[0])
    energy = energy[:i_nz]
    energy_db = 10 * np.log10(energy + 1e-12)
    energy_db -= energy_db[0]

    try:
        i_decay = np.min(np.where(- decay_db - energy_db > 0)[0])
    except ValueError:
        return np.inf
    t_decay = i_decay / fs
    decay_time = t_decay
    est_edt = (60 / decay_db) * decay_time
    return est_edt

def measure_rt60(h, fs=16000, decay_db=20):
    h = np.array(h)
    fs = float(fs)
    power = h ** 2
    energy = np.cumsum(power[::-1])[::-1] 

    if np.max(energy) <= 0: return np.inf
    i_nz = np.max(np.where(energy > 0)[0])
    energy = energy[:i_nz]
    energy_db = 10 * np.log10(energy + 1e-12)
    energy_db -= energy_db[0]
    
    try:
        i_5db = np.min(np.where(-5 - energy_db > 0)[0])
    except ValueError:
        return np.inf
    t_5db = i_5db / fs

    try:
        i_decay = np.min(np.where(-5 - decay_db - energy_db > 0)[0])
    except ValueError:
        return np.inf
    t_decay = i_decay / fs

    decay_time = t_decay - t_5db
    est_rt60 = (60 / decay_db) * decay_time
    return est_rt60

# ==========================================
# 数据集定义
# ==========================================
class RIRDataset(Dataset):
    def __init__(self, data_pairs, fs=16000):
        import librosa
        self.librosa = librosa
        self.data_pairs = data_pairs
        self.fs = fs
        self.bands = get_valid_bands(fs)
        
        self.filters = {}
        for name, (low, high) in self.bands.items():
            self.filters[name] = butter(4, [low, high], btype='bandpass', fs=fs)

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        gt_path, pred_path = self.data_pairs[idx]
        try:
            gt_rir = self.librosa.load(gt_path, sr=self.fs)[0][:8000]
            pred_rir = self.librosa.load(pred_path, sr=self.fs)[0][:8000]
            min_len = min(len(gt_rir), len(pred_rir))
            gt_rir = gt_rir[:min_len]
            pred_rir = pred_rir[:min_len]
        except Exception:
            gt_rir = np.zeros(8000)
            pred_rir = np.zeros(8000)

        out_gt = {}
        out_pred = {}
        for name, (b, a) in self.filters.items():
            f_gt = filtfilt(b, a, gt_rir).copy()
            f_pred = filtfilt(b, a, pred_rir).copy()
            out_gt[name] = torch.from_numpy(f_gt).float()
            out_pred[name] = torch.from_numpy(f_pred).float()
            
        return out_gt, out_pred

# ==========================================
# 核心评测函数
# ==========================================
def evaluate_dataset(data_pairs, fs=16000, batch_size=32, num_workers=4):
    dataset = RIRDataset(data_pairs, fs=fs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 存储各个频段的误差列表
    c50_errors = {b: [] for b in dataset.bands.keys()}
    edt_errors = {b: [] for b in dataset.bands.keys()}
    t60_errors = {b: [] for b in dataset.bands.keys()}

    for batch_gt, batch_pred in tqdm(dataloader, desc="Evaluating Batches"):
        B = list(batch_gt.values())[0].shape[0] 
        
        for band in dataset.bands.keys():
            # 获取当前 batch 该频段的数据并转为 numpy
            gt_np = batch_gt[band].numpy()
            pred_np = batch_pred[band].numpy()
            
            for i in range(B):
                gt_sig = gt_np[i]
                pred_sig = pred_np[i]
                
                # 计算指标
                gt_c50 = measure_clarity(gt_sig, fs=fs)
                pred_c50 = measure_clarity(pred_sig, fs=fs)
                gt_edt = measure_edt(gt_sig, fs=fs)
                pred_edt = measure_edt(pred_sig, fs=fs)
                gt_t60 = measure_rt60(gt_sig, fs=fs)
                pred_t60 = measure_rt60(pred_sig, fs=fs)
                
                # 遵循 evaluator.py 的误差定义逻辑
                # C50: MAE (绝对误差)
                if not np.isinf(gt_c50) and not np.isinf(pred_c50) and not np.isnan(gt_c50) and not np.isnan(pred_c50):
                    c50_errors[band].append(abs(pred_c50 - gt_c50))
                
                # EDT: MAE (绝对误差)
                if not np.isinf(gt_edt) and not np.isinf(pred_edt):
                    edt_errors[band].append(abs(pred_edt - gt_edt))
                    
                # T60: Relative Error (相对误差)
                if not np.isinf(gt_t60) and not np.isinf(pred_t60) and gt_t60 > 0:
                    t60_errors[band].append(abs(pred_t60 - gt_t60) / gt_t60 * 100)

    # 计算均值
    c50_mean = {b: round(float(np.mean(v)), 3) if len(v)>0 else np.nan for b, v in c50_errors.items()}
    edt_mean = {b: round(float(np.mean(v)), 3) if len(v)>0 else np.nan for b, v in edt_errors.items()}
    t60_mean = {b: round(float(np.mean(v)), 3) if len(v)>0 else np.nan for b, v in t60_errors.items()}

    return {'c50_mean': c50_mean, 'edt_mean': edt_mean, 't60_mean': t60_mean}

def main(audio_dir, gt_df):
    data_pairs = []
    for i in range(len(gt_df)):
        loc = gt_df.iloc[i]
        scene_id = loc['scene_id']
        gt_rir_path = loc['rir_path']
        rir_filename = os.path.basename(gt_rir_path)
        pred_audio_filename = f'{scene_id}__{rir_filename}' 
        pred_audio_path = os.path.join(audio_dir, pred_audio_filename)

        if os.path.exists(pred_audio_path) and os.path.exists(gt_rir_path):
            data_pairs.append((gt_rir_path, pred_audio_path))
            
    if len(data_pairs) == 0:
        return {'c50_mean': {}, 'edt_mean': {}, 't60_mean': {}}

    print(f"Discovered {len(data_pairs)} valid audio pairs.")
    
    # 纯 CPU 计算声学指标，可调大 num_workers 加速文件读取
    results = evaluate_dataset(data_pairs, fs=16000, batch_size=64, num_workers=4)
    
    print("\n[Evaluation Results]")
    print("C50 Mean Abs Error:")
    for b, v in results['c50_mean'].items(): print(f"  {b}: {v:.4f}")
    print("EDT Mean Abs Error:")
    for b, v in results['edt_mean'].items(): print(f"  {b}: {v:.4f}")
    print("T60 Mean Rel Error:")
    for b, v in results['t60_mean'].items(): print(f"  {b}: {v:.4f}")
    
    return results

if __name__ == "__main__":
    out_root = '/mnt/data/jingchong/eigenet/output'
    gt_jsonl_path = '/data/250010171/code/EigeNet_discriminant/data/AcousticRooms_test_unseen.jsonl'
    gt_df = pd.read_json(gt_jsonl_path, lines=True)
    
    model_rename = {
        'discriminant_ablation_ca_noalign': 'Cross-Attention',
        'discriminant_ablation_only_loc_noalign': 'Location-only',
        'discriminant_ablation_only_depth_noalign': 'Depth-only',
        'discriminant_ablation_sa_noalign': 'Self-Attention',
        'discriminant_base2_g2_noalign_debug': 'Ours w/o align',
        'discriminant_base2_aa_align_toy3.4': 'Ours',
    }
    
    for K in [1,4,8]:
        # 初始化三个指标的 DataFrame
        columns_base = ['model', 'exp']
        bands_cols = ['oct1 (63Hz)', 'oct2 (125Hz)', 'oct3 (250Hz)', 'oct4 (500Hz)', 'oct5 (1kHz)', 'oct6 (2kHz)', 'oct7 (4kHz)']
        
        metrics_c50_df = pd.DataFrame(columns=columns_base + [f"c50_{b}" for b in bands_cols])
        metrics_edt_df = pd.DataFrame(columns=columns_base + [f"edt_{b}" for b in bands_cols])
        metrics_t60_df = pd.DataFrame(columns=columns_base + [f"t60_{b}" for b in bands_cols])
        
        for model in os.listdir(out_root):
            model_dir = os.path.join(out_root, model)
            if model not in model_rename:
                continue
            new_model_name = model_rename[model]
            
            for exp in os.listdir(model_dir):
                exp_dir = os.path.join(model_dir, exp)
                audio_dir = os.path.join(exp_dir, 'unseen', str(K))
                
                print(f"\n--- Evaluating Model: {new_model_name} | Exp: {exp} | K: {K} ---")
                results = main(audio_dir, gt_df)
                
                if not results.get('c50_mean'):
                    continue
                    
                # 提取数据装入 DataFrame
                c50_row = [new_model_name, exp] + [results['c50_mean'].get(b, np.nan) for b in bands_cols]
                edt_row = [new_model_name, exp] + [results['edt_mean'].get(b, np.nan) for b in bands_cols]
                t60_row = [new_model_name, exp] + [results['t60_mean'].get(b, np.nan) for b in bands_cols]
                
                metrics_c50_df.loc[len(metrics_c50_df)] = c50_row
                metrics_edt_df.loc[len(metrics_edt_df)] = edt_row
                metrics_t60_df.loc[len(metrics_t60_df)] = t60_row
                
        save_dir = '/data/250010171/code/EigeNet_discriminant/data/visualization/octave_analysis/'
        os.makedirs(save_dir, exist_ok=True)
        
        c50_csv_path = os.path.join(save_dir, f'c50_error_K{K}.csv')
        edt_csv_path = os.path.join(save_dir, f'edt_error_K{K}.csv')
        t60_csv_path = os.path.join(save_dir, f't60_error_K{K}.csv')
        
        metrics_c50_df.to_csv(c50_csv_path, index=False)
        metrics_edt_df.to_csv(edt_csv_path, index=False)
        metrics_t60_df.to_csv(t60_csv_path, index=False)
        
        print(f"\n>>> Saved K={K} results to:")
        print(f"    - {c50_csv_path}")
        print(f"    - {edt_csv_path}")
        print(f"    - {t60_csv_path}")