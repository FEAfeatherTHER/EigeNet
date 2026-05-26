import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os
from tqdm import tqdm
import pandas as pd
import sys
import librosa

src = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # AnyTrainer
sys.path.insert(0, src)
from models.loss.evaluator import Evaluator
from models.layers.utils import compute_metrics, plot_waveform

def plot_heatmap_on_ax(ax, x_coords, y_coords, values, vmin, vmax, title):
    """
    在指定的 ax 上绘制单张 2D 空间热力图。
    """
    # 1. 生成密集网格
    grid_x, grid_y = np.mgrid[min(x_coords):max(x_coords):200j, 
                              min(y_coords):max(y_coords):200j]
    
    # 2. 线性插值（主体平滑）
    grid_values = griddata((x_coords, y_coords), values, (grid_x, grid_y), method='linear')
    
    # 3. 填补边界空白 (补齐缺角变成标准矩形)
    nan_mask = np.isnan(grid_values)
    if np.any(nan_mask):
        grid_nearest = griddata((x_coords, y_coords), values, (grid_x, grid_y), method='nearest')
        grid_values[nan_mask] = grid_nearest[nan_mask]
    
    cmap_style = 'viridis' 
    
    # 4. 绘制等高线填充图
    ax.set_facecolor('#f0f0f0') 
    contour = ax.contourf(grid_x, grid_y, grid_values, levels=30, cmap=cmap_style, vmin=vmin, vmax=vmax)
    
    # 5. 设置图表属性
    ax.set_title(title, fontsize=16)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 返回 contour 对象，用于在外部生成共享的 Colorbar
    return contour

if __name__ == "__main__":
    test_len = int(0.363*16000)
    ours_rir_dir = '/data/250010171/code/EigeNet_discriminant/visualization/ours_dampenedbase_pred_rir'
    xrir_rir_dir = '/data/250010171/code/EigeNet_discriminant/visualization/xrir_dampenedbase_pred_rir'
    evaluator = Evaluator()
    jsonl_path = '/data/250010171/code/EigeNet_discriminant/data/haa_finetune_test.jsonl'
    
    df = pd.read_json(jsonl_path, lines=True)
    df = df[df['scene_name'] == 'dampenedBase']
    
    x_list = []
    y_list = []
    c50_gt_list = []
    c50_ours_list = []
    c50_xrir_list = []
    
    first_rec_loc = df.iloc[0]['rec_loc']
    ref_x = first_rec_loc[0]
    ref_z = first_rec_loc[2] 
    
    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        src_idx = row['src_idx']
        rec_loc = row['rec_loc']  
        src_loc = row['src_loc']
        
        gt_rir_path = row['rir_path']
        ours_rir_path = os.path.join(ours_rir_dir, f'dampenedBase_{src_idx}.wav')
        xrir_rir_path = os.path.join(xrir_rir_dir, f'dampenedBase_{src_idx}.wav')
        
        gt_rir = librosa.load(gt_rir_path, sr=16000)[0][:test_len]
        ours_rir = librosa.load(ours_rir_path, sr=16000)[0][:test_len]
        xrir_rir = librosa.load(xrir_rir_path, sr=16000)[0][:test_len]
        
        gt_c50 = evaluator.measure_clarity(gt_rir)
        ours_c50 = evaluator.measure_clarity(ours_rir)
        xrir_c50 = evaluator.measure_clarity(xrir_rir)
        
        plot_x = src_loc[0] - ref_x
        plot_y = src_loc[2] - ref_z 
        
        x_list.append(plot_x)
        y_list.append(plot_y)
        c50_gt_list.append(gt_c50)
        c50_ours_list.append(ours_c50)
        c50_xrir_list.append(xrir_c50)
        
    heatmap_dir = '/data/250010171/code/EigeNet_discriminant/visualization/heatmap_c50'
    os.makedirs(heatmap_dir, exist_ok=True)
    
    # 转为 numpy 数组
    x_coords = np.array(x_list)
    y_coords = np.array(y_list)
    gt_values = np.array(c50_gt_list)
    ours_values = np.array(c50_ours_list)
    xrir_values = np.array(c50_xrir_list)
    
    # 【1】跨越三组数据，计算全局统一的极值
    global_vmin = min(np.min(gt_values), np.min(ours_values), np.min(xrir_values))
    global_vmax = max(np.max(gt_values), np.max(ours_values), np.max(xrir_values))
    
    # 【2】修改为 3 行 1 列的画布
    # figsize=(8, 12) 更适合垂直堆叠，你可以根据实际需要微调
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    
    # 【3】从上到下依次作图
    contour = plot_heatmap_on_ax(axes[0], x_coords, y_coords, gt_values, global_vmin, global_vmax, "Ground Truth")
    plot_heatmap_on_ax(axes[2], x_coords, y_coords, xrir_values, global_vmin, global_vmax, "xRIR")
    plot_heatmap_on_ax(axes[1], x_coords, y_coords, ours_values, global_vmin, global_vmax, "Ours")
    
    # 【4】调整子图的垂直间距 (hspace) 以防止标题重叠，并给右侧留出空间 (right=0.85)
    fig.subplots_adjust(hspace=0.3, right=0.85, top=0.92) 
    
    # 【5】在右侧添加一个贯穿三张图的统一 Colorbar
    # [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.88, 0.15, 0.025, 0.7]) 
    cbar = fig.colorbar(contour, cax=cbar_ax)
    cbar.set_label("C50 (dB)", fontsize=14)
    
    
    save_path = os.path.join(heatmap_dir, 'heatmap_c50.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"垂直堆叠对比热力图已成功保存至: {save_path}")