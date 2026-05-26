"""Plot Main Results on HAA dataset for different K values.
Aggregates (averages) across all scene types (classroomBase, hallwayBase, etc.).
Generates a 1x3 plot (Cols: EDT, C50, T60).

Outputs:
    /data/250010171/code/EigeNet_discriminant/visualization/main_results/haa/haa_metrics.pdf
"""

from __future__ import annotations

import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import io

# --------------------------------------------------------------------------- #
# Paths & Config
# --------------------------------------------------------------------------- #
SAVE_DIR = Path("/data/250010171/code/EigeNet_discriminant/data/visualization/main_results/")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Global style matching publication reference
TITLE_FS, LABEL_FS, TICK_FS, LEGEND_FS = 16, 14, 14, 18
sns.set_theme(style="whitegrid", context="paper")
# --------------------------------------------------------------------------- #
# Style definitions
# --------------------------------------------------------------------------- #
MODEL_STYLES = {
    "Ours": {"color": "#D95F02", "linestyle": "-", "marker": "*"},            # Orange solid star
    "xRIR": {"color": "#1F78B4", "linestyle": "--", "marker": "s"},           # Blue dashed square
}

# --------------------------------------------------------------------------- #
# Data (内嵌数据，或者你可以替换为 pd.read_csv 读取实际文件)
# --------------------------------------------------------------------------- #
CSV_DATA = """model,scene_type,reference_count,edt_error,c50_error,t60_error
Ours,classroomBase,1,0.039,0.782,3.066
Ours,classroomBase,4,0.038,0.816,3.227
Ours,classroomBase,8,0.037,0.842,3.463
Ours,hallwayBase,1,0.042,0.576,2.516
Ours,hallwayBase,4,0.038,0.557,2.537
Ours,hallwayBase,8,0.036,0.519,2.44
Ours,dampenedBase,1,0.026,1.935,49.09
Ours,dampenedBase,4,0.026,1.925,46.072
Ours,dampenedBase,8,0.025,2.002,49.261
Ours,complexBase,1,0.024,0.547,2.62
Ours,complexBase,4,0.022,0.527,2.717
Ours,complexBase,8,0.023,0.562,2.764
xRIR,classroomBase,1,0.045,1.035,10.693
xRIR,classroomBase,4,0.04,0.944,11.438
xRIR,classroomBase,8,0.039,0.953,11.134
xRIR,hallwayBase,1,0.046,0.95,3.019
xRIR,hallwayBase,4,0.051,1.1,3.905
xRIR,hallwayBase,8,0.047,1.029,3.668
xRIR,dampenedBase,1,0.021,3.912,238.325
xRIR,dampenedBase,4,0.019,4.124,243.508
xRIR,dampenedBase,8,0.021,4.006,254.719
xRIR,complexBase,1,0.032,0.742,13.591
xRIR,complexBase,4,0.04,0.908,13.209
xRIR,complexBase,8,0.035,0.86,13.643
"""

# --------------------------------------------------------------------------- #
# Main Plotting Logic
# --------------------------------------------------------------------------- #
def plot_haa_main_results_vs_k():
    print("\n--- Generating HAA Main Results Plot (Averaged across scenes) ---")
    
    # 1. Load data
    df = pd.read_csv(io.StringIO(CSV_DATA))
    # 如果要从本地读取，解除下面这行的注释并修改路径：
    # df = pd.read_csv("/data/250010171/code/EigeNet_discriminant/data/visualization/main_results/haa/haa_metrics.csv")
    
    # 2. 核心：忽略 scene_type，对 (model, reference_count) 分组并求平均值
    df_mean = df.groupby(['model', 'reference_count'])[['edt_error', 'c50_error', 't60_error']].mean().reset_index()
    
    # 获取唯一的模型和 K 值
    models = ["xRIR", "Ours"]
    k_values = [1, 4, 8]
    
    # 定义子图：1 行 3 列 (分别画 EDT, C50, T60)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 4.5))
    
    metrics = [
        {"col": "edt_error", "title": "EDT"},
        {"col": "c50_error", "title": "C50"},
        {"col": "t60_error", "title": "T60"}
    ]
    
    handles_dict = {}

    # 3. Plotting loop
    for col_idx, metric_info in enumerate(metrics):
        ax = axes[col_idx]
        metric_col = metric_info["col"]
        
        for model_name in models:
            style = MODEL_STYLES.get(model_name, {"color": "gray", "linestyle": "-", "marker": "o"})
            
            # 筛选当前模型的数据并按 K 排序 (这里 K 就是 reference_count)
            model_data = df_mean[df_mean['model'] == model_name].sort_values(by="reference_count")
            
            if not model_data.empty:
                line, = ax.plot(
                    model_data['reference_count'], model_data[metric_col], 
                    label=model_name,
                    color=style['color'], linestyle=style['linestyle'],
                    marker=style['marker'], markerfacecolor=style['color'], markeredgecolor=style['color']
                )
                if model_name not in handles_dict:
                    handles_dict[model_name] = line
                    
        # -------------------
        # Subplot Aesthetics
        # -------------------
        ax.set_title(metric_info["title"], fontsize=TITLE_FS, pad=12)
        ax.set_xlabel("$K$", fontsize=LABEL_FS)
        #ax.set_ylabel(metric_info["ylabel"], fontsize=LABEL_FS)
        ax.tick_params(axis="both", labelsize=TICK_FS)
        
        # 强制设定 X 轴刻度仅为 1, 4, 8
        ax.set_xticks(k_values)
        ax.set_xticklabels([str(k) for k in k_values])

    # -------------------
    # Global Legend & Layout
    # -------------------
    handles = [handles_dict[m] for m in models if m in handles_dict]
    labels = [m for m in models if m in handles_dict]
    
    # 放置全局居中图例
    fig.legend(
        handles, labels, 
        loc="upper center", 
        ncol=len(models),
        frameon=False, 
        bbox_to_anchor=(0.5, 1.05),
        fontsize=LEGEND_FS,
    )
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)

    # Save
    out_file = SAVE_DIR / "k_scaling_haa.pdf"
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Successfully saved to: {out_file}")

if __name__ == "__main__":
    plot_haa_main_results_vs_k()