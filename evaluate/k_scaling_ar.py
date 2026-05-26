"""Plot Main Results on AcousticRooms dataset for different K values.
Generates a 1x3 plot (Cols: EDT, C50, T60).

Outputs:
    /data/250010171/code/EigeNet_discriminant/data/visualization/main_results/ar_metrics.pdf
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
SAVE_DIR = Path("/data/250010171/code/EigeNet_discriminant/data/visualization/main_results")
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
# Data (内嵌数据，方便直接运行)
# --------------------------------------------------------------------------- #
CSV_DATA = """model,K,EDT,C50,T60
xRIR,1,0.076,2.124,12.617
xRIR,4,0.054,1.540,10.052
xRIR,8,0.050,1.442,9.393
Ours,1,0.052,1.488,10.213
Ours,4,0.047,1.398,8.061
Ours,8,0.041,1.242,7.605
"""

# --------------------------------------------------------------------------- #
# Main Plotting Logic
# --------------------------------------------------------------------------- #
def plot_main_results_vs_k():
    print("\n--- Generating Main Results Plot (Metrics vs K) ---")
    
    # 1. Load data
    df = pd.read_csv(io.StringIO(CSV_DATA))
    
    # 获取唯一的模型和 K 值
    models = ["xRIR", "Ours"]
    k_values = [1, 4, 8]
    
    # 定义子图：1 行 3 列 (分别画 EDT, C50, T60)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 4.5))
    
    metrics = [
        {"col": "EDT", "title": "EDT"},
        {"col": "C50", "title": "C50"},
        {"col": "T60", "title": "T60"}
    ]
    
    handles_dict = {}

    # 2. Plotting loop
    for col_idx, metric_info in enumerate(metrics):
        ax = axes[col_idx]
        metric_col = metric_info["col"]
        
        for model_name in models:
            style = MODEL_STYLES.get(model_name, {"color": "gray", "linestyle": "-", "marker": "o"})
            
            # 筛选当前模型的数据并按 K 排序
            model_data = df[df['model'] == model_name].sort_values(by="K")
            
            if not model_data.empty:
                line, = ax.plot(
                    model_data['K'], model_data[metric_col], 
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
    out_file = SAVE_DIR / "k_scaling_ar.pdf"
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Successfully saved to: {out_file}")

if __name__ == "__main__":
    plot_main_results_vs_k()