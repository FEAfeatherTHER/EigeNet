"""Plot Octave-band analysis curves for EigeNet, grouped by ablation studies.
Generates one 3x3 plot per ablation group (Rows: EDT, C50, T60, Cols: K=1,4,8).

Outputs:
    /data/250010171/code/EigeNet_discriminant/data/octave_analysis/fig/ablation_geo_input.pdf
    /data/250010171/code/EigeNet_discriminant/data/octave_analysis/fig/ablation_attention.pdf
    /data/250010171/code/EigeNet_discriminant/data/octave_analysis/fig/ablation_alignment.pdf
"""

from __future__ import annotations

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --------------------------------------------------------------------------- #
# Paths & Config
# --------------------------------------------------------------------------- #
ROOT_DIR = Path("/data/250010171/code/EigeNet_discriminant/data/visualization/octave_analysis")
FIG_DIR = ROOT_DIR / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Global style matching publication reference
TITLE_FS, LABEL_FS, TICK_FS, LEGEND_FS = 16, 14, 14, 18
sns.set_theme(style="whitegrid", context="paper")

# --------------------------------------------------------------------------- #
# Style definitions ensuring 'Ours' remains consistent across all plots
# --------------------------------------------------------------------------- #
MODEL_STYLES = {
    "Ours": {"color": "#D95F02", "linestyle": "-", "marker": "*"},            # Orange solid
    "Ours w/o align": {"color": "#E6AB02", "linestyle": "-.", "marker": "p"}, # Yellow dash-dot
    "Cross-Attention": {"color": "#33A02C", "linestyle": "--", "marker": "s"},# Green dashed
    "Self-Attention": {"color": "#1F78B4", "linestyle": "--", "marker": "D"}, # Blue dashed
    "Location-only": {"color": "#33A02C", "linestyle": "--", "marker": "s"},  # Green dashed
    "Depth-only": {"color": "#1F78B4", "linestyle": "--", "marker": "D"},     # Green dashed
}

# --------------------------------------------------------------------------- #
# Definitions
# --------------------------------------------------------------------------- #
X_LABELS = [f"Oct {i}" for i in range(1, 8)]
K_VALUES = [1, 4, 8]

ABLATION_GROUPS = {
    "ablation_geo_input": ["Ours w/o align", "Depth-only", "Location-only"],
    "ablation_attention": ["Ours w/o align", "Self-Attention", "Cross-Attention"],
    "ablation_alignment": ["Ours", "Ours w/o align"]
}

# --------------------------------------------------------------------------- #
# Main Plotting Logic
# --------------------------------------------------------------------------- #
def plot_ablation_group(group_name, target_models):
    print(f"\n--- Processing {group_name} ---")
    
    # Create a 3x3 grid: 3 rows (EDT, C50, T60) x 3 columns (K=1, 4, 8)
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12.5, 9.5))
    x_positions = np.arange(len(X_LABELS))
    
    handles_dict = {}  # Store one handle per model for the global legend

    for col_idx, K in enumerate(K_VALUES):
        # 修改点 1：重新分配各行的坐标轴 (Row 0: EDT, Row 1: C50, Row 2: T60)
        ax_edt = axes[0, col_idx]
        ax_c50 = axes[1, col_idx]
        ax_t60 = axes[2, col_idx]
        
        # Load Data
        c50_csv_path = ROOT_DIR / f"c50_error_K{K}.csv"
        edt_csv_path = ROOT_DIR / f"edt_error_K{K}.csv"
        t60_csv_path = ROOT_DIR / f"t60_error_K{K}.csv"
        
        df_c50 = pd.read_csv(c50_csv_path) if c50_csv_path.exists() else pd.DataFrame()
        df_edt = pd.read_csv(edt_csv_path) if edt_csv_path.exists() else pd.DataFrame()
        df_t60 = pd.read_csv(t60_csv_path) if t60_csv_path.exists() else pd.DataFrame()

        # Process C50
        if not df_c50.empty:
            df_c50 = df_c50[df_c50['model'].isin(target_models)]
            df_c50_mean = df_c50.groupby('model').mean(numeric_only=True).reset_index()
            c50_cols = [col for col in df_c50.columns if "oct" in col.lower() and col not in ['model', 'exp']]
        else:
            df_c50_mean, c50_cols = pd.DataFrame(), []

        # Process EDT
        if not df_edt.empty:
            df_edt = df_edt[df_edt['model'].isin(target_models)]
            df_edt_mean = df_edt.groupby('model').mean(numeric_only=True).reset_index()
            edt_cols = [col for col in df_edt.columns if "oct" in col.lower() and col not in ['model', 'exp']]
        else:
            df_edt_mean, edt_cols = pd.DataFrame(), []

        # Process T60
        if not df_t60.empty:
            df_t60 = df_t60[df_t60['model'].isin(target_models)]
            df_t60_mean = df_t60.groupby('model').mean(numeric_only=True).reset_index()
            t60_cols = [col for col in df_t60.columns if "oct" in col.lower() and col not in ['model', 'exp']]
        else:
            df_t60_mean, t60_cols = pd.DataFrame(), []

        # Plot Data
        for model_name in target_models:
            style = MODEL_STYLES.get(model_name, {"color": "gray", "linestyle": "-", "marker": "o"})
            if model_name == "Ours w/o align" and group_name == "ablation_alignment":
                style = {"color": "#1F78B4", "linestyle": "--", "marker": "D"}
            
            # 1. Plot EDT (Row 0)
            if not df_edt_mean.empty and model_name in df_edt_mean['model'].values:
                row_edt = df_edt_mean[df_edt_mean['model'] == model_name].iloc[0]
                edt_vals = row_edt[edt_cols].values
                if len(edt_vals) == len(X_LABELS):
                    line, = ax_edt.plot(
                        x_positions, edt_vals, label=model_name,
                        color=style['color'], linestyle=style['linestyle'],
                        marker=style['marker'], markerfacecolor=style['color'], markeredgecolor=style['color']
                    )
                    # Save handle for global legend
                    if model_name not in handles_dict:
                        handles_dict[model_name] = line
                        
            # 2. Plot C50 (Row 1)
            if not df_c50_mean.empty and model_name in df_c50_mean['model'].values:
                row_c50 = df_c50_mean[df_c50_mean['model'] == model_name].iloc[0]
                c50_vals = row_c50[c50_cols].values
                if len(c50_vals) == len(X_LABELS):
                    ax_c50.plot(
                        x_positions, c50_vals, label=model_name,
                        color=style['color'], linestyle=style['linestyle'],
                        marker=style['marker'], markerfacecolor=style['color'], markeredgecolor=style['color']
                    )
                    
            # 3. Plot T60 (Row 2)
            if not df_t60_mean.empty and model_name in df_t60_mean['model'].values:
                row_t60 = df_t60_mean[df_t60_mean['model'] == model_name].iloc[0]
                t60_vals = row_t60[t60_cols].values
                if len(t60_vals) == len(X_LABELS):
                    ax_t60.plot(
                        x_positions, t60_vals, label=model_name,
                        color=style['color'], linestyle=style['linestyle'],
                        marker=style['marker'], markerfacecolor=style['color'], markeredgecolor=style['color']
                    )

        # -------------------
        # Subplot Aesthetics
        # -------------------
        # 修改点 2：将列标题挂载在最上方的 ax_edt 上
        ax_edt.set_title(f"$K={K}$", fontsize=TITLE_FS, pad=10)
        
        # 修改点 3：确保按 [ax_edt, ax_c50, ax_t60] 的顺序配置坐标轴
        for row_idx, ax in enumerate([ax_edt, ax_c50, ax_t60]):
            ax.set_xticks(x_positions)
            ax.set_xticklabels(X_LABELS, rotation=45, ha="right")
            ax.tick_params(axis="both", labelsize=TICK_FS)
            
            # 只在最左侧的一列显示 Y 轴名称，同步调整顺序
            if col_idx == 0:
                if row_idx == 0:
                    ax.set_ylabel("EDT", fontsize=LABEL_FS)
                elif row_idx == 1:
                    ax.set_ylabel("C50", fontsize=LABEL_FS)
                elif row_idx == 2:
                    ax.set_ylabel("T60", fontsize=LABEL_FS)

    # -------------------
    # Global Legend & Layout
    # -------------------
    # Extract legend items maintaining the order in `target_models`
    handles = [handles_dict[m] for m in target_models if m in handles_dict]
    labels = [m for m in target_models if m in handles_dict]
    
    fig.legend(
        handles, labels, 
        loc="upper center", 
        ncol=len(target_models),
        frameon=False, 
        bbox_to_anchor=(0.5, 0.95),
        fontsize=LEGEND_FS,
    )
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)

    out_file = FIG_DIR / f"{group_name}.pdf"
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {out_file}")

if __name__ == "__main__":
    for group_name, target_models in ABLATION_GROUPS.items():
        plot_ablation_group(group_name, target_models)