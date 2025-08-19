import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BG = "#0B0E11"
PANEL = "#12161C"
TEXT = "#E8E8E8"
MUTED = "#9BA3AF"
ACCENT = "#21CE99"
ACCENT2 = "#3DD6A0"
ACCENT3 = "#1ABC9C"

def _dark_axes(ax):
    ax.set_facecolor(PANEL)
    ax.figure.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1a1f29")
    ax.grid(False)

def donut_series(series: pd.Series, title: str = ""):
    fig, ax = plt.subplots(figsize=(4.8,4.8))
    vals = series.fillna(0.0).values
    colors = [ACCENT, ACCENT2, ACCENT3, "#2ECC71","#3498DB","#9B59B6","#E67E22","#1ABC9C","#16A085","#95A5A6"]
    ax.pie(vals, startangle=140, colors=colors[:len(vals)])
    centre_circle = plt.Circle((0,0),0.62,fc=BG)
    fig.gca().add_artist(centre_circle)
    _dark_axes(ax); ax.set_title(title, color=TEXT, fontsize=11)
    return fig

def heatmap_corr(corr: pd.DataFrame, title: str = "Correlation Heatmap"):
    fig, ax = plt.subplots(figsize=(5.5,4.5))
    im = ax.imshow(corr.values, aspect='auto', cmap='viridis', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=90, color=TEXT, fontsize=8)
    ax.set_yticks(range(len(corr.index))); ax.set_yticklabels(corr.index, color=TEXT, fontsize=8)
    _dark_axes(ax); ax.set_title(title, color=TEXT, fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color=TEXT); 
    return fig

def line_series(df: pd.DataFrame, title: str = ""):
    fig, ax = plt.subplots(figsize=(7,3.4))
    palette = [ACCENT, "#46B3E6", "#9B59B6", "#E67E22", "#95A5A6"]
    for i, col in enumerate(df.columns):
        ax.plot(df.index, df[col], label=col, linewidth=2.0, color=palette[i % len(palette)])
    _dark_axes(ax); ax.legend(facecolor=PANEL, edgecolor="#1a1f29", labelcolor=TEXT); ax.set_title(title, color=TEXT, fontsize=11)
    return fig

def risk_gauge(score: float):
    fig, ax = plt.subplots(figsize=(4.5,2.2), subplot_kw={'projection':'polar'})
    ax.set_theta_zero_location('W'); ax.set_theta_direction(-1)
    ax.set_thetamin(0); ax.set_thetamax(180)
    ax.set_yticklabels([]); ax.set_xticklabels([])
    ax.set_facecolor(BG); fig.patch.set_facecolor(BG)
    import numpy as np
    angles = np.linspace(0, np.pi, 100)
    ax.bar(angles, [1]*len(angles), width=np.pi/100, bottom=0.0, color="#1a1f29")
    theta = (score/100.0)*np.pi
    ax.bar([theta], [1.0], width=0.03*np.pi, bottom=0.0, color=ACCENT)
    ax.text(0, -0.2, f"Risk {score:.1f}", ha='left', va='center', color=TEXT, fontsize=11, transform=ax.transAxes)
    return fig
