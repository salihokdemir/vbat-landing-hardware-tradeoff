"""Reproduce a small subset of final summary figures from processed CSV files.
Run from the repository root:
    python analysis/reproduce_summary_figures.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "strict_common_set"
OUT = ROOT / "figures_reproduced"
OUT.mkdir(exist_ok=True)

ratios = pd.read_csv(RESULTS / "tables" / "final_component_ratios.csv")
rank = pd.read_csv(RESULTS / "tables" / "final_frequency_ranking_slide.csv")

# Component heatmap from ratio table
metric_cols = [c for c in ratios.columns if c not in {"frequency_hz", "frequency_label"}]
fig, ax = plt.subplots(figsize=(7.0, 3.6))
mat = ratios[metric_cols].to_numpy()
im = ax.imshow(mat, aspect="auto")
ax.set_yticks(range(len(ratios)))
ax.set_yticklabels(ratios["frequency_hz"].astype(str) + " Hz")
ax.set_xticks(range(len(metric_cols)))
ax.set_xticklabels(metric_cols, rotation=30, ha="right")
ax.set_title("Component-wise scenario-relative hardware ratios")
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im, ax=ax, label="ratio, lower is better")
fig.tight_layout()
fig.savefig(OUT / "component_ratio_heatmap_reproduced.png", dpi=200)
plt.close(fig)

# Ranking bar
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.bar(rank["Frequency"].astype(str), rank["Score"])
ax.set_xlabel("Frequency")
ax.set_ylabel("Hardware burden score, lower is better")
ax.set_title("Final selected-frequency ranking")
fig.tight_layout()
fig.savefig(OUT / "frequency_score_reproduced.png", dpi=200)
plt.close(fig)
print(f"Wrote figures to {OUT}")
