import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS_PATH = os.path.join(
    BASE_DIR,
    "results",
    "LUAD_sig_output",
    "Assignment_Solution",
    "Activities",
    "Assignment_Solution_Activities.txt"
)

PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

if not os.path.exists(RESULTS_PATH):
    raise FileNotFoundError("Activities file not found: " + RESULTS_PATH)

print("Loading:", RESULTS_PATH)


#####
df = pd.read_csv(RESULTS_PATH, sep="\t", index_col=0)

sbs_cols = [c for c in df.columns if c.startswith("SBS") and c[3:].isdigit()]
df = df[sbs_cols].copy()

df = df[sorted(df.columns, key=lambda x: int(x[3:]))]

# remove rows with zero sum
row_sums = df.sum(axis=1)
df = df[row_sums > 0]

df = df.div(df.sum(axis=1), axis=0)


## show fewer x tick labels
def set_spaced_xticks(ax, labels, step=3):
    ax.set_xticks(np.arange(len(labels)) + 0.5)
    ax.set_xticklabels(labels, rotation=90)
    for i, t in enumerate(ax.get_xticklabels()):
        if i % step != 0:
            t.set_visible(False)


# 1) Grouped heatmap (groups of 50 samples)
group_size = 50
group_means = []

for start in range(0, len(df), group_size):
    end = min(start + group_size, len(df))
    group_name = f"Samples {start+1}-{end}"
    group_means.append(pd.Series(df.iloc[start:end].mean(axis=0), name=group_name))

grouped_df = pd.DataFrame(group_means)

plt.figure(figsize=(16, 8))
ax = sns.heatmap(grouped_df, cmap="Blues", linewidths=0.2, linecolor="gray", vmin=0, vmax=1)
set_spaced_xticks(ax, grouped_df.columns, step=3)
ax.set_title("Signature contributions (grouped samples)")
plt.tight_layout()

out1 = os.path.join(PLOTS_DIR, "heatmap_grouped_samples.png")
plt.savefig(out1, dpi=300)
plt.close()
print("Saved:", out1)


# 2) Top 50 samples heatmap
#    (by SBS4 if available, otherwise by max signature contribution)
top_df = df.copy()

if "SBS4" in top_df.columns:
    top_df = top_df.sort_values(by="SBS4", ascending=False).head(50)
else:
    top_df["MaxSig"] = top_df.max(axis=1)
    top_df = top_df.sort_values(by="MaxSig", ascending=False).drop(columns=["MaxSig"]).head(50)

plt.figure(figsize=(16, 10))
ax = sns.heatmap(top_df, cmap="Blues", linewidths=0.2, linecolor="gray", vmin=0, vmax=1)
set_spaced_xticks(ax, top_df.columns, step=3)
ax.set_title("Top 50 samples (highest SBS4 / dominant signature)")
plt.tight_layout()

out2 = os.path.join(PLOTS_DIR, "heatmap_top50_samples.png")
plt.savefig(out2, dpi=300)
plt.close()
print("Saved:", out2)
