"""
Heatmaps of fitted mutational signature activities by smoking category (TCGA-LUAD)
"""

import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ACTIVITIES_PATH = os.path.join(
    BASE_DIR, "results", "LUAD_sig_output", "Assignment_Solution",
    "Activities", "Assignment_Solution_Activities.txt"
)

CLINICAL_PATH = os.path.join(BASE_DIR, "data", "clinical_exposure_merged.tsv")

OUTPUT_DIR = os.path.join(BASE_DIR, "plots", "supervised_heatmaps")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------
# Load data
# ----------------------
sig_df = pd.read_csv(ACTIVITIES_PATH, sep="\t", index_col=0)
clin_df = pd.read_csv(CLINICAL_PATH, sep="\t")

print("Raw signature shape:", sig_df.shape)
print("Raw clinical shape:", clin_df.shape)

# ----------------------
# Keep only SBS columns + sanitize numeric values
# ----------------------
sig_cols = [col for col in sig_df.columns if str(col).startswith("SBS") and str(col)[3:].isdigit()]
sig_cols = sorted(sig_cols, key=lambda x: int(x[3:]))

if len(sig_cols) == 0:
    raise RuntimeError("No SBS columns found in activities file.")

sig_df = sig_df[sig_cols].copy()
sig_df[sig_cols] = sig_df[sig_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

# ----------------------
# Normalize per sample
# ----------------------
row_sums = sig_df.sum(axis=1)
print("Samples with zero total activity:", (row_sums == 0).sum())

sig_df = sig_df[row_sums > 0].copy()
sig_df = sig_df.div(sig_df.sum(axis=1), axis=0)

# ----------------------
# Patient ID extraction
# ----------------------
tcga_pat = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})")

def extract_patient_id(x):
    m = tcga_pat.search(str(x))
    return m.group(1) if m else np.nan

sig_df["Patient_ID"] = sig_df.index.map(extract_patient_id)
clin_df["Patient_ID"] = clin_df["cases.submitter_id"].map(extract_patient_id)

sig_df = sig_df.dropna(subset=["Patient_ID"])
clin_df = clin_df.dropna(subset=["Patient_ID"])

# ----------------------
# Merge
# ----------------------
merged_df = pd.merge(
    sig_df,
    clin_df,
    on="Patient_ID",
    how="inner",
    validate="many_to_one"
)

print("Merged rows:", len(merged_df))

# ----------------------
# Smoking mapping
# ----------------------
def map_smoking(x):
    if pd.isna(x) or str(x).strip().lower() in ["nan", "not reported", "unknown", ""]:
        return "Unknown"

    x = str(x).strip().lower()

    if "never" in x or "lifelong" in x or "non-smoker" in x or "nonsmoker" in x:
        return "Never"
    if "former" in x or "reformed" in x:
        return "Former"
    if "current" in x:
        return "Current"
    return "Unknown"

merged_df["Smoking_Category"] = merged_df["exposures.tobacco_smoking_status"].map(map_smoking)

print("\nSmoking counts before filtering:")
print(merged_df["Smoking_Category"].value_counts(dropna=False))

n_before = len(merged_df)
merged_df = merged_df[merged_df["Smoking_Category"].isin(["Never", "Former", "Current"])].copy()
print("Removed due to missing/unknown smoking status:", n_before - len(merged_df))

# ----------------------
# NEW: 1 patient = 1 row
# ----------------------
merged_df = (
    merged_df
    .groupby(["Patient_ID", "Smoking_Category"], as_index=False)[sig_cols]
    .mean()
)

print("Rows after patient-level aggregation:", len(merged_df))

df_clean = merged_df.copy()
df_clean = df_clean.dropna(subset=["SBS4"])

# ----------------------
# Heatmaps
# ----------------------
for category in ["Never", "Former", "Current"]:
    subset = merged_df[merged_df["Smoking_Category"] == category]
    if subset.empty:
        continue

    heatmap_data = subset[sig_cols].T

    # voliteľné: nech tam nezostanú úplne nulové stĺpce
    non_zero_cols = heatmap_data.columns[heatmap_data.sum(axis=0) > 0]
    if len(non_zero_cols) > 0:
        heatmap_data = heatmap_data[non_zero_cols]

    plt.figure(figsize=(16, 8))
    ax = sns.heatmap(
        heatmap_data,
        cmap="YlOrRd",
        linewidths=0.3,
        linecolor="gray",
        cbar_kws={"label": "Signature Contribution"},
        yticklabels=True,
        xticklabels=False
    )

    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.title(f"Mutational Signatures - {category} Smokers (n={heatmap_data.shape[1]})", fontsize=14)
    plt.ylabel("Mutational Signatures")
    plt.xlabel("Patients")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"heatmap_{category}.png"), dpi=300, bbox_inches="tight")
    plt.close()

# ----------------------
# SBS4 boxplot + stripplot
# ----------------------
print("Creating SBS4 boxplot + stripplot...")

plt.figure(figsize=(10, 6))

order = ["Never", "Former", "Current"]
palette = {
    "Never": "lightgreen",
    "Former": "orange",
    "Current": "red"
}

sns.boxplot(
    data=df_clean,
    x="Smoking_Category",
    y="SBS4",
    hue="Smoking_Category",
    order=order,
    palette=palette,
    width=0.6,
    fliersize=0,
    dodge=False,
    legend=False
)

sns.stripplot(
    data=df_clean,
    x="Smoking_Category",
    y="SBS4",
    order=order,
    jitter=0.22,
    size=2.5,
    alpha=0.6,
    color="black"
)

plt.title("SBS4 Contribution by Smoking Category", fontsize=14)
plt.xlabel("Smoking Category")
plt.ylabel("SBS4 Contribution")
plt.tight_layout()

out_boxplot = os.path.join(BASE_DIR, "plots", "SBS4_boxplot_stripplot_clean.png")
plt.savefig(out_boxplot, dpi=300, bbox_inches="tight")
plt.close()

print("Saved:", out_boxplot)

print("Done.")