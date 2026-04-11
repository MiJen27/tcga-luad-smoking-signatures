"""
Balanced Random Forest (binary) on TCGA-LUAD:
- predictors: SBS96 profile (96 channels) + clinical covariates (age, sex)   [NO pack-years]
- target: Never vs Ever-smoker (Former+Current)
- evaluation (clean workflow, NO OOB):
    * TRAIN: hyperparam tuning by manual grid (fit on TRAIN, score on existing VAL)
    * VAL:   holdout evaluation of selected model
    * TEST:  final holdout evaluation of refit model trained on TRAIN+VAL with best params
- outputs:
    - grid results table + best params
    - classification report + confusion matrix for TRAIN_FIT_SANITY, VAL, TEST_FINAL
    - ROC curve + AUC table for VAL and TEST_FINAL
    - feature importance ranking (all predictors)
    - SBS96-only importance ranking
    - overlap of SBS96 predictor importance with COSMIC SBS4 (Signature_4_GRCh38)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import product
from scipy.stats import spearmanr
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    f1_score, balanced_accuracy_score
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from imblearn.ensemble import BalancedRandomForestClassifier


# ======================
# 0) PATHS
# ======================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root

SBS96_PATH = os.path.join(BASE_DIR, "data", "maf", "output", "SBS", "LUAD.SBS96.all")
CLIN_PATH = os.path.join(BASE_DIR, "data", "clinical_exposure_merged.tsv")
COSMIC_SBS4_PATH = os.path.join(BASE_DIR, "data", "v3.2_SBS4_DIFFERENCE.txt") # from COSMIC page

# output dirs if not exist, will be created
PLOTS_DIR = os.path.join(BASE_DIR, "plots", "brf_split_binary")
OUT_DIR = os.path.join(BASE_DIR, "results", "brf_split_binary")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42 ### potrebujeme random state ???
TOP_N_OVERLAP = 20  ### TODO: automaticky ?

print("\n Balanced RF BINARY (Never vs Ever) + TRAIN/VAL/TEST + SBS4 overlap analysis spúšťam...\n")

# ======================
# 1) LOAD DATA
# ======================
if not os.path.exists(SBS96_PATH):
    raise FileNotFoundError(f"❌ SBS96 matrix sa nenašla: {SBS96_PATH}")
if not os.path.exists(CLIN_PATH):
    raise FileNotFoundError(f"❌ Clinical TSV sa nenašiel: {CLIN_PATH}")
if not os.path.exists(COSMIC_SBS4_PATH):
    raise FileNotFoundError(f"❌ COSMIC SBS4 file sa nenašiel: {COSMIC_SBS4_PATH}")

sbs = pd.read_csv(SBS96_PATH, sep="\t", index_col=0)
clin = pd.read_csv(CLIN_PATH, sep="\t")

print("Raw SBS shape:", sbs.shape)
print("Raw clinical shape:", clin.shape)

# orientácia SBS: chceme (samples x 96)
if sbs.shape[1] != 96 and sbs.shape[0] == 96:
    sbs = sbs.T

if sbs.shape[1] != 96:
    raise RuntimeError(f"❌ SBS96 nie je so 96 stĺpcami, ale shape je {sbs.shape}")

#check print
print("Final SBS shape:", sbs.shape)

sbs_cols = list(sbs.columns)

print(f'TU JE SBS_COLS ===============> {sbs_cols}')
#for c in sbs_cols:
#    sbs[c] = pd.to_numeric(sbs[c], errors="coerce").fillna(0.0)

# Patient_ID
sbs = sbs.copy()
sbs["Patient_ID"] = sbs.index.astype(str).str[:12] # first 12 chars of index TCGA patient barcode

clin = clin.copy()
#if "cases.submitter_id" not in clin.columns:
#    raise KeyError("❌ V clinical súbore chýba stĺpec 'cases.submitter_id'.")

clin["Patient_ID"] = clin["cases.submitter_id"].astype(str).str[:12] # TODO clin data has only 12 chars of patient barcode, therefore we need to use this, just in case it is not string

merged = pd.merge(
    sbs,
    clin,
    on="Patient_ID"
)

print("Merged N (after inner join):", len(merged))

#if len(merged) == 0:
 #   raise RuntimeError("❌ Merge dal 0 riadkov. Skontroluj Patient_ID mapping.")

# ======================
# 2) TARGET: binary (Never vs Ever)
# ======================
list_unknown_smoking_status = []
def map_smoking_3class(x):
    if pd.isna(x) or str(x).strip() == "": # missing or empty string
        list_unknown_smoking_status.append(x)
        return "Unknown"

    x = str(x).strip().lower()

    if "lifelong" in x or "non-smoker" in x or "nonsmoker" in x or "never" in x:
        return "Never"

    if "current reformed" in x or "reformed" in x or "former" in x: # we have to track, Former group before Current group, as "curent reformed" also includes "curent"
        return "Former"

    if "current" in x:
        return "Current"

    list_unknown_smoking_status.append(x)
    return "Unknown"

merged["Smoking_3"] = merged["exposures.tobacco_smoking_status"].map(map_smoking_3class) # we create new column with mapped values of smoking status by 3 categories: Never, Former, Current
print(f"Unknown smoking status: {list_unknown_smoking_status} and count {len(list_unknown_smoking_status)}")

print("\nSmoking_3 counts BEFORE filtering:")
print(merged["Smoking_3"].value_counts(dropna=False))

n_before = len(merged)
merged = merged[merged["Smoking_3"].isin(["Never", "Former", "Current"])].copy()
n_after = len(merged)

print(f"\nRemoved due to unknown smoking status: {n_before - n_after}") ## we remove all rows with unknown smoking status - it was 14 rows in total 'nan'

# Binary label: Never=0, Ever=1
merged["Smoking_Bin"] = (merged["Smoking_3"] != "Never").astype(int) # we create new column with binary label: 0=Never, 1=Ever

print("\nBinary counts (0=Never, 1=Ever):")
print(merged["Smoking_Bin"].value_counts())

#if len(merged) == 0:
#    raise RuntimeError("❌ Po mapovaní smoking kategórií je N=0.")


# ======================
# 3) SPLIT: PATIENT-LEVEL STRATIFIED TRAIN/VAL/TEST
# ======================

# saniity check -> aby nedoslo k data leakage pri splitovani, pozerame na pacientov nie na zaznamy
patient_labels = (
    merged[["Patient_ID", "Smoking_Bin"]]
    .drop_duplicates("Patient_ID") # aby nedoslo k data leakage pri splitovani
    .reset_index(drop=True) # TODO potrebujeme to ?
)

print("\nUnique patients after smoking filtering:", len(patient_labels))
print("Patient-level class counts:")
print(patient_labels["Smoking_Bin"].value_counts())

trainval_p, test_p = train_test_split(
    patient_labels,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=patient_labels["Smoking_Bin"] # to keep the same distribution of classes in (train/val) and test
)

train_p, val_p = train_test_split(
    trainval_p,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=trainval_p["Smoking_Bin"] # to keep the same distribution of classes in train/ val
)

# we save patient ids in sets
train_ids = set(train_p["Patient_ID"])
val_ids = set(val_p["Patient_ID"])
test_ids = set(test_p["Patient_ID"])

# sanity check: no leakage
if train_ids & val_ids:
    raise ValueError("Leakage medzi TRAIN a VAL")
if train_ids & test_ids:
    raise ValueError("Leakage medzi TRAIN a TEST")
if val_ids & test_ids:
    raise ValueError("Leakage medzi VAL a TEST")

train_df = merged[merged["Patient_ID"].isin(train_ids)].copy()
val_df = merged[merged["Patient_ID"].isin(val_ids)].copy()
test_df = merged[merged["Patient_ID"].isin(test_ids)].copy()

#TODO asi netreba
train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
test_df.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

print(
    "\nSplit sizes (rows):",
    f"TRAIN={len(train_df)} | VAL={len(val_df)} | TEST={len(test_df)}"
)

print(
    "Split sizes (unique patients):",
    f"TRAIN={train_df['Patient_ID'].nunique()} | "
    f"VAL={val_df['Patient_ID'].nunique()} | "
    f"TEST={test_df['Patient_ID'].nunique()}"
)

#######################
# Check whether any patient has inconsistent smoking labels across rows
smoking_consistency = (
    merged.groupby("Patient_ID")
    .agg(
        n_rows=("Patient_ID", "size"),
        n_raw_status=("exposures.tobacco_smoking_status", "nunique"),
        n_smoking3=("Smoking_3", "nunique"),
        n_smoking_bin=("Smoking_Bin", "nunique"),
    )
    .reset_index()
)

conflicting_patients = smoking_consistency[
    (smoking_consistency["n_raw_status"] > 1) |
    (smoking_consistency["n_smoking3"] > 1) |
    (smoking_consistency["n_smoking_bin"] > 1)
].copy()

print("\nPatients with duplicated rows:", (smoking_consistency["n_rows"] > 1).sum())
print("Patients with conflicting smoking annotations:", len(conflicting_patients))

if len(conflicting_patients) > 0:
    print("\nConflicting patients:")
    print(conflicting_patients.to_string(index=False))
##########################


# ======================
# 4) FEATURES: SBS96 proportions + age + gender (train-impute only)
# ======================
def prep_features(df, age_median=None, gender_fill_value=None):
    df = df.copy()

    # number of missing age values
    print(f'kolko je missing age ===> {df["demographic.age_at_index"].isna().sum()}')

    # convert age to numeric; non-numeric values (e.g. 'unknown') are set to NaN
    df["demographic.age_at_index"] = pd.to_numeric(
        df["demographic.age_at_index"],
        errors="coerce"      #toto nebude treba, data su konzistentne ??
    )

#   for x in df["demographic.age_at_index"]:
#       if x is None or x == 'nan' or x == 'unknown' or x == 'Unknown' or x == 'nan':
#           print(f'taketo su {x}')

    print(f'kolko je missing gender ===> {df["demographic.gender"].isna().sum()}')
    # gender -> binary
    df["demographic.gender"] = (
        df["demographic.gender"]
        .astype(str)
        .str.lower()
        .map({"male": 1, "female": 0})
    )

    # fit imputation values on TRAIN only
    if age_median is None:
        age_median = df["demographic.age_at_index"].median()

    if gender_fill_value is None:
        gender_fill_value = df["demographic.gender"].median()

    # fallback ak by bolo všetko missing --- >>> netreba mame valid data !!
   # if pd.isna(age_median):
    #    age_median = 0.0
    #if pd.isna(gender_fill_value):
     #   gender_fill_value = 0.5

    df["demographic.age_at_index"] = df["demographic.age_at_index"].fillna(age_median)
    df["demographic.gender"] = df["demographic.gender"].fillna(gender_fill_value) ## toto nebude treba, data su konzistentne ?? ani raz sme nedostali nan pre gender

    df["age_years"] = df["demographic.age_at_index"] # change column name

    clinical_cols = ["age_years", "demographic.gender"]

    # --- SBS96: extract mutation counts ---
    X_sbs = df[sbs_cols].copy()

    # --- compute total number of mutations per sample (row-wise sum) ---
    row_sum = X_sbs.sum(axis=1)

    # --- normalize counts to proportions ---
    # divide each SBS channel by the total number of mutations in that sample
    X_sbs = X_sbs.div(
        row_sum.replace(0, np.nan),  # avoid division by zero
        axis=0
    )

    # --- replace NaN values (e.g. samples with zero mutations) ---
    X_sbs = X_sbs.fillna(0.0)

    # --- combine SBS features with clinical variables ---
    X = pd.concat([X_sbs, df[clinical_cols]], axis=1) # axis aby sme joinly tie 2 stlpce dokopy 96 SBS + 2 clinical
    y = df["Smoking_Bin"].values # y = [1, 0, 1, 1, 0, ...]

    return X, y, age_median, gender_fill_value


X_train, y_train, age_med, gender_fill = prep_features(train_df)
X_val, y_val, _, _ = prep_features(val_df, age_median=age_med, gender_fill_value=gender_fill)
X_test, y_test, _, _ = prep_features(test_df, age_median=age_med, gender_fill_value=gender_fill)

print("\nFeatures:", X_train.shape[1], "(SBS96=96 + clinical=2)")
print("Age median used for imputation (TRAIN):", age_med)
print("Gender fill value used for imputation (TRAIN):", gender_fill) ## nebude treba ?


# ======================
# 5) GRID SEARCH using existing VAL (NO CV, NO OOB)
# ======================
# hyperparameter grid for BalancedRandomForestClassifier, spolu teda 3 x 1 x 2 = 6 modelov
param_grid = {
    "n_estimators": [200, 400, 800],
    "max_features": ["sqrt"],
    "max_depth": [None, 20],
}

keys = list(param_grid.keys())
values = [param_grid[k] for k in keys]

best_score = -1.0
best_params = None
best_model = None
results = []

print("\n🔎 Grid search BalancedRF (fit TRAIN → score VAL) ...\n")

# generation of all possible combinations of hyperparameter values
for combination in product(*values):
    params = dict(zip(keys, combination))

    brf = BalancedRandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        sampling_strategy="auto",
        replacement=False,
        bootstrap=True,
        **params
    )

    brf.fit(X_train, y_train) # fit on TRAIN only, model sa nauci X-> Y
    val_pred = brf.predict(X_val) # predict class labels on VAL set
    val_proba = brf.predict_proba(X_val)[:, 1] # prvy stlpec = P(class 0) a druhy stlpec = P(class 1) 1=Ever, 0=Never, my berieme P(Ever) -> ROC krivka pre pozitivnu triedu, P(Never) si vieme dopocitat P(0) = 1 - P(1) (spolu je to 1 vzdy)

    # evaluate model performance
    score_f1 = f1_score(y_val, val_pred, average="macro") # kombinuje precision a recall, vrati hodnotu v rozsahu [0,1], macro znamena ze sa pocita pre kazdu triedu a potom priemeruje
    score_bal_acc = balanced_accuracy_score(y_val, val_pred) # priemer recall pre kazdu triedu

    # compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_val, val_proba)
    score_auc = auc(fpr, tpr)

    # store results for this parameter combination
    results.append({
        **params,
        "val_f1_macro": score_f1,
        "val_balanced_accuracy": score_bal_acc,
        "val_auc": score_auc
    })

    #update best model based on F1 score
    if score_f1 > best_score:
        best_score = score_f1
        best_params = params
        best_model = brf

print("✅ Best VAL score (macro-F1):", best_score)
print("✅ Best params:", best_params)

'''
pd.Series(best_params).to_csv(
    os.path.join(OUT_DIR, "best_params.tsv"),
    sep="\t",
    header=False
)

pd.DataFrame(results).sort_values(
    "val_f1_macro",
    ascending=False
).to_csv(
    os.path.join(OUT_DIR, "gridsearch_val_results.tsv"),
    sep="\t",
    index=False
)
'''

brf_selected = best_model


# ======================
# 6) HELPERS: report + CM + ROC/AUC
# ======================
def save_report_and_cm_binary(y_true, y_pred, tag):
    acc = (y_pred == y_true).mean()
    bal_acc = balanced_accuracy_score(y_true, y_pred) # balaced accurancy berie recall pre never a ever a spriemeruje ich

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Never", "Ever"],
        zero_division=0
    )

    print(f"\n[{tag}] accuracy={acc:.4f} | balanced_accuracy={bal_acc:.4f}\n")
    print(report)

    out_txt = os.path.join(OUT_DIR, f"{tag}_report.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"accuracy: {acc:.4f}\n")
        f.write(f"balanced_accuracy: {bal_acc:.4f}\n\n")
        f.write(report)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Never", "Ever"],
        yticklabels=["Never", "Ever"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{tag} Confusion Matrix (acc={acc:.3f}, bal={bal_acc:.3f})")
    plt.tight_layout()

    cm_path = os.path.join(PLOTS_DIR, f"{tag}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    print("Uložené:", out_txt)
    print("Uložené:", cm_path)


def plot_binary_roc(y_true, proba_pos, tag):
    fpr, tpr, _ = roc_curve(y_true, proba_pos)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.6f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{tag} ROC (Ever vs Never)")
    plt.legend(loc="lower right")
    plt.tight_layout()

    out_png = os.path.join(PLOTS_DIR, f"{tag}_roc.png")
    plt.savefig(out_png, dpi=300)
    plt.close()

    auc_out = os.path.join(OUT_DIR, f"{tag}_auc.tsv")
    pd.DataFrame([{"auc": roc_auc}]).to_csv(auc_out, sep="\t", index=False)

    print("Uložené:", out_png)
    print("Uložené:", auc_out)
    print(f"[{tag}] AUC: {roc_auc:.6f}")


# ======================
# 7) TRAIN sanity (fit on TRAIN, just for sanity)
# ======================
train_pred = brf_selected.predict(X_train)
save_report_and_cm_binary(y_train, train_pred, tag="TRAIN_FIT_SANITY")


# ======================
# 8) VAL evaluation (selected model)
# ======================
val_pred = brf_selected.predict(X_val)
save_report_and_cm_binary(y_val, val_pred, tag="VAL")

val_proba = brf_selected.predict_proba(X_val)[:, 1]  # P(Ever)
plot_binary_roc(y_val, val_proba, tag="VAL")


# ======================
# 9) FINAL MODEL: refit on TRAIN+VAL, evaluate on TEST
# ======================
X_trainval = pd.concat([X_train, X_val], axis=0)
y_trainval = np.concatenate([y_train, y_val], axis=0)

brf_final = BalancedRandomForestClassifier(
    random_state=RANDOM_STATE,
    n_jobs=-1,
    sampling_strategy="auto",
    replacement=False,
    bootstrap=True,
    **best_params
)

brf_final.fit(X_trainval, y_trainval)

test_pred = brf_final.predict(X_test)
save_report_and_cm_binary(y_test, test_pred, tag="TEST_FINAL")

test_proba = brf_final.predict_proba(X_test)[:, 1]
plot_binary_roc(y_test, test_proba, tag="TEST_FINAL")


# ======================
# 10) FEATURE IMPORTANCE (FINAL MODEL)          TODO PRIDAT PEMURATION IMPORTANCE     ->>> hotovo
# ======================
# --- 1) Standard Random Forest feature importance (impurity-based) ---
fi = pd.Series(
    brf_final.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

# save full ranking
fi.to_csv(
    os.path.join(OUT_DIR, "feature_importance_final_model.tsv"),
    sep="\t",
    header=["importance"]
)

# plot top 30 features
top30 = fi.head(30)

plt.figure(figsize=(10, 9))
sns.barplot(x=top30.values, y=top30.index)

plt.title("Top 30 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()

fi_path = os.path.join(PLOTS_DIR, "feature_importance_top30_final.png")
plt.savefig(fi_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved:", fi_path)


# --- 2) Permutation importance (more interpretable) ---
from sklearn.inspection import permutation_importance

perm = permutation_importance(
    brf_final,
    X_test,
    y_test,
    n_repeats=10,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# convert to pandas Series
perm_importance = pd.Series(
    perm.importances_mean,
    index=X_test.columns
).sort_values(ascending=False)

# save permutation importance
perm_importance.to_csv(
    os.path.join(OUT_DIR, "feature_importance_permutation.tsv"),
    sep="\t",
    header=["importance"]
)

# plot top 30 permutation features
top30_perm = perm_importance.head(30)

plt.figure(figsize=(10, 9))
sns.barplot(x=top30_perm.values, y=top30_perm.index)

plt.title("Top 30 Feature Importances (Permutation)")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()

perm_path = os.path.join(PLOTS_DIR, "feature_importance_top30_permutation.png")
plt.savefig(perm_path, dpi=300)
plt.close()

print("Saved:", perm_path)

# ======================
# 11) SBS96 ranking + overlap with COSMIC SBS4
# ======================
fi_sbs = fi.reindex(sbs_cols).dropna().sort_values(ascending=False)

fi_sbs.to_csv(
    os.path.join(OUT_DIR, "feature_importance_sbs96_only.tsv"),
    sep="\t",
    header=["importance"]
)

top20_sbs = fi_sbs.head(20)

plt.figure(figsize=(10, 8))
sns.barplot(x=top20_sbs.values, y=top20_sbs.index)
plt.title("Top 20 SBS96 Feature Importances (Binary BalancedRF)")
plt.xlabel("Importance")
plt.ylabel("SBS96 channel")
plt.tight_layout()

top20_sbs_plot = os.path.join(PLOTS_DIR, "feature_importance_sbs96_top20.png")
plt.savefig(top20_sbs_plot, dpi=300)
plt.close()

print("Uložené:", top20_sbs_plot)

# načítanie COSMIC SBS4 súboru
cosmic = pd.read_csv(COSMIC_SBS4_PATH, sep="\t", index_col=0)

if "Signature_4_GRCh38" not in cosmic.columns:
    raise KeyError("❌ V COSMIC súbore chýba stĺpec 'Signature_4_GRCh38'.")

cosmic_sbs4 = cosmic["Signature_4_GRCh38"].copy()
cosmic_sbs4.index = cosmic_sbs4.index.astype(str)

# zoradenie podľa poradia SBS96 kanálov v tvojej matici
cosmic_sbs4 = cosmic_sbs4.reindex(sbs_cols)

if cosmic_sbs4.isna().any():
    missing_channels = cosmic_sbs4[cosmic_sbs4.isna()].index.tolist()
    raise RuntimeError(f"❌ Missing SBS4 values for channels: {missing_channels}")

# normalizácia pre porovnanie
imp_vec = fi_sbs.reindex(sbs_cols).astype(float)
imp_vec = imp_vec / imp_vec.sum()

sbs4_vec = pd.to_numeric(cosmic_sbs4, errors="coerce").astype(float)
sbs4_vec = sbs4_vec / sbs4_vec.sum()

if sbs4_vec.isna().any():
    bad_channels = sbs4_vec[sbs4_vec.isna()].index.tolist()
    raise RuntimeError(f"❌ Non-numeric / missing SBS4 values for channels: {bad_channels}")

# ranky
imp_rank = imp_vec.rank(ascending=False, method="min").astype(int)
sbs4_rank = sbs4_vec.rank(ascending=False, method="min").astype(int)

comparison_df = pd.DataFrame({
    "channel": sbs_cols,
    "importance": imp_vec.values,
    "importance_rank": imp_rank.values,
    "SBS4_weight": sbs4_vec.values,
    "SBS4_rank": sbs4_rank.values
}).sort_values("importance", ascending=False)

comparison_df.to_csv(
    os.path.join(OUT_DIR, "sbs96_importance_vs_SBS4.tsv"),
    sep="\t",
    index=False
)

# top-N overlap
top_imp = set(imp_vec.sort_values(ascending=False).head(TOP_N_OVERLAP).index)
top_sbs4 = set(sbs4_vec.sort_values(ascending=False).head(TOP_N_OVERLAP).index)
overlap = sorted(top_imp & top_sbs4)

pd.DataFrame({"overlap_channel": overlap}).to_csv(
    os.path.join(OUT_DIR, f"sbs4_overlap_top{TOP_N_OVERLAP}_channels.tsv"),
    sep="\t",
    index=False
)

# podobnosť cez všetkých 96 kanálov
spearman_r, spearman_p = spearmanr(imp_vec.values, sbs4_vec.values)
cos_sim = cosine_similarity(
    imp_vec.values.reshape(1, -1),
    sbs4_vec.values.reshape(1, -1)
)[0, 0]

metrics_df = pd.DataFrame([{
    "top_n": TOP_N_OVERLAP,
    "n_overlap": len(overlap),
    "spearman_r": spearman_r,
    "spearman_p": spearman_p,
    "cosine_similarity": cos_sim
}])

metrics_df.to_csv(
    os.path.join(OUT_DIR, "sbs4_overlap_metrics.tsv"),
    sep="\t",
    index=False
)

print("\nSBS96 vs COSMIC SBS4 comparison done.")
print(metrics_df.to_string(index=False))
print("Top overlap channels:", overlap)

print("\n✅ Hotovo.\n")