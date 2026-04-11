'''
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
clin_path = os.path.join(BASE_DIR, "data", "clinical.tsv")
expos_path = os.path.join(BASE_DIR, "data", "exposure.tsv")
output_path = os.path.join(BASE_DIR, "data", "clinical_exposure_merged.tsv")

clin = pd.read_csv(clin_path, sep="\t")
expos = pd.read_csv(expos_path, sep="\t")

clin_subset = clin[[
    "cases.submitter_id",
    "demographic.age_at_index",
    "demographic.gender",
    "demographic.race",
    "demographic.ethnicity"
]].copy()

expos_subset = expos[[
    "cases.submitter_id",
    "exposures.tobacco_smoking_status",
    "exposures.pack_years_smoked",
    "exposures.cigarettes_per_day"
]].copy()

# sprav 1 riadok na pacienta (klinika má duplicity kvôli treatments atď.)
clin_subset = clin_subset.drop_duplicates(subset=["cases.submitter_id"], keep="first")
#expos_subset = expos_subset.drop_duplicates(subset=["cases.submitter_id"], keep="first")

merged = pd.merge(clin_subset, expos_subset, on="cases.submitter_id", how="left")

print("Merged rows:", len(merged), "| unique patients:", merged["cases.submitter_id"].nunique())
merged.to_csv(output_path, sep="\t", index=False)

print(f"Spojené dáta uložené do: {output_path}")


'''
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
clin_path = os.path.join(BASE_DIR, "data", "clinical.tsv")
expos_path = os.path.join(BASE_DIR, "data", "exposure.tsv")
output_path = os.path.join(BASE_DIR, "data", "clinical_exposure_merged.tsv")

NA_TOKENS = ["'--", "--", "Not Reported", "not reported", "Unknown", "unknown", ""]

clin = pd.read_csv(clin_path, sep="\t", na_values=NA_TOKENS, keep_default_na=True)
expos = pd.read_csv(expos_path, sep="\t", na_values=NA_TOKENS, keep_default_na=True)

clin_subset = clin[[
    "cases.submitter_id",
    "demographic.age_at_index",
    "demographic.gender",
    "demographic.race",
    "demographic.ethnicity"
]].copy()

expos_subset = expos[[
    "cases.submitter_id",
    "exposures.tobacco_smoking_status",
    "exposures.pack_years_smoked",
    "exposures.cigarettes_per_day"
]].copy()

# odstráň riadky bez patient ID
clin_subset = clin_subset.dropna(subset=["cases.submitter_id"]).copy()
expos_subset = expos_subset.dropna(subset=["cases.submitter_id"]).copy()

# odstráň úplne prázdne klinické záznamy
clin_info_cols = [
    "demographic.age_at_index",
    "demographic.gender",
    "demographic.race",
    "demographic.ethnicity"
]
clin_subset = clin_subset[~clin_subset[clin_info_cols].isna().all(axis=1)].copy()

# odstráň úplne prázdne exposure záznamy
expos_info_cols = [
    "exposures.tobacco_smoking_status",
    "exposures.pack_years_smoked",
    "exposures.cigarettes_per_day"
]
expos_subset = expos_subset[~expos_subset[expos_info_cols].isna().all(axis=1)].copy()

# 1 riadok na pacienta v klinike
clin_dups_before = clin_subset["cases.submitter_id"].duplicated().sum()
clin_subset = clin_subset.drop_duplicates(subset=["cases.submitter_id"], keep="first")

# 1 riadok na pacienta aj v exposure
expos_dups_before = expos_subset["cases.submitter_id"].duplicated().sum()
expos_subset = expos_subset.drop_duplicates(subset=["cases.submitter_id"], keep="first")

merged = pd.merge(
    clin_subset,
    expos_subset,
    on="cases.submitter_id",
    how="left",
    validate="one_to_one"
)

print("Clinical duplicates removed:", clin_dups_before)
print("Exposure duplicates removed:", expos_dups_before)
print("Merged rows:", len(merged))
print("Unique patients:", merged["cases.submitter_id"].nunique())

print("\nMissing values after merge:")
print(merged.isna().sum())

print("\nPatients with smoking annotation:")
print(merged["exposures.tobacco_smoking_status"].notna().sum())

merged.to_csv(output_path, sep="\t", index=False)
print(f"\nSpojené dáta uložené do: {output_path}")
