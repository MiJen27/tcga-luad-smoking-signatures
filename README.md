# TCGA-LUAD pipeline notebooks

This repository contains the notebook-based pipeline used for the analysis of TCGA-LUAD data.

## Notebook order

1. `01_setup_environment.ipynb`
2. `02_sigprofiler_generation_and_fitting.ipynb`
3. `03_merge_clinical_and_exposure.ipynb`
4. `04_unsupervised_signature_plots.ipynb`
5. `05_supervised_data_preparation.ipynb`
6. `06_supervised_balanced_random_forest.ipynb`

## Description

The workflow is divided into separate notebooks so that each part of the analysis can be run and reviewed independently.

- `01_setup_environment.ipynb` prepares the working environment and required resources.
- `02_sigprofiler_generation_and_fitting.ipynb` generates mutational matrices and performs COSMIC signature fitting.
- `03_merge_clinical_and_exposure.ipynb` merges clinical and smoking-related metadata into a single table.
- `04_unsupervised_signature_plots.ipynb` creates exploratory plots for the mutational signature data.
- `05_supervised_data_preparation.ipynb` prepares the dataset for supervised classification.
- `06_supervised_balanced_random_forest.ipynb` trains and evaluates the Balanced Random Forest model.

