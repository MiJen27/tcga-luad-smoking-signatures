# Split notebooks for the TCGA-LUAD pipeline

Order:
1. 01_setup_environment.ipynb
2. 02_sigprofiler_generation_and_fitting.ipynb
3. 03_merge_clinical_and_exposure.ipynb
4. 04_unsupervised_signature_plots.ipynb
5. 05_supervised_data_preparation.ipynb
6. 06_supervised_balanced_random_forest.ipynb

Main changes:
- setup based on requirements.txt and GRCh38 download
- smaller blocks
- more previews and checks
- direct plot display after each plotting block
- separate notebook for supervised split creation and model training
- clearer variable names without changing the main logic
