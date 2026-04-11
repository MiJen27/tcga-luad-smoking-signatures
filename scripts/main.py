"""
Main pipeline for TCGA-LUAD mutational signature analysis
Author: Michal Jendrušák
"""

import os
import sys
import subprocess

SCRIPTS_DIR = ""


def run_matrix_generator():
    print("=== 1. Generating SBS96 matrices from input MAF files ===")
    subprocess.run([sys.executable, os.path.join(SCRIPTS_DIR, "run_matrix_generator.py")], check=True)


def run_matrix_analyzer():
    print("=== 2. Fitting SBS96 profiles to COSMIC signatures ===")
    subprocess.run([sys.executable, os.path.join(SCRIPTS_DIR, "run_matrix_analyzator.py")], check=True)


def create_heatmaps():
    print("=== 3. Creating basic heatmaps ===")
    subprocess.run([sys.executable, os.path.join(SCRIPTS_DIR, "heat_map_creation.py")], check=True)


def merge_clinical_exposure():
    print("=== 4. Merging clinical and exposure data ===")
    subprocess.run([sys.executable, os.path.join(SCRIPTS_DIR, "merge-clinical_exposure- to one tsv.py")], check=True)


def create_heatmaps_clinical():
    print("=== 5. Creating clinical heatmaps and SBS4 plot ===")
    subprocess.run([sys.executable, os.path.join(SCRIPTS_DIR, "heat_map_creation - clinical data.py")], check=True)


def run_supervised_analysis():
    print("=== 6. Running supervised analysis ===")
    subprocess.run([sys.executable, os.path.join(SCRIPTS_DIR, "supervised_random_forest_Balanced_never-ever.py")], check=True)


if __name__ == "__main__":
    run_matrix_generator()
    run_matrix_analyzer()
    create_heatmaps()
    merge_clinical_exposure()
    create_heatmaps_clinical()
    run_supervised_analysis()

    print("\nDONE! The whole pipeline finished successfully.")