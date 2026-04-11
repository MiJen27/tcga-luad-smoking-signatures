# TCGA-LUAD Smoking Signatures

This repository contains a pipeline for mutational signature analysis in TCGA-LUAD lung adenocarcinoma data.

The workflow includes:
- generation of SBS96 mutation matrices from MAF files
- fitting to known COSMIC signatures
- merging with clinical and exposure data
- basic visualization of the results

## Project structure

```text
tcga-luad-smoking-signatures/
├── data/
│   ├── maf/
│   │   └── input/                  # input MAF files
│   ├── clinical.tsv
│   ├── exposure.tsv
│   └── clinical_exposure_merged.tsv
├── plots/                          # generated plots
├── results/                        # signature fitting outputs
├── scripts/                        # Python scripts
├── README.md
└── requirements.txt
```

## How to run

To run the whole pipeline, simply execute:

```bash
  bash setup_and_run.sh

# tcga-luad-smoking-signatures
