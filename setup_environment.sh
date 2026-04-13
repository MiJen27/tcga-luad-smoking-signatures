#!/bin/bash

set -e

ENV_NAME="tcga-luad-smoking-signatures"

echo "Creating conda environment: $ENV_NAME"
conda env remove -n "$ENV_NAME" -y || true
conda create -y -n "$ENV_NAME" python=3.10

echo "Installing project requirements"
conda run -n "$ENV_NAME" python -m pip install -r requirements.txt

echo "Installing the GRCh38 reference for SigProfilerMatrixGenerator"
conda run -n "$ENV_NAME" python -c "from SigProfilerMatrixGenerator import install as genInstall; genInstall.install('GRCh38', rsync=False, bash=True)"

echo "Environment is ready."
echo "Now open Jupyter and choose the kernel from: $ENV_NAME"
