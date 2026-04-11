#!/bin/bash

set -e

ENV_NAME="tcga-luad-smoking-signatures"

echo "=== 0. Removing old environment if it already exists ==="
conda env remove -n $ENV_NAME -y || true

echo "=== 1. Creating conda environment with Python 3.10 ==="
conda create -y -n $ENV_NAME python=3.10

echo "=== 2. Installing requirements ==="
conda run -n $ENV_NAME python -m pip install -r requirements.txt
conda run -n $ENV_NAME python -c "from SigProfilerMatrixGenerator import install as genInstall; genInstall.install('GRCh38', rsync=False, bash=True)"

echo "=== 3. Running pipeline ==="
conda run -n $ENV_NAME python main.py

echo "DONE!"