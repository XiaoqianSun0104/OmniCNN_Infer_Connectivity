#!/bin/bash

# Run from repository root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# $PYTHON
if [[ -z "${PYTHON:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
  else
    echo "Error: no python executable found. Activate the conda environment first: conda activate omnicnn-connectivity"
    exit 1
  fi
fi


echo "Running many-to-one-post pipeline"
echo "Using Python: $($PYTHON --version)"

mkdir -p data results

echo "Step1 - Base / Perturbation Simulation"
for i in {0..4}; do
    python scripts/01_simulate_many_to_one.py \
      --dataFolderName data/m2post_smoke \
      --segFolderName "base_$i" \
      --simulationT 600000 \
      --num_pre 40 \
      --per_preE 0.5 \
      --per_preI 0.5 \
      --preBP_lb 1.0 \
      --wij_lb 0.8 \
      --generateUnconnectedCCG
done

python scripts/01_simulate_many_to_one.py \
  --dataFolderName data/m2post_smoke \
  --segFolderName BP8 \
  --simulationT 600000 \
  --num_pre 40 \
  --per_preE 0.5 \
  --per_preI 0.5 \
  --preBP_lb 0.8 \
  --preBP_ub 0.8 \
  --wij_lb 0.8 \
  --generateUnconnectedCCG


echo "Step2 - Generate DLs"
# generate TTV dataloaders for CNN training using base dataset
python scripts/02_make_m21_dataloaders.py \
  --dataFolderName data/m2post_smoke \
  --segFolderName base \
  --genTTVDL

# generate one dataloaders for trained CNN to test on
python scripts/02_make_m21_dataloaders.py \
  --dataFolderName data/m2post_smoke \
  --segFolderName BP \
  --genOneDL


echo "Step3 - Training"
python scripts/06_train_connectivity_cnn.py \
  --dataFolderName data/m2post_smoke \
  --dataloaderType Raw \
  --dataloaderName ttvLoader_base.pkl \
  --resultFolderName m2post_smoke \
  --unseen_dlName oneLoader_BP.pkl \
  --numRuns 5


echo "Step4 - Calculating Indicators & FMM"
python scripts/08_compute_ccg_indicators_fmm.py \
  --dataFolderName data/m2post_smoke \
  --base_segFolderName base \
  --perb_segFolderName BP \
  --dataloaderName ttvLoader_base.pkl \
  --unseen_dlName oneLoader_BP.pkl \
  --resultFolderName m2post_smoke \
  --indiFolderName indi


echo "Step5 - SHAP"
python scripts/09_SHAP.py \
  --base_segFolderName base \
  --perb_segFolderName BP \
  --resultFolderName m2post_smoke \
  --indiFolderName indi


echo "Smoke test finished."