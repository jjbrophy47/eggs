#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load python3/3.7.5

dataset=$1
n_fold_samples=$2
# nrows=$2
# val_frac=$3
# test_frac=$4

python3 scripts/preprocess/features.py \
  --dataset $dataset \
  --feature_type 'limited' \
  --relations \
  --n_fold_samples $n_fold_samples
  # --nrows $nrows \
  # --val_frac $val_frac \
  # --test_frac $test_frac

python3 scripts/preprocess/features.py \
  --dataset $dataset \
  --feature_type 'full' \
  --n_fold_samples $n_fold_samples
  # --nrows $nrows \
  # --val_frac $val_frac \
  # --test_frac $test_frac
