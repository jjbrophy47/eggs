#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load python3/3.7.5

dataset=$1
val_frac=$2
test_frac=$3

python3 scripts/preprocess/features.py \
  --dataset $dataset \
  --feature_type 'limited' \
  --relations \
  --val_frac $val_frac \
  --test_frac $test_frac

python3 scripts/preprocess/features.py \
  --dataset $dataset \
  --feature_type 'full' \
  --val_frac $val_frac \
  --test_frac $test_frac
