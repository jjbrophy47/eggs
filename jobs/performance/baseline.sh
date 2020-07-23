#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load python3/3.7.5
module load libra/1.1.2
modeul load java/1.8.0

dataset=$1
rs=$2
feature_type=$3
test_type=$4
base_estimator=$5

python3 experiments/scripts/performance.py \
  --dataset $dataset \
  --rs $rs \
  --feature_type $feature_type \
  --test_type $test_type \
  --base_estimator $base_estimator \
