#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load python3/3.7.5
module load libra/1.1.2
module load java/1.8.0

dataset=$1
rs=$2

if [ $dataset == 'youtube' ]; then
    relations=('user' 'text')
elif [ $dataset == 'twitter' ]; then
    relations=('user' 'text' 'hashuser')
elif [ $dataset == 'soundcloud' ]; then
    relations=('user' 'text' 'link')
else
    echo 'Unknown dataset!'
fi

feature_type_list=('full' 'limited')
test_type_list=('full' 'limited')
base_estimator_list=('lr' 'lgb')

sgl_method_list=('holdout' 'cv')
sgl_stacks_list=(1 2)

for feature_type in ${feature_type_list[@]}; do
    for test_type in ${test_type_list[@]}; do
        for base_estimator in ${base_estimator_list[@]}; do

            python3 experiments/scripts/performance.py \
              --eggs \
              --dataset $dataset \
              --rs $rs \
              --feature_type $feature_type \
              --test_type $test_type \
              --base_estimator $base_estimator \
              --relations "${relations[@]}" \
              --pgm 'psl'

            for sgl_method in ${sgl_method_list[@]}; do
                for sgl_stacks in ${sgl_stacks_list[@]}; do

                    python3 experiments/scripts/performance.py \
                      --eggs \
                      --dataset $dataset \
                      --rs $rs \
                      --feature_type $feature_type \
                      --test_type $test_type \
                      --base_estimator $base_estimator \
                      --relations "${relations[@]}" \
                      --sgl_method $sgl_method \
                      --sgl_stacks $sgl_stacks \
                      --pgm 'psl'
