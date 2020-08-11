#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load python3/3.7.5

dataset=$1
n_folds=$2

test_type_list=('full' 'inductive')
eval_set_list=('val' 'test')

if [ $dataset == 'youtube' ]; then
    relations=('user' 'text')
elif [ $dataset == 'twitter' ]; then
    relations=('user' 'text' 'hashuser')
elif [ $dataset == 'soundcloud' ]; then
    relations=('user' 'text' 'link')
else
    echo 'Unknown dataset!'
fi

for test_type in ${test_type_list[@]}; do
    for eval_set in ${eval_set_list[@]}; do
        python3 scripts/analysis/components.py \
          --dataset $dataset \
          --test_type $test_type \
          --relations "${relations[@]}" \
          --n_folds $n_folds \
          --eval_set $eval_set
    done
done
