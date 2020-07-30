#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load python3/3.7.5

dataset=$1

test_type_list=('full' 'inductive')

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
    python3 scripts/analysis/components.py \
      --dataset $dataset \
      --test_type $test_type \
      --relations "${relations[@]}"
done
