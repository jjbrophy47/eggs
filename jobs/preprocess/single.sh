dataset=$1
n_fold_samples=$2
val_frac=$3
test_frac=$4
cpu=$5
time=$6
partition=$7

sbatch --cpus-per-task=$cpu \
       --time=$time \
       --partition=$partition \
       --job-name=ED_$dataset \
       --output=jobs/logs/preprocess/$dataset \
       --error=jobs/errors/preprocess/$dataset \
       jobs/preprocess/features.sh $dataset $n_fold_samples \
       $val_frac $test_frac
