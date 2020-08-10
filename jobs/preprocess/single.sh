dataset=$1
n_fold_samples=$2
cpu=$3
time=$4
partition=$5

sbatch --cpus-per-task=$cpu \
       --time=$time \
       --partition=$partition \
       --job-name=ED_$dataset \
       --output=jobs/logs/preprocess/$dataset \
       --error=jobs/errors/preprocess/$dataset \
       jobs/preprocess/features.sh $dataset $n_fold_samples
