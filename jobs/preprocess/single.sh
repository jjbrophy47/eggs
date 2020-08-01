dataset=$1
nrows=$2
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
       jobs/preprocess/features.sh $dataset $nrows \
       $val_frac $test_frac
