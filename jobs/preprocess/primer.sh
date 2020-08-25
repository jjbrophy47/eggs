dataset=$1
val_frac=$2
test_frac=$3
mem=$4
time=$5
partition=$6

sbatch --mem=${mem}G \
       --time=$time \
       --partition=$partition \
       --job-name=ED_$dataset \
       --output=jobs/logs/preprocess/$dataset \
       --error=jobs/errors/preprocess/$dataset \
       jobs/preprocess/runner.sh $dataset $val_frac $test_frac
