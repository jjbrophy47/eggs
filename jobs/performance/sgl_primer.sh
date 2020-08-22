dataset=$1
n_folds=$2
cpu=$3
time=$4
partition=$5

rs=1
feature_type_list=('full' 'limited')
test_type_list=('full' 'inductive')
base_estimator_list=('lr')

sgl_method_list=('holdout' 'cv')
sgl_stacks_list=(1 2)

for fold in $(seq 0 $n_folds); do

    for feature_type in ${feature_type_list[@]}; do
        for test_type in ${test_type_list[@]}; do
            for base_estimator in ${base_estimator_list[@]}; do

                # baseline
                sbatch --cpus-per-task=$cpu \
                       --time=$time \
                       --partition=$partition \
                       --job-name=BASE_$dataset \
                       --output=jobs/logs/performance/$dataset \
                       --error=jobs/errors/performance/$dataset \
                       jobs/performance/baseline_runner.sh $dataset $rs \
                       $feature_type $test_type $base_estimator $fold

                for sgl_method in ${sgl_method_list[@]}; do
                    for sgl_stacks in ${sgl_stacks_list[@]}; do

                        # SGL only
                        sbatch --cpus-per-task=$cpu \
                               --time=$time \
                               --partition=$partition \
                               --job-name=SGL_$dataset \
                               --output=jobs/logs/performance/$dataset \
                               --error=jobs/errors/performance/$dataset \
                               jobs/performance/eggs_runner.sh $dataset $rs \
                               $feature_type $test_type $base_estimator \
                               $sgl_method $sgl_stacks 'None' $fold 'auc'
                    done
                done
            done
        done
    done
done
