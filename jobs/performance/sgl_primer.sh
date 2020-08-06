dataset=$1
cpu=$2
time=$3
partition=$4

rs=1
fold_list=(0)
feature_type_list=('full' 'limited')
test_type_list=('full' 'inductive')
base_estimator_list=('lr')

sgl_method_list=('holdout', 'cv')
sgl_stacks_list=(1 2)

for fold in ${fold_list[@]}; do

    for feature_type in ${feature_type_list[@]}; do
        for test_type in ${test_type_list[@]}; do
            for base_estimator in ${base_estimator_list[@]}; do

                # baseline
                sbatch --cpus-per-task=$cpu \
                       --time=$time \
                       --partition=$partition \
                       --job-name=EP_$dataset \
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
                               --job-name=EP_$dataset \
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
