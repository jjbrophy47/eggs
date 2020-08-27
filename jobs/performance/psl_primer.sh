dataset=$1
scoring=$2
n_folds=$3
mem=$4
time=$5
partition=$6

rs=1
feature_type_list=('full' 'limited')
test_type_list=('full' 'inductive')
base_estimator_list=('lr')

sgl_method_list=('holdout' 'cv')
sgl_stacks_list=(1 2)
pgm_list=('psl')

for fold in $(seq 0 $n_folds); do

    for feature_type in ${feature_type_list[@]}; do
        for test_type in ${test_type_list[@]}; do
            for base_estimator in ${base_estimator_list[@]}; do

                for pgm in ${pgm_list[@]}; do

                    # PGM only
                    sbatch --mem=${mem}G \
                           --time=$time \
                           --partition=$partition \
                           --job-name=PSL_$dataset \
                           --output=jobs/logs/performance/$dataset \
                           --error=jobs/errors/performance/$dataset \
                           jobs/performance/eggs_runner.sh $dataset $rs \
                           $feature_type $test_type $base_estimator \
                           'None' 0 $pgm $fold $scoring
                done

                for sgl_method in ${sgl_method_list[@]}; do
                    for sgl_stacks in ${sgl_stacks_list[@]}; do
                        for pgm in ${pgm_list[@]}; do

                            # SGL + PGM
                            sbatch --mem=${mem}G \
                                   --time=$time \
                                   --partition=$partition \
                                   --job-name=PSL_$dataset \
                                   --output=jobs/logs/performance/$dataset \
                                   --error=jobs/errors/performance/$dataset \
                                   jobs/performance/eggs_runner.sh $dataset $rs \
                                   $feature_type $test_type $base_estimator \
                                   $sgl_method $sgl_stacks $pgm $fold $scoring
                        done
                    done
                done
            done
        done
    done
done
