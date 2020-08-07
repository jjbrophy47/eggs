dataset=$1
scoring=$2
cpu=$3
time=$4
partition=$5

rs=1
fold_list=(0)
feature_type_list=('full' 'limited')
test_type_list=('full' 'inductive')
base_estimator_list=('lr')

sgl_method_list=('holdout', 'cv')
sgl_stacks_list=(1 2)
pgm_list=('mrf')

for fold in ${fold_list[@]}; do

    for feature_type in ${feature_type_list[@]}; do
        for test_type in ${test_type_list[@]}; do
            for base_estimator in ${base_estimator_list[@]}; do

                for pgm in ${pgm_list[@]}; do

                    # PGM only
                    sbatch --cpus-per-task=$cpu \
                           --time=$time \
                           --partition=$partition \
                           --job-name=MRF_$dataset \
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
                            sbatch --cpus-per-task=$cpu \
                                   --time=$time \
                                   --partition=$partition \
                                   --job-name=MRF_$dataset \
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
