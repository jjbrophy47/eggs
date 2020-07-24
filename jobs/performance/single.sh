dataset=$1
cpu=$2
time=$3
partition=$4

rs=1
feature_type_list=('full' 'limited')
test_type_list=('full' 'inductive')
base_estimator_list=('lr' 'lgb')

sgl_method_list=('holdout' 'cv')
sgl_stacks_list=(1 2)
pgm_list=('mrf')

# PSL cannot run parallel jobs due to database connection
sbatch --cpus-per-task=$cpu \
       --time=$time \
       --partition=$partition \
       --job-name=EP_$dataset \
       --output=jobs/logs/performance/$dataset \
       --error=jobs/errors/performance/$dataset \
       jobs/performance/eggs_psl.sh $dataset $rs

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
                   jobs/performance/baseline.sh $dataset $rs \
                   $feature_type $test_type $base_estimator

            for sgl_method in ${sgl_method_list[@]}; do
                for sgl_stacks in ${sgl_stacks_list[@]}; do

                    # SGL only
                    sbatch --cpus-per-task=$cpu \
                           --time=$time \
                           --partition=$partition \
                           --job-name=EP_$dataset \
                           --output=jobs/logs/performance/$dataset \
                           --error=jobs/errors/performance/$dataset \
                           jobs/performance/eggs.sh $dataset $rs \
                           $feature_type $test_type $base_estimator \
                           $sgl_method $sgl_stacks 'None'
                done
            done

            for pgm in ${pgm_list[@]}; do

                # PGM only
                sbatch --cpus-per-task=$cpu \
                       --time=$time \
                       --partition=$partition \
                       --job-name=EP_$dataset \
                       --output=jobs/logs/performance/$dataset \
                       --error=jobs/errors/performance/$dataset \
                       jobs/performance/eggs.sh $dataset $rs \
                       $feature_type $test_type $base_estimator \
                       'None' 0 $pgm
            done

            for sgl_method in ${sgl_method_list[@]}; do
                for sgl_stacks in ${sgl_stacks_list[@]}; do
                    for pgm in ${pgm_list[@]}; do

                        # SGL + PGM
                        sbatch --cpus-per-task=$cpu \
                               --time=$time \
                               --partition=$partition \
                               --job-name=EP_$dataset \
                               --output=jobs/logs/performance/$dataset \
                               --error=jobs/errors/performance/$dataset \
                               jobs/performance/eggs.sh $dataset $rs \
                               $feature_type $test_type $base_estimator \
                               $sgl_method $sgl_stacks $pgm
                    done
                done
            done
        done
    done
done
