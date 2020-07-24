dataset=$1
cpu=$2
time=$3
partition=$4

rs=1
feature_type_list=('full' 'limited')
test_type_list=('full' 'limited')
base_estimator_list=('lr' 'lgb')

sgl_method_list=('holdout' 'cv')
sgl_stacks_list=(1 2)
pgm_list=('psl' 'mrf')

if [ $dataset == 'youtube' ]; then
    relations=('user text')
elif [ $dataset == 'twitter' ]; then
    relations=('user text hashuser')
elif [ $dataset == 'soundcloud' ]; then
    relations=('user text link')
else
    echo 'Unknown dataset!'
fi

for feature_type in ${feature_type_list[@]}; do
    for test_type in ${test_type_list[@]}; do
        for base_estimator in ${base_estimator_list[@]}; do

        sbatch --cpus-per-task=$cpu \
               --time=$time \
               --partition=$partition \
               --job-name=EP_$dataset \
               --output=jobs/logs/peformance/$dataset \
               --error=jobs/errors/peformance/$dataset \
               jobs/performance/baseline.sh $dataset $rs \
               $feature_type $test_type $base_estimator

            # SGL only
            for sgl_method in ${sgl_method_list[@]}; do
                for sgl_stacks in ${sgl_stacks_list[@]}; do

                    sbatch --cpus-per-task=$cpu \
                           --time=$time \
                           --partition=$partition \
                           --job-name=EP_$dataset \
                           --output=jobs/logs/peformance/$dataset \
                           --error=jobs/errors/peformance/$dataset \
                           jobs/performance/eggs.sh $dataset $rs \
                           $feature_type $test_type $base_estimator \
                           $relations $sgl_method $sgl_stacks 'None'

            # PGM only
            for pgm in ${pgm_list[@]}; do

                sbatch --cpus-per-task=$cpu \
                       --time=$time \
                       --partition=$partition \
                       --job-name=EP_$dataset \
                       --output=jobs/logs/peformance/$dataset \
                       --error=jobs/errors/peformance/$dataset \
                       jobs/performance/eggs.sh $dataset $rs \
                       $feature_type $test_type $base_estimator \
                       $relations 'None' 0 $pgm

            # SGL + PGM
            for sgl_method in ${sgl_method_list[@]}; do
                for sgl_stacks in ${sgl_stacks_list[@]}; do
                    for pgm in ${pgm_list[@]}; do

                        sbatch --cpus-per-task=$cpu \
                               --time=$time \
                               --partition=$partition \
                               --job-name=EP_$dataset \
                               --output=jobs/logs/peformance/$dataset \
                               --error=jobs/errors/peformance/$dataset \
                               jobs/performance/eggs.sh $dataset $rs \
                               $feature_type $test_type $base_estimator \
                               $relations $sgl_method $sgl_stacks $pgm
        done
    done
done
