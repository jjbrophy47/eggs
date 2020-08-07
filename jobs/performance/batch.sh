# baseline and SGL
# ./jobs/performance/sgl_primer.sh youtube 2 1440 short
# ./jobs/performance/sgl_primer.sh twitter 2 1440 short
# ./jobs/performance/sgl_primer.sh soundcloud 2 1440 short

# MRF and SGL + MRF
# ./jobs/performance/mrf_primer.sh youtube auc 3 4320 long
# ./jobs/performance/mrf_primer.sh twitter auc 3 4320 long
# ./jobs/performance/mrf_primer.sh soundcloud auc 3 4320 long

# PSL and SGL + PSL - must be run sequentially
dataset='youtube'
sbatch --cpus-per-task=3 \
       --time=1440 \
       --partition=short \
       --job-name=PSL_$dataset \
       --output=jobs/logs/performance/$dataset \
       --error=jobs/errors/performance/$dataset \
       jobs/performance/baseline_runner.sh $dataset 1 0
