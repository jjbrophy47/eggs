# baseline and SGL
# ./jobs/performance/sgl_primer.sh youtube 9 5 1440 short
# ./jobs/performance/sgl_primer.sh twitter 9 5 1440 short
./jobs/performance/sgl_primer.sh soundcloud 8 8 1440 short

# MRF and SGL + MRF
# ./jobs/performance/mrf_primer.sh youtube auc 9 11 4320 long
# ./jobs/performance/mrf_primer.sh twitter auc 9 15 4320 long
./jobs/performance/mrf_primer.sh soundcloud auc 8 45 4320 long

# PSL and SGL + PSL
./jobs/performance/psl_primer.sh youtube auc 9 11 4320 long
./jobs/performance/psl_primer.sh twitter auc 9 15 4320 long
./jobs/performance/psl_primer.sh soundcloud auc 8 45 4320 long
