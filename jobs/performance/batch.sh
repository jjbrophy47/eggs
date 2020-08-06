# baseline and SGL
./jobs/performance/sgl_primer.sh youtube 2 1440 short
./jobs/performance/sgl_primer.sh twitter 2 1440 short
./jobs/performance/sgl_primer.sh soundcloud 2 1440 short

# MRF and MRF + SGL
./jobs/performance/mrf_primer.sh youtube auc 3 4320 long
./jobs/performance/mrf_primer.sh twitter auc 3 4320 long
./jobs/performance/mrf_primer.sh soundcloud auc 3 4320 long
