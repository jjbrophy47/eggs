sbatch --cpus-per-task=3 \
       --time=1440 \
       --partition=short \
       --job-name=AC_youtube \
       --output=jobs/logs/components/$dataset \
       --error=jobs/errors/components/$dataset \
       jobs/components/single.sh youtube

sbatch --cpus-per-task=3 \
       --time=1440 \
       --partition=short \
       --job-name=AC_twitter \
       --output=jobs/logs/components/$dataset \
       --error=jobs/errors/components/$dataset \
       jobs/components/single.sh twitter

sbatch --cpus-per-task=15 \
       --time=1440 \
       --partition=short \
       --job-name=AC_soundcloud \
       --output=jobs/logs/components/$dataset \
       --error=jobs/errors/components/$dataset \
       jobs/components/single.sh soundcloud
