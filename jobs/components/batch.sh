sbatch --cpus-per-task=3 \
       --time=1440 \
       --partition=short \
       --job-name=AC_youtube \
       --output=jobs/logs/components/youtube \
       --error=jobs/errors/components/youtube \
       jobs/components/single.sh youtube

sbatch --cpus-per-task=3 \
       --time=1440 \
       --partition=short \
       --job-name=AC_twitter \
       --output=jobs/logs/components/twitter \
       --error=jobs/errors/components/twitter \
       jobs/components/single.sh twitter

sbatch --cpus-per-task=15 \
       --time=1440 \
       --partition=short \
       --job-name=AC_soundcloud \
       --output=jobs/logs/components/soundcloud \
       --error=jobs/errors/components/soundcloud \
       jobs/components/single.sh soundcloud
