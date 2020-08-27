sbatch --mem=10G \
       --time=1440 \
       --partition=short \
       --job-name=AC_youtube \
       --output=jobs/logs/components/youtube \
       --error=jobs/errors/components/youtube \
       jobs/components/runner.sh youtube 10

sbatch --mem=10G \
       --time=1440 \
       --partition=short \
       --job-name=AC_twitter \
       --output=jobs/logs/components/twitter \
       --error=jobs/errors/components/twitter \
       jobs/components/runner.sh twitter 10

sbatch --mem=15G \
       --time=1440 \
       --partition=short \
       --job-name=AC_soundcloud \
       --output=jobs/logs/components/soundcloud \
       --error=jobs/errors/components/soundcloud \
       jobs/components/runner.sh soundcloud 9
