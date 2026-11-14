#!/bin/bash
#SBATCH --job-name=aya32-train-report
#SBATCH --partition=mini
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=aya32-train-report-%j.out
set -euo pipefail
export LC_ALL=C LANG=C
umask 077
: "${AYA32_TRAIN_JOB:?}" "${AYA32_RUN_ROOT:?}"
[[ "$AYA32_TRAIN_JOB" =~ ^[0-9]+$ ]]
ROOT="/scratch/${USER:?}/aya_expanse_32b_pg"
cd "$ROOT/full_trainer"
sha256sum -c MANIFEST.sha256
export TRAIN_STATE
TRAIN_STATE=$(sacct -n -P -j "$AYA32_TRAIN_JOB" --format=State,ExitCode | head -n 1)
LD_LIBRARY_PATH="/home/apps/Python/Python-3.10.16/lib:/home/apps/openssl-1.1.1w/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
 /home/apps/Python/Python-3.10.16/bin/python3 monitor.py
