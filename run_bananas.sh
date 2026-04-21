#!/bin/bash
#SBATCH --job-name=banana_MCMC
#SBATCH --output=logs/banana_%A_%a.out
#SBATCH --error=logs/banana_%A_%a.err
#SBATCH --account=galacticbulge
#SBATCH --partition=mb
#SBATCH --qos=fast                     # 12h max
#SBATCH --time=11:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1              # one star per job, serial MCMC
#SBATCH --array=0-90                   # one job per star (43 stars)
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

set -euo pipefail
mkdir -p logs

PYENV="${PYENV:-$HOME/python_projects/venv}"
source "$PYENV/bin/activate"

KHSTUFF="/project/galacticbulge/kiauhoku_NJM/KHstuff"
cd "$KHSTUFF"

echo "Job array index: $SLURM_ARRAY_TASK_ID"
echo "Running on host: $(hostname)"
echo "Started: $(date)"

python make_platinum_bananas.py --star_index "$SLURM_ARRAY_TASK_ID"

echo "Finished: $(date)"
