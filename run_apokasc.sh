#!/bin/bash
#SBATCH --job-name=apokasc_bananas
#SBATCH --array=0-2056%20
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --output=logs/apokasc_%a.out
#SBATCH --error=logs/apokasc_%a.err
#SBATCH --account=galacticbulge
#SBATCH --partition=mb
#SBATCH --qos=fast
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

source ~/python_projects/venv/bin/activate
cd /project/galacticbulge/kiauhoku_NJM/KHstuff
mkdir -p logs results/apokasc/chains results/apokasc/plots

python make_apokasc_bananas.py \
    --star_index "$SLURM_ARRAY_TASK_ID" \
    --n_walkers 24 \
    --n_burnin 200 \
    --n_iter 600
