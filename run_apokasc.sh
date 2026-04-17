#!/bin/bash
#SBATCH --job-name=apokasc_bananas
#SBATCH --array=0-2999
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --output=logs/apokasc_%a.out

source ~/python_projects/venv/bin/activate
cd /project/galacticbulge/kiauhoku_NJM/KHstuff
python make_apokasc_bananas.py --star_index $SLURM_ARRAY_TASK_ID
