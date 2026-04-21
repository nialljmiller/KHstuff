#!/bin/bash
#SBATCH --job-name=banana_combine
#SBATCH --output=logs/banana_combine_%j.out
#SBATCH --error=logs/banana_combine_%j.err
#SBATCH --account=galacticbulge
#SBATCH --partition=mb
#SBATCH --qos=fast
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -euo pipefail
mkdir -p logs

PYENV="${PYENV:-$HOME/python_projects/venv}"
source "$PYENV/bin/activate"

KHSTUFF="/project/galacticbulge/kiauhoku_NJM/KHstuff"
cd "$KHSTUFF"

echo "Combining banana chains..."
echo "Started: $(date)"

python make_platinum_bananas.py --combine

echo "Finished: $(date)"
