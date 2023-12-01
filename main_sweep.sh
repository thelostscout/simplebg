#!/usr/bin/env bash
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=10:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kortkamp@stud.uni-heidelberg.de

. ~/.bashrc
echo "$@"
module load devel/miniconda
module load devel/cuda
conda activate lightning_bg
python src/main_sweep.py "$@"
echo "done"
