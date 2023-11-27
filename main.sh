#!/usr/bin/env bash
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=100:00:00
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kortkamp@stud.uni-heidelberg.de

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

. ~/.bashrc
echo "$@"
module load devel/miniconda
module load devel/cuda
conda activate lightning_bg
python src/main.py "$@" --disable_progress_bar