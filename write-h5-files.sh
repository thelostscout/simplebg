#!/usr/bin/env bash
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kortkamp@stud.uni-heidelberg.de

. ~/.bashrc
module load devel/miniconda
module load devel/cuda
conda activate lightning_bg
python src/loadNsave_system.py
echo "done"
