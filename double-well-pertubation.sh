#!/usr/bin/env bash
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kortkamp@stud.uni-heidelberg.de

. /home/hd/hd_hd/hd_qo191/.bashrc
echo "$@"
module load devel/miniconda
module load devel/cuda
conda activate lightning_bg
python src/double-well-pertubation.py "$@"