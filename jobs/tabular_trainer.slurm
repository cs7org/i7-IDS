#!/bin/bash -l
#
#SBATCH --gres=gpu:1
#SBATCH --partition=work
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --output=tabular_trainer.out

unset SLURM_EXPORT_ENV
module load python
conda activate pytorch
# export proxy
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
pip install -e .

export PROJECT_DIR="$HOME/i7-IDS"

# python3 $HOME/i7-IDS/trainers/tabular_trainer.py --project_dir $PROJECT_DIR --model fnn &
# python3 $HOME/i7-IDS/trainers/tabular_trainer.py --project_dir $PROJECT_DIR --model cnn &
python3 $HOME/i7-IDS/trainers/tabular_trainer.py --project_dir $PROJECT_DIR --model fnn --data_type original &
python3 $HOME/i7-IDS/trainers/tabular_trainer.py --project_dir $PROJECT_DIR --model cnn --data_type original