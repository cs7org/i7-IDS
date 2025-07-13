#!/bin/bash -l
#
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --partition=work
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --output=adversarial_generator_tabular.out

unset SLURM_EXPORT_ENV
module load python
conda activate pytorch
# export proxy
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
pip install -e .

export PROJECT_DIR="$HOME/i7-IDS"

python3 $HOME/i7-IDS/adversarial/generate_tabular.py --project_dir $PROJECT_DIR --models fnn_original &
python3 $HOME/i7-IDS/adversarial/generate_tabular.py --project_dir $PROJECT_DIR --models cnn_original &
python3 $HOME/i7-IDS/adversarial/generate_tabular.py --project_dir $PROJECT_DIR --models fnn_synthetic &
python3 $HOME/i7-IDS/adversarial/generate_tabular.py --project_dir $PROJECT_DIR --models cnn_synthetic