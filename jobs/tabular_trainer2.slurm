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
# echo "Training tabular models in project directory: $PROJECT_DIR"
# python3 $HOME/i7-IDS/trainers/fnn_trainer.py --project_dir $PROJECT_DIR --model fnn &
# python3 $HOME/i7-IDS/trainers/fnn_trainer.py --project_dir $PROJECT_DIR --model cnn

# echo "Generating adversarial samples for original models"
python3 $HOME/i7-IDS/adversarial/generate_tabular.py --project_dir $PROJECT_DIR --models original_cnn &
python3 $HOME/i7-IDS/adversarial/generate_tabular.py --project_dir $PROJECT_DIR --models original_fnn

# echo "Training adversarial models"
python $HOME/i7-IDS/trainers/fnn_adv_trainer.py &
python $HOME/i7-IDS/trainers/tabular_ae_trainer.py 

echo "Evaluating adversarial samples"
python /home/hpc/iwi7/iwi7101h/i7-IDS/adversarial/evaluate_from_generated_tabular.py
