#!/bin/bash -l
#
#SBATCH --gres=gpu:1
#SBATCH --partition=work
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --output=adv_eval_tabular.out

unset SLURM_EXPORT_ENV
module load python
conda activate pytorch
# export proxy
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
pip install -e .

# python /home/hpc/iwi7/iwi7101h/i7-IDS/adversarial/evaluate_from_generated_tabular.py --adv_gen_model $ADV_GEN_MODEL --data_dir $ADV_SAMPLES_DIR/$ADV_GEN_MODEL --batch_size 16 --data_type normal
python /home/hpc/iwi7/iwi7101h/i7-IDS/adversarial/evaluate_from_generated_tabular.py
