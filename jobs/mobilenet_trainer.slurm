#!/bin/bash -l
#
#SBATCH --gres=gpu:1
#SBATCH --partition=work
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --output=mobilenet.out

unset SLURM_EXPORT_ENV
module load python
conda activate pytorch
# export proxy
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
pip install -e .

echo "Unzipping files to $TMPDIR"
unzip -q "$HOME/i7-IDS/data/session_images.zip" -d "$TMPDIR"
echo "Unzipped session_images.zip"
cp $HOME/i7-IDS/data/labelled_sessions.csv $TMPDIR

export DATA_DIR="$TMPDIR"
export PROJECT_DIR="$HOME/i7-IDS"
export SAMPLING_METHOD="nosampling"

export CLF_MODE="multiclass"
python3 $HOME/i7-IDS/trainers/session_image_trainer_backbone.py --clf_mode $CLF_MODE --data_dir $DATA_DIR --project_dir $PROJECT_DIR --backbone mobilenet_v3_large --image_type normalized --batch_size 128 --attack_only
python3 $HOME/i7-IDS/trainers/session_image_trainer_backbone.py --clf_mode $CLF_MODE --data_dir $DATA_DIR --project_dir $PROJECT_DIR --backbone mobilenet_v3_large --image_type normal --batch_size 128 

# export CLF_MODE="binary"
# python3 $HOME/i7-IDS/trainers/session_image_trainer_backbone.py --clf_mode $CLF_MODE --data_dir $DATA_DIR --project_dir $PROJECT_DIR --backbone mobilenet_v3_large --image_type normalized --batch_size 128
# python3 $HOME/i7-IDS/trainers/session_image_trainer_backbone.py --clf_mode $CLF_MODE --data_dir $DATA_DIR --project_dir $PROJECT_DIR --backbone mobilenet_v3_large --image_type normal --batch_size 128

# export CLF_MODE="multiclass"
# python3 $HOME/i7-IDS/trainers/session_image_trainer_backbone.py --clf_mode $CLF_MODE --data_dir $DATA_DIR --project_dir $PROJECT_DIR --backbone mobilenet_v3_large --image_type normalized --batch_size 128 --attack_only
# python3 $HOME/i7-IDS/trainers/session_image_trainer_backbone.py --clf_mode $CLF_MODE --data_dir $DATA_DIR --project_dir $PROJECT_DIR --backbone mobilenet_v3_large --image_type normal --batch_size 128 --attack_only

