#!/bin/bash -l
#
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --partition=work
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --output=adversarial_generator.out

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

python3 $HOME/i7-IDS/adversarial/generate_adversarial_image.py --data_dir $DATA_DIR --project_dir $PROJECT_DIR --model_names mobilenet_v3_large_nosampling --image_type normal --batch_size 64 &
python3 $HOME/i7-IDS/adversarial/generate_adversarial_image.py --data_dir $DATA_DIR --project_dir $PROJECT_DIR --model_names mobilenet_v3_large_normalized_nosampling --image_type normalized --batch_size 64
