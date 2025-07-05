#!/bin/bash -l
#
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --partition=rtx3080
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --output=temp_job.out

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

python3 $HOME/i7-IDS/trainers/pcap_image_trainer_normal.py &
python3 $HOME/i7-IDS/trainers/pcap_image_trainer_normalized.py