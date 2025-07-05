#!/bin/bash -l
#
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --partition=tg08x
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --output=temp_job.out

unset SLURM_EXPORT_ENV

module load python
pip install -e .
conda activate pytorch

# export proxy
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

echo "Unzipping files to $TMPDIR"
unzip -q "$HOME/i7-IDS/data/session_images.zip" -d "$TMPDIR"
echo "Unzipped session_images.zip"

export DATA_DIR="$TMPDIR/session_images"
export PROJECT_DIR = "$HOME/i7-IDS"

echo "Running trainer.py"

python3 $HOME/trainers/pcap_image_trainer_normal.py