#!/bin/bash -l
#
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --partition=work
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --output=ae_trainer.out

unset SLURM_EXPORT_ENV
module load python
conda activate pytorch

# export proxy
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

pip install -e .

echo "Copying compressed adversarial files to TMPDIR"

# Use the predefined directories
export DATA_DIR="$TMPDIR"
export PROJECT_DIR="$HOME/i7-IDS"


target_root="$DATA_DIR/adv_samples"

mkdir -p "$target_root"
adversarial_root="$PROJECT_DIR/results/adversarial_attacks/resnet18_nosampling"

# Copy and unzip all train and val zip files for each subfolder
for subfolder in "$adversarial_root"/*; do
    if [ -d "$subfolder" ]; then
        name=$(basename "$subfolder")
        for split in train val; do
            zip_file="${subfolder}/${name}_${split}.zip"
            dest_dir="${target_root}/${name}/${split}"
            mkdir -p "$dest_dir"
            if [ -f "$zip_file" ]; then
                echo "Unzipping $zip_file to $dest_dir"
                unzip -q "$zip_file" -d "$dest_dir"
            else
                echo "Warning: $zip_file not found."
            fi
        done
    fi
done

adversarial_root="$PROJECT_DIR/results/adversarial_attacks/mobilenet_v3_large_nosampling"

# Copy and unzip all train and val zip files for each subfolder
for subfolder in "$adversarial_root"/*; do
    if [ -d "$subfolder" ]; then
        name=$(basename "$subfolder")
        for split in train val; do
            zip_file="${subfolder}/${name}_${split}.zip"
            dest_dir="${target_root}/${name}/${split}"
            mkdir -p "$dest_dir"
            if [ -f "$zip_file" ]; then
                echo "Unzipping $zip_file to $dest_dir"
                unzip -q "$zip_file" -d "$dest_dir"
            else
                echo "Warning: $zip_file not found."
            fi
        done
    fi
done

echo "All adversarial files prepared in $target_root"

export BATCH_SIZE=64
python3 $HOME/i7-IDS/trainers/session_ae_trainer.py --data_dir "$DATA_DIR" --project_dir "$PROJECT_DIR" --ae_type unet --batch_size "$BATCH_SIZE" --data_type normal

# Clean up: delete the target_root directory
echo "Cleaning up $target_root"
rm -rf "$target_root"
echo "Cleanup complete."