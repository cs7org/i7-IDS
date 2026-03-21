#!/bin/bash -l
#
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --partition=work
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --output=adv_eval_normalized.out

unset SLURM_EXPORT_ENV
module load python
conda activate pytorch
# export proxy
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
pip install -e .

## Normal ##

# Set up directories
INPUT_ROOT="$WORK/i7-IDS/results/adversarial_attacks/resnet18_normalized_nosampling"
OUTPUT_ROOT="$TMPDIR"
DATA_SPLIT="val"

# Extract last folder name from INPUT_ROOT for ADV_GEN_MODEL
ADV_GEN_MODEL=$(basename "$INPUT_ROOT")

# Create adv_samples directory as OUTPUT_ROOT/ADV_GEN_MODEL
ADV_SAMPLES_DIR="$OUTPUT_ROOT/$ADV_GEN_MODEL"
mkdir -p "$ADV_SAMPLES_DIR"

echo "INPUT_ROOT:   $INPUT_ROOT"
echo "OUTPUT_ROOT:  $OUTPUT_ROOT"
echo "ADV_GEN_MODEL: $ADV_GEN_MODEL"
echo "ADV_SAMPLES_DIR: $ADV_SAMPLES_DIR"

echo "Processing adversarial samples for model: $ADV_GEN_MODEL"
echo "Data split to process: $DATA_SPLIT"
echo "Input directory: $INPUT_ROOT"
echo "Output directory: $ADV_SAMPLES_DIR"
echo ""

# Determine which splits to process
if [ "$DATA_SPLIT" == "both" ]; then
    splits_to_process=("train" "val")
elif [ "$DATA_SPLIT" == "train" ]; then
    splits_to_process=("train")
elif [ "$DATA_SPLIT" == "val" ]; then
    splits_to_process=("val")
else
    echo "Error: DATA_SPLIT must be 'train', 'val', or 'both'"

fi

# Process each attack method
for method_dir in "$INPUT_ROOT"/*; do
    if [ -d "$method_dir" ]; then
        method_name=$(basename "$method_dir")
        echo "Processing method: $method_name"
        
        # Unzip specified split files
        for split in "${splits_to_process[@]}"; do
            # Create output directory with train/val subdirectory
            output_method_dir="$ADV_SAMPLES_DIR/$ADV_GEN_MODEL/$method_name/$split"
            mkdir -p "$output_method_dir"
            
            zip_file="$method_dir/${method_name}_${split}.zip"
            if [ -f "$zip_file" ]; then
                echo "  - Unzipping ${method_name}_${split}.zip to $output_method_dir"
                unzip -q "$zip_file" -d "$output_method_dir"
                echo "    ✓ Extracted ${split} data"
            else
                echo "  - Warning: $zip_file not found"
            fi
        done
        echo ""
    fi
done

echo "All adversarial samples processed!"
echo "Output location: $ADV_SAMPLES_DIR/$ADV_GEN_MODEL"

python /home/hpc/iwi7/iwi7101h/i7-IDS/adversarial/evaluate_from_generated.py --adv_gen_model $ADV_GEN_MODEL --data_dir $ADV_SAMPLES_DIR/$ADV_GEN_MODEL --batch_size 16 --data_type normalized

