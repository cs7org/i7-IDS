# Enhancing Smart Grid Security: A Deep Learning Approach to Adversarial Intrusion Detection
## Experiments
### Setting Up
1. Download the dataset DNP3 Intrusion Detection Dataset from [Zenodo](https://zenodo.org/records/7348493)
2. Unzip and copy all the CSV files related to CICFlowmeter and paste in a single folder.
3. These files will be the main data files.
4. Read all files and combine them all together in a single CSV file. This file will be used to train models.
5. Install this project as `pip install -e .` and all its requirements too.

### Feature Importance
Based on [Permutation Feature Importance](https://christophm.github.io/interpretable-ml-book/feature-importance.html). [This](https://pmc.ncbi.nlm.nih.gov/articles/PMC8323609/pdf/nihms-1670270.pdf) is also a good read.

**Feature importance could vary for each attack.**

### Imbalance Handling
* Synthetic Data generation to oversample data based on [CTGAN](https://arxiv.org/html/2410.16326v1). 
* Here, we loop through each label and generate synthetic dataset needed. 
* So for each attack labels there is one CTGAN model.
* CTGAN can be installed from [here](https://github.com/sdv-dev/CTGAN).


### PCAP to Image
* This is experimental at the moment. But the goal is to reverse-engineer the labels from CSV to raw PCAPs file for packet based session generation.
* If the approach looks good, the image based IDS might worth an experiment to try.
* Read the CSV file for 120s Timeout and the corresponding PCAP files.
* For each CSV:
    * Read each row.
    * Find the matching packets in the PCAP file.
    * Call matched packets session and assign the label to it.
    * Convert session to image.
* Functional script is: `feature_importance\dnp3_pcap_to_img.py`


### Baseline Model Training
* First baseline model is from the [Data Authors](https://ieeexplore.ieee.org/document/9881726).

## Debug on HPC
* `salloc.tinygpu --gres=gpu:1 --time=01:00:00`
* `srun --jobid=1121191 --overlap --pty /bin/bash -l`
* `tqdm` should be disabled in HPC.
* Quota: `shownicerquota.pl`

## Model Training
* PyTorch 2.5.0 with GPU.
* As MLFlow is being used for logging the parameters, command `mlflow server` should be run before training a model.
* Dataset for tabular data: `ids_expt\data\dataset.py`.
* Dataset for session image data: `ids_expt\data\session_image_dataset.py`.
* Trainer: `\ids_expt\models\trainer.py`. A single trainer to train all models.
* `pip install "tabpfn-extensions[all] @ git+https://github.com/PriorLabs/tabpfn-extensions.git"`

