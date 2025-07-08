from pathlib import Path
from loguru import logger

import torch
from ids_expt.adversarial.adversarial_experiment import AdversarialExperiment, ClfModel
from ids_expt.core.defs import TOP_CIC_FEATURES
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod
from art.estimators.classification import PyTorchClassifier
from ids_expt.data.dataset import (
    DataSetConfig,
    SamplingMethod,
    CLFDataSet as DataSet,
    DFDataSet,
)

model_paths = [
    Path(
        r"C:\Users\Viper\Desktop\thesis_code\results\cic_fnn\ctgan_oversampling\best_model_full.pth"
    ),
    Path(
        r"C:\Users\Viper\Desktop\thesis_code\results\cic_cnn\ctgan_oversampling\best_model_full.pth"
    ),
]
for model_path in model_paths:
    if not model_path.exists():
        logger.error(f"Model path {model_path} does not exist.")
        exit(1)
    # model_path = Path(
    #     r"C:\Users\Viper\Desktop\thesis_code\results\cic_cnn\ctgan_oversampling\best_model_full.pth"
    # )
    project_dir = Path(r"C:\Users\Viper\Desktop\thesis_code")

    train_dataset, val_dataset = DFDataSet(
        config=DataSetConfig(
            csv_path=project_dir / "data/cic_ctgan_merged_synthetic_data.csv",
            features=TOP_CIC_FEATURES,
            sampling_method=SamplingMethod.NONE,
            max_data=-100,
            train_ratio=0.8,
        )
    ).get_datasets()

    val_dataset.data = val_dataset.data.query("is_synthetic != True")

    val_dataset = DataSet(val_dataset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
    )
    model = torch.load(
        model_path,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        weights_only=False,
    )

    logger.info(f"Running adversarial attacks on model: {model_path.name}")
    epsilons = [0.0001, 0.001, 0.01, 0.1]
    iterations = 10
    input_shape = (1, len(TOP_CIC_FEATURES))

    attacks = [
        FastGradientMethod(
            estimator=PyTorchClassifier(
                model=ClfModel(model),
                loss=torch.nn.CrossEntropyLoss(),
                clip_values=(0, 1),
                input_shape=input_shape,
                nb_classes=len(train_dataset.label_encoding),
                optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
            ),
            eps=eps,
        )
        for eps in epsilons
    ]
    attacks.extend(
        [
            BasicIterativeMethod(
                estimator=PyTorchClassifier(
                    model=ClfModel(model),
                    loss=torch.nn.CrossEntropyLoss(),
                    clip_values=(0, 1),
                    input_shape=input_shape,
                    nb_classes=len(train_dataset.label_encoding),
                    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
                ),
                eps=eps,
                max_iter=iterations,
            )
            for eps in epsilons
        ]
    )
    adv = AdversarialExperiment(
        model=model,
        model_name=model_path.parent.parent.name,
        attacks=attacks,
        train_dataset=DataSet(train_dataset),
        test_dataset=DataSet(val_dataset),
    )

    adv.run()

    logger.info("Adversarial attacks completed successfully.")
    logger.info("Generating adversarial attack data...")

    selected_attacks = [atk for atk in attacks if atk.eps in [0.1, 0.01]]

    for attack in selected_attacks:
        logger.info(
            f"Generating adversarial data for attack: {attack.__class__.__name__} with eps: {attack.eps}"
        )
        out_folder = attack.__class__.__name__.lower() + f"_eps_{attack.eps}"
        adv.generate(attack, out_folder=out_folder, is_image_dataset=False)
