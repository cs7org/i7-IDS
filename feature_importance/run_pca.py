from ids_expt.data.pca import PCAProjector
from ids_expt.data.utils import oversample_class, get_dnp_cic_data

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from loguru import logger
from sklearn.metrics import f1_score, accuracy_score
import time


samples = -1000
dnp3_df, cicflow_df = get_dnp_cic_data(samples=samples)

results = []
max_steps = -2
epochs = 50
data_types = ["dnp3", "cicflow"]
for i, df in enumerate([dnp3_df, cicflow_df]):

    dtype = data_types[i]
    # df = oversample_class(df, "Label")
    features_df = df.drop(columns=["Label"])
    labels_df = df["Label"]
    train_features_df, test_features, train_labels_df, test_labels = train_test_split(
        features_df,
        labels_df,
        test_size=0.25,
        random_state=42,
        stratify=labels_df,
    )
    num_features = len(features_df.columns)

    pca_projector = PCAProjector(n_components=num_features, scale=True)
    selected_features = features_df.columns.tolist()
    experiment_results = {
        "acc": [],
        "f1": [],
        "time": [],
    }
    for step in range(1, len(selected_features) + 1):
        if step > max_steps and max_steps > 0:
            break
        logger.info(f"\nRunning Step {step}")
        curr_num_features = num_features - step
        if curr_num_features <= 1:
            break

        # train

        pca_projector.n_components = curr_num_features

        t0 = time.perf_counter()
        pca_projector.fit_transform(features_df)

        # projection
        pca_train = pca_projector.project(features_df)
        pca_test = pca_projector.project(test_features)

        # Check the shape of the projected data
        assert pca_train.shape[1] == curr_num_features
        assert pca_test.shape[1] == curr_num_features

        # train model
        model = RandomForestClassifier()
        model.fit(pca_train, labels_df)
        acc = accuracy_score(test_labels, model.predict(pca_test))
        f1 = f1_score(test_labels, model.predict(pca_test), average="weighted")
        experiment_results["acc"].append(acc)
        experiment_results["f1"].append(f1)
        experiment_results["time"].append(time.perf_counter() - t0)

        logger.info(
            f"Step {step}: Feature Shape: {pca_train.shape}, Acc Score = {acc:.4f}, F1 Score = {f1:.4f}"
        )

    res_df = pd.DataFrame(
        experiment_results,
        index=[i for i in range(len(experiment_results["acc"]))],
        columns=["acc", "f1", "time"],
    )
    results.append(res_df)

    res_df.to_csv(f"results_{dtype}_pca.csv", index=False)
