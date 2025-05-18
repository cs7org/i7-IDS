from ids_expt.data.rfe import RFE
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

    rfe = RFE(
        RandomForestClassifier,
        features_df,
        labels_df,
        test_features,
        test_labels,
        num_steps=num_features,
        stopping_score=0.5,
    )
    selected_features = features_df.columns.tolist()
    experiment_results = {
        "acc": [],
        "f1": [],
        "time": [],
    }
    t0 = time.perf_counter()
    for step in range(1, len(selected_features) + 1):
        if step > max_steps and max_steps > 0:
            break
        logger.info(f"\nRunning Step {step}")
        curr_num_features = num_features - step
        if curr_num_features <= 1:
            break

        # train

        t0 = time.perf_counter()
        try:
            res, selected_features = rfe.fit(feature_names=selected_features)
        except Exception as e:
            logger.error(f"Error in RFE: {e}")
            break

        # projection
        rfe_train = rfe.project(features_df)
        rfe_test = rfe.project(test_features)

        # Check the shape of the projected data
        assert rfe_train.shape[1] == curr_num_features
        assert rfe_test.shape[1] == curr_num_features

        # train model
        model = RandomForestClassifier()
        model.fit(rfe_train, labels_df)
        acc = accuracy_score(test_labels, model.predict(rfe_test))
        f1 = f1_score(test_labels, model.predict(rfe_test), average="weighted")
        experiment_results["acc"].append(acc)
        experiment_results["f1"].append(f1)
        experiment_results["time"].append(time.perf_counter() - t0)

        logger.info(
            f"Step {step}: Feature Shape: {rfe_train.shape}, Acc Score = {acc:.4f}, F1 Score = {f1:.4f}"
        )

    res_df = pd.DataFrame(
        experiment_results,
        index=[i for i in range(len(experiment_results["acc"]))],
        columns=["acc", "f1", "time"],
    )
    results.append(res_df)

    res_df.to_csv(f"results_{dtype}_rfe.csv", index=False)
