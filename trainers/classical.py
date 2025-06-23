import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from loguru import logger
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from ids_expt.models.ffnn import FFNN
from ids_expt.data.dataset import (
    DataSetConfig,
    SamplingMethod,
    CLFDataSet as DataSet,
    DFDataSet,
)
from ids_expt.core.defs import TOP_FEATURES, TOP_CIC_FEATURES
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def plot_classification_report(y_true, y_pred, labels, title, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    results = []
    output_dir = Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    for data_path in [
        Path("E:/MSc Works/IDS/data/cicflow_combined.csv"),
        Path(r"E:\MSc Works\IDS\data\cic_ctgan_merged_synthetic_data.csv"),
    ]:
        logger.info(f"Processing data from: {data_path}")
        feature_set = (
            TOP_CIC_FEATURES if "cic" in str(data_path).lower() else TOP_FEATURES
        )
        if "ctgan" not in str(data_path).lower():
            train_dataset, val_dataset = DFDataSet(
                config=DataSetConfig(
                    csv_path=data_path,
                    sampling_method=SamplingMethod.OVERSAMPLE,
                    max_data=-1,
                    train_ratio=0.75,
                    features=feature_set,
                )
            ).get_datasets()
            X_train = train_dataset.data[feature_set]
            y_train = train_dataset.data["Label"]
            X_val = val_dataset.data[feature_set]
            y_val = val_dataset.data["Label"]

        else:
            df_gen = pd.read_csv(data_path, low_memory=False)
            df_gen.columns = df_gen.columns.str.strip()
            df_orig = df_gen.query("is_synthetic!=True")
            df_train, df_test = train_test_split(
                df_orig, test_size=0.25, random_state=42, stratify=df_orig.Label
            )

            # add needed samples in df_train
            max_lbl_cnt = df_train["Label"].value_counts().max()
            for label in df_test["Label"].unique():
                if label == "NORMAL":
                    continue
                label_count = df_train["Label"].value_counts().get(label, 0)
                if label_count < max_lbl_cnt:
                    needed_samples = max_lbl_cnt - label_count
                    additional_samples = df_gen.query(
                        f"Label == '{label}' and is_synthetic == True"
                    ).sample(n=needed_samples, random_state=42, replace=False)
                    df_train = pd.concat(
                        [df_train, additional_samples], ignore_index=True
                    )

            df_test = df_test[TOP_CIC_FEATURES + ["Label"]]
            df_train = df_train[TOP_CIC_FEATURES + ["Label"]]
            scaler = StandardScaler()
            X_train = df_train.drop(columns=["Label"])
            X_val = df_test.drop(columns=["Label"])

            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            y_train = df_train["Label"]
            y_val = df_test["Label"]
            # save train/test data
            df_train.to_csv("cic_merged_train_data.csv", index=False)
            df_test.to_csv("cic_merged_test_data.csv", index=False)

        unique_labels = sorted(y_val.unique())

        for model in [
            RandomForestClassifier(n_estimators=100, random_state=42),
            GaussianNB(),
            DecisionTreeClassifier(random_state=42),
            KNeighborsClassifier(n_neighbors=5),
        ]:
            model_name = model.__class__.__name__
            logger.info(f"Training model: {model_name}")

            model.fit(X_train, y_train)

            predictions = model.predict(X_val)

            # predictions = model.predict(X_val)

            acc = accuracy_score(y_val, predictions)
            f1 = f1_score(y_val, predictions, average="weighted")
            prec = precision_score(y_val, predictions, average="weighted")
            rec = recall_score(y_val, predictions, average="weighted")

            logger.info(f"{model_name}:\n{classification_report(y_val, predictions)}")

            # Save confusion matrix as image
            img_file = output_dir / f"{data_path.stem}_{model_name}_report.png"
            plot_classification_report(
                y_val,
                predictions,
                labels=unique_labels,
                title=f"{model_name} on {data_path.stem}",
                save_path=img_file,
            )
            logger.info(f"Saved classification report to {img_file}")

            results.append(
                {
                    "data_type": data_path.stem,
                    "model": model_name,
                    "accuracy": acc,
                    "f1": f1,
                    "precision": prec,
                    "recall": rec,
                }
            )

    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv("model_evaluation_results.csv", index=False)
    logger.info("Saved results to model_evaluation_results.csv")
