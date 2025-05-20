from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone, BaseEstimator
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger


class PermutationFeatureImportance:
    """
    src: https://christophm.github.io/interpretable-ml-book/feature-importance.html#theory
    * Instead of simple MSE, this uses log_loss.
    * Feature importance is permuted_log_loss/original_log_loss - 1.
    * When the feature has no effect, importance is 0.
    * When the feature has a strong effect, importance is > 0.
    * When the feature has a negative effect, importance is < 0.
    """

    def __init__(
        self,
        model_object: BaseEstimator,
        X,
        y,
        split_size=0.2,
        random_state=42,
        k_fold: int = 5,
        handle_imbalance: bool = True,
        oversampling: bool = False,
        normal_label: str = "NORMAL",
        classwise: bool = False,
        out_dir: str = r"E:\MSc Works\IDS\assets",
        image_exts: list[str] = ["png", "pdf"],
        permute_rate: float = 1.0,
    ):
        """
        Initialize the PermutationFeatureImportance class.
        Parameters:
        ----------
        * model_object: The model object to use for prediction.
        * X: The feature data.
        * y: The target data.
        * split_size: The size of the test set.
        * random_state: The random state for reproducibility.
        * k_fold: The number of folds for cross-validation. Higher values are slower but more accurate.
        * handle_imbalance: Whether to handle class imbalance.
        * oversampling: Whether to use oversampling or undersampling.
        * normal_label: The label for the normal class.
        * classwise: Whether to calculate feature importance for each class separately.
        * out_dir: The output directory for saving results.
        * image_exts: The image file extensions for saving plots.
        * permute_rate: The rate of permutation for feature importance calculation.
        """
        self.model_object = model_object
        self.X = X
        self.y = y
        self.permute_rate = permute_rate
        self.split_size = split_size
        self.random_state = np.random.RandomState(random_state)
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(
            self.X,
            self.y,
            test_size=self.split_size,
            random_state=self.random_state,
            stratify=self.y,
        )
        self.orig_train_X = self.train_X.copy()
        self.orig_train_y = self.train_y.copy()
        self.orig_test_X = self.test_X.copy()
        self.orig_test_y = self.test_y.copy()

        self.k_fold = k_fold
        self.handle_imbalance = handle_imbalance
        self.oversampling = oversampling
        self.normal_label = normal_label
        self.classwise = classwise
        self.results_df = None
        self.out_dir = out_dir
        if self.out_dir is not None:
            from pathlib import Path

            self.out_dir = Path(self.out_dir)
            if not self.out_dir.exists():
                logger.info(f"Creating directory: {self.out_dir}")
                self.out_dir.mkdir(parents=True, exist_ok=True)

        self.image_exts = image_exts

    def _fit(self, X, y):
        """
        Fit the model to the data.
        """
        # Clone the model to avoid modifying the original
        model_clone = clone(self.model_object)
        # Fit the model to the data
        model_clone.fit(X, y)
        return model_clone

    def fit(self):
        final_results = []
        if self.classwise:
            all_labels = self.y.unique()
            for label in all_labels:
                if label == self.normal_label:
                    continue
                selected_labels = [self.normal_label, label]
                logger.info(f"Fitting model for label: {label}")
                # Filter the train/test dataset for the selected labels
                self.train_X = self.orig_train_X[
                    self.orig_train_y.isin(selected_labels)
                ].copy()
                self.train_y = self.orig_train_y[
                    self.orig_train_y.isin(selected_labels)
                ].copy()
                self.test_X = self.orig_test_X[
                    self.orig_test_y.isin(selected_labels)
                ].copy()
                self.test_y = self.orig_test_y[
                    self.orig_test_y.isin(selected_labels)
                ].copy()
                # Fit the model
                results = self.fit_one()
                results["label"] = label
                final_results.append(results)
                self.plot(
                    results_df=results,
                    plot_label=label,
                )

        logger.info("Fitting model for all labels")
        # Fit the model
        results = self.fit_one()
        results["label"] = "All"
        final_results.append(results)
        self.plot(
            results_df=results,
            plot_label="All",
        )
        # Concatenate all results
        results = pd.concat(final_results, ignore_index=True)

        self.results_df = results
        if self.out_dir is not None:
            results.to_csv(
                f"{self.out_dir}/pfi_results.csv",
                index=False,
            )
            logger.info(f"Saved results to {self.out_dir}/pfi_results.csv")
        return results

    def fit_one(self):
        if self.handle_imbalance:
            class_counts = self.train_y.value_counts()
            logger.info(
                f"Class counts before imbalance handling: {class_counts.to_dict()}"
            )
            if self.oversampling:
                # Handle imbalance using oversampling
                from imblearn.over_sampling import SMOTE

                smote = SMOTE(random_state=self.random_state)
                self.train_X, self.train_y = smote.fit_resample(
                    self.train_X, self.train_y
                )
            else:
                # Handle imbalance using undersampling
                from imblearn.under_sampling import RandomUnderSampler

                rus = RandomUnderSampler(random_state=self.random_state)
                self.train_X, self.train_y = rus.fit_resample(
                    self.train_X, self.train_y
                )
            logger.info(
                f"Class counts after imbalance handling: {self.train_y.value_counts().to_dict()}"
            )
        trained_model = self._fit(self.train_X, self.train_y)
        y_test = self.test_y
        y_pred = trained_model.predict(self.test_X)

        original_acc = accuracy_score(y_test, y_pred)
        logger.info(f"Original Accuracy: {original_acc:.4f}")

        # Encode string labels to integers
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_test)
        y_pred_encoded = label_encoder.transform(y_pred)
        y_pred_proba = trained_model.predict_proba(self.test_X)
        original_log_loss = log_loss(y_test, y_pred_proba)
        if len(self.y.unique()) == 2:
            original_f1 = f1_score(y_encoded, y_pred_encoded, average="binary")
        else:
            original_f1 = f1_score(y_encoded, y_pred_encoded, average="weighted")
        logger.info(f"Original F1: {original_f1:.4f}")
        results = {
            "accuracy": [original_acc],
            "f1": [original_f1],
            "log_loss": [original_log_loss],
            "feature": ["All"],
            "importance": [-1],
            "importance_std": [0],
        }

        for feature in tqdm(self.X.columns):
            # Save the original values
            original_values = self.test_X[feature].copy()
            feat_acc = 0
            feat_f1 = 0
            feat_log_loss_value = 0
            importances = []

            for fold in range(self.k_fold):
                # Shuffle the feature values
                len_samples = len(self.test_X[feature])
                values = original_values.values.copy()

                # Randomly select a subset of indices to permute
                if self.permute_rate < 1.0:
                    # randomly select indices to permute
                    num_permute = int(len_samples * self.permute_rate)
                    idxs = self.random_state.choice(
                        len_samples, num_permute, replace=False
                    )
                    # permute the selected indices
                    values[idxs] = self.random_state.permutation(values[idxs])
                else:
                    num_permute = len_samples
                    values = self.random_state.permutation(values)

                # Replace the feature values with the permuted values
                self.test_X[feature] = values

                # Get predictions
                y_pred = trained_model.predict(self.test_X)
                y_pred_encoded = label_encoder.transform(y_pred)

                # Calculate metrics
                acc = accuracy_score(y_test, y_pred)
                if len(self.y.unique()) == 2:
                    f1 = f1_score(y_encoded, y_pred_encoded, average="binary")
                else:
                    f1 = f1_score(y_encoded, y_pred_encoded, average="weighted")
                # log loss between curr prediction and true encoded labels
                log_loss_value = log_loss(
                    y_encoded, trained_model.predict_proba(self.test_X)
                )
                # add metrics
                feat_acc += acc
                feat_f1 += f1
                feat_log_loss_value += log_loss_value
                self.test_X[feature] = original_values
                importances.append(log_loss_value / original_log_loss - 1)
            feat_acc /= self.k_fold
            feat_f1 /= self.k_fold
            feat_log_loss_value /= self.k_fold
            # feat_importance = feat_log_loss_value / original_log_loss - 1
            avg_importance = np.mean(importances)
            std_importance = np.std(importances)

            results["accuracy"].append(feat_acc)
            results["f1"].append(feat_f1)
            results["log_loss"].append(feat_log_loss_value)
            results["feature"].append(feature)
            results["importance"].append(avg_importance)
            results["importance_std"].append(std_importance)

            logger.info(
                f"feature: {feature}, accuracy: {feat_acc:.4f}, f1: {feat_f1:.4f}, log_loss: {feat_log_loss_value:.4f}, importance: {avg_importance:.4f}"
            )
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by="importance", ascending=False)
        results_df = results_df.reset_index(drop=True)
        return results_df

    def plot(
        self,
        figsize=(10, 16),
        results_df: pd.DataFrame = None,
        plot_label: str = None,
    ):
        """
        Plot the results using Seaborn with error bars.
        """
        if self.results_df is None and results_df is None:
            raise ValueError("No results to plot. Please run fit() first.")
        if self.results_df is not None and plot_label is None:
            results_df = self.results_df
        results_df = results_df.query("feature != 'All'")
        for label in results_df["label"].unique():
            if plot_label is not None and label != plot_label:
                continue
            label_df = results_df[results_df["label"] == label]
            if label_df.empty:
                continue

            fig, ax = plt.subplots(figsize=figsize)
            sns.barplot(
                x="importance",
                y="feature",
                data=label_df,
                ci=None,  # Disable automatic confidence intervals
                palette="viridis",  # Optional: Use a color palette
                ax=ax,
            )

            # Add error bars manually
            ax.errorbar(
                label_df["importance"],
                range(len(label_df["feature"])),
                xerr=label_df["importance_std"],
                fmt="none",
                capsize=5,
                color="black",
            )

            ax.set_title(f"Permutation Feature Importance for `{label}` Feature")
            ax.set_xlabel("Importance")
            ax.set_ylabel("Feature")
            plt.tight_layout()
            if self.out_dir is not None:
                for ext in self.image_exts:
                    file_path = f"{self.out_dir}/pfi_{label}.{ext}"
                    plt.savefig(
                        file_path,
                        dpi=300,
                        bbox_inches="tight",
                    )
                    logger.info(f"Saved plot to {file_path}")
            # plt.show()


if __name__ == "__main__":
    df = data_df["45_timeout"].sample(n=15000, replace=False, random_state=42)
    X = df.drop(columns=["Label"])
    X = X.select_dtypes(include=[np.number])
    features = [
        f
        for f in X.columns.tolist()
        if f not in ["frameSrc", "frameDst", "frameProtocol"] or "Unnamed" not in f
    ]
    X = X[features]
    pfi = PermutationFeatureImportance(
        model_object=RandomForestClassifier(random_state=42),
        X=X,
        y=df["Label"],
        split_size=0.2,
        random_state=42,
        k_fold=5,
        handle_imbalance=True,
        oversampling=False,
        classwise=True,
    )
    results = pfi.fit()
    pfi.plot()
