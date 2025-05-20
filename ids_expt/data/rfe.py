import time
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class RFE:
    def __init__(
        self,
        model_cls,
        train_features: pd.DataFrame,
        train_labels: pd.Series,
        test_features: pd.DataFrame,
        test_labels: pd.Series,
        num_steps: int = 72,
        stopping_score: float = 0.5,
    ):
        self.model_cls = model_cls
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.num_steps = num_steps
        self.stopping_score = stopping_score
        self.result = []
        self.curr_step = 0
        self.feature_mask = None

        self.orig_feature_names = train_features.columns.tolist()

        self.scaler = MinMaxScaler()
        self.scaler.fit(self.train_features)
        self.train_features = self.scaler.transform(self.train_features)
        self.test_features = self.scaler.transform(self.test_features)

    def fit(self, feature_names: list | None = None):
        if feature_names is None:
            feature_names = self.orig_feature_names
        # init feat mask
        feature_mask = np.array([f in feature_names for f in self.orig_feature_names])
        t_start = time.perf_counter()
        x_train_sel = self.train_features[:, feature_mask]
        x_test_sel = self.test_features[:, feature_mask]
        current_indices = np.flatnonzero(feature_mask)
        current_feature_names = [self.orig_feature_names[i] for i in current_indices]
        logger.info(f"Current feature shape: {x_train_sel.shape}")
        # Train model
        model = self.model_cls()
        t0 = time.perf_counter()
        model.fit(x_train_sel, self.train_labels)
        logger.info(f"Training time: {time.perf_counter() - t0:.4f}s")
        # Evaluate
        score = model.score(x_test_sel, self.test_labels)
        logger.info(f"Score: {score:.4f}")
        importances = model.feature_importances_
        max_i = importances.argmax()
        min_i = importances.argmin()
        logger.info(
            f"Max importance: {importances[max_i]:.4f} ({current_feature_names[max_i]})"
        )
        logger.info(
            f"Min importance: {importances[min_i]:.4f} ({current_feature_names[min_i]})"
        )
        # Stopping criteria
        if score < self.stopping_score:
            logger.info(
                f"Score dropped below stopping threshold of {self.stopping_score}"
            )
            return
        if self.curr_step >= self.num_steps - 1:
            logger.info(f"Reached max number of steps: {self.num_steps}")
            return
        if feature_mask.sum() == 1:
            logger.info("Only one feature left, stopping")
            return
        # Eliminate least important feature
        remove_global_idx = current_indices[min_i]
        removed_feature_name = self.orig_feature_names[remove_global_idx]
        feature_mask[remove_global_idx] = False
        self.result.append(
            [
                self.curr_step,
                removed_feature_name,
                current_feature_names[max_i],
                current_feature_names[min_i],
                importances[max_i],
                importances[min_i],
                score,
                x_test_sel.shape[1],
            ]
        )
        logger.info(f"Removed feature: {removed_feature_name}")
        self.curr_step += 1
        logger.info(f"Total time: {time.perf_counter() - t_start:.4f}s")
        self.current_feature_names = [
            self.orig_feature_names[i] for i in np.flatnonzero(feature_mask)
        ]
        self.feature_mask = feature_mask
        return self.result, self.current_feature_names

    def project(self, data: pd.DataFrame):
        if self.scaler is not None:
            data = self.scaler.transform(data)

        data = data[:, self.feature_mask]

        return data
