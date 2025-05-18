import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from loguru import logger
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class PCAProjector:
    def __init__(self, n_components: int = None, scale: bool = True):
        """
        PCA wrapper with optional MinMax scaling.

        Args:
            n_components (int): Number of PCA components.
            scale (bool): Whether to apply MinMax scaling before PCA.
        """
        self.n_components = n_components
        self.scale = scale

    def before_fit(self):
        logger.info("Before PCA Fit")
        if self.n_components is None:
            raise ValueError("Number of components must be set before fitting PCA.")
        if self.scale:
            logger.info("Scaling data with MinMaxScaler")
        else:
            logger.info("No scaling applied")
        self.pca = PCA(n_components=self.n_components)
        self.metrics = {
            "explained_variance_ratio": [],
            "fit_time": None,
            "n_components": self.n_components,
        }
        self.scaler = MinMaxScaler() if self.scale else None

    def fit_transform(self, train_features: pd.DataFrame):
        self.before_fit()
        logger.info(
            f"Starting PCA with {self.n_components} components. Scaling: {self.scale}"
        )
        X_train = train_features.values
        if self.scale:
            X_train = self.scaler.fit_transform(X_train)

        t_start = time.perf_counter()
        X_train_pca = self.pca.fit_transform(X_train)
        t_end = time.perf_counter()

        self.metrics["fit_time"] = t_end - t_start
        self.metrics["explained_variance_ratio"] = (
            self.pca.explained_variance_ratio_.tolist()
        )

        logger.info(f"PCA completed in {self.metrics['fit_time']:.4f}s")
        # logger.info(
        #     f"Explained variance (first {self.n_components} components): {self.metrics['explained_variance_ratio']}"
        # )

        return X_train_pca

    def project(self, data: pd.DataFrame):
        """
        Project data using the fitted PCA model.

        Args:
            data (pd.DataFrame): Data to project.

        Returns:
            np.ndarray: Projected data.
        """
        if self.scale:
            data = self.scaler.transform(data)
        return self.pca.transform(data)


# if __name__ == "__main__":
#     projector = PCAProjector(n_components=72, scale=True)
#     projector.fit_transform(features_df)

#     train_projected = projector.project(features_df)
#     test_projected = projector.project(test_features)
