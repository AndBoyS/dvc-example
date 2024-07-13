from typing import TYPE_CHECKING
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


if TYPE_CHECKING:
    from src.pipelines.example.params import DataConfig


def get_data(data_config: "DataConfig") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = make_classification(data_config.n_samples, data_config.n_features, n_classes=data_config.n_classes)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=data_config.test_size,
        random_state=data_config.random_state,
    )
    return X_train, X_test, y_train, y_test
