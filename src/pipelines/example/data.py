from pathlib import Path
import numpy as np
from pydantic import BaseModel
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


class DataConfig(BaseModel):
    data_path: Path
    n_samples: int
    n_features: int
    n_classes: int
    test_size: float
    random_state: int


def get_data(data_config: DataConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = make_classification(data_config.n_samples, data_config.n_features, n_classes=data_config.n_classes)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=data_config.test_size,
        random_state=data_config.random_state,
    )
    return X_train, X_test, y_train, y_test
