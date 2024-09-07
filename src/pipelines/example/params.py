from pathlib import Path
from typing import Any, Literal, Type

from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier

from src.filesystem import load_yaml


PARAMS_PATH = Path(__file__).parent / "params.yaml"


class DataConfig(BaseModel):
    data_path: Path
    n_samples: int
    n_features: int
    n_classes: int
    test_size: float
    random_state: int


class RandomForestConfig(BaseModel):
    n_estimators: int
    criterion: Literal["gini", "entropy", "log_loss"]
    max_depth: int | None
    min_samples_split: float | int
    struct: Type = RandomForestClassifier


model_type = Literal["random_forest"]


class TrainConfig(BaseModel):
    model: model_type
    output_path: Path
    log_root: Path
    random_forest_config: RandomForestConfig

    @property
    def log_train_path(self) -> Path:
        return self.log_root / "train"

    @property
    def log_eval_path(self) -> Path:
        return self.log_root / "eval"

    @property
    def model_param_config(self) -> BaseModel:
        return getattr(self, f"{self.model}_config")

    def load_model(self) -> Any:
        model_struct: Type = getattr(self.model_param_config, "struct")
        model_params = self.model_param_config.model_dump()
        model_params.pop("struct")
        return model_struct(**model_params)


def load_configs() -> tuple[DataConfig, TrainConfig]:
    config_data = load_yaml(PARAMS_PATH)

    data_config = DataConfig.model_validate(config_data["data_config"])
    train_config = TrainConfig.model_validate(config_data["train_config"])
    return data_config, train_config
