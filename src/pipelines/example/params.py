from pathlib import Path
from typing import Literal, Type

from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier

from src.dvc import TrainConfigBase
from src.filesystem import load_yaml
from src.pipelines.example.data import DataConfig


PARAMS_PATH = Path(__file__).parent / "params.yaml"


class RandomForestConfig(BaseModel):
    n_estimators: int
    criterion: Literal["gini", "entropy", "log_loss"]
    max_depth: int | None
    min_samples_split: float | int
    struct: Type = RandomForestClassifier


model_type = Literal["random_forest"]


class TrainConfig(TrainConfigBase):
    model: model_type
    output_path: Path
    log_root: Path
    random_forest_config: RandomForestConfig


def load_configs() -> tuple[DataConfig, TrainConfig]:
    config_data = load_yaml(PARAMS_PATH)

    data_config = DataConfig.model_validate(config_data["data_config"])
    train_config = TrainConfig.model_validate(config_data["train_config"])
    return data_config, train_config
