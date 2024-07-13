from typing import Literal, NamedTuple, Type

from sklearn.ensemble import RandomForestClassifier


class DataConfig(NamedTuple):
    n_samples: int = 100
    n_features: int = 10
    n_classes: int = 2
    test_size: float = 0.2
    random_state: int = 42


class RandomForestConfig(NamedTuple):
    n_estimators: int = 100
    criterion: Literal["gini", "entropy", "log_loss"] = "gini"
    max_depth: int | None = None
    min_samples_split: float | int = 2


model_type = Literal["random_forest"]


# If model type is changed, model config needs to be edited in dvc.yaml
class TrainConfig:
    model: model_type = "random_forest"
    output_path = "data/models/example/random_forest.pth"
    artifact_name = "example/random_forest"
    model_constructor: Type = RandomForestClassifier
    metrics_train_path = "dvclive_train"
    metrics_eval_path = "dvclive_eval"

    model_config_dict: dict[model_type, NamedTuple] = {
        "random_forest": RandomForestConfig(),
    }

    @property
    def model_config(self) -> NamedTuple:
        return self.model_config_dict[self.model]
