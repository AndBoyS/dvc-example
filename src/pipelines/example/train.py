from pathlib import Path
import joblib

from dvclive import Live
from sklearn.pipeline import Pipeline

from src.pipelines.example.data import get_data
from src.pipelines.example.params import DataConfig, TrainConfig


def main() -> None:
    train_config = TrainConfig()
    data_config = DataConfig()

    X_train, X_test, y_train, y_test = get_data(data_config)

    model_config = train_config.model_config
    model = train_config.model_constructor(**model_config._asdict())

    pipeline_steps = [
        ("model", model),
    ]

    model = Pipeline(pipeline_steps)

    model.fit(X_train, y_train)

    Path(train_config.output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, train_config.output_path)

    with Live(train_config.metrics_train_path) as logger:
        logger.log_artifact(train_config.output_path, type="model", name=train_config.artifact_name)


if __name__ == "__main__":
    main()
