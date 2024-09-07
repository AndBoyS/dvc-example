from pathlib import Path
import joblib

from sklearn.pipeline import Pipeline

from src.pipelines.example.params import load_configs


def main() -> None:
    data_config, train_config = load_configs()

    X_train, X_test, y_train, y_test = joblib.load(data_config.data_path)
    model = train_config.load_model()

    pipeline_steps = [
        ("model", model),
    ]

    model = Pipeline(pipeline_steps)

    model.fit(X_train, y_train)

    Path(train_config.output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, train_config.output_path)


if __name__ == "__main__":
    main()
