from dvclive import Live  # type: ignore[attr-defined]
import joblib
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from src.pipelines.example.data import get_data
from src.pipelines.example.params import load_configs


def main() -> None:
    data_config, train_config = load_configs()
    X_train, X_test, y_train, y_test = get_data(data_config)
    model: Pipeline = joblib.load(train_config.output_path)

    pred_test = model.predict(X_test)

    with Live(str(train_config.log_eval_path)) as logger:
        f1 = f1_score(y_true=y_test, y_pred=pred_test)
        logger.log_metric("f1_test", float(f1))


if __name__ == "__main__":
    main()
