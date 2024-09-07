import joblib

from src.pipelines.example.data import get_data
from src.pipelines.example.params import load_configs


def main() -> None:
    data_config, _ = load_configs()
    data_config.data_path.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(get_data(data_config), data_config.data_path)


if __name__ == "__main__":
    main()
