from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> Any:
    """Load yaml file.

    Arguments:
        path -- path to yaml file

    Returns:
        Contents of the yaml
    """
    with open(path, "rt", encoding="utf-8") as f:
        return yaml.safe_load(f)
