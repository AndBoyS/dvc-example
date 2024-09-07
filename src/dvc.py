from pathlib import Path
from typing import Any, Type
from pydantic import BaseModel


class MlModelConfig(BaseModel):
    struct: Type


class TrainConfigBase(BaseModel):
    model: str
    log_root: Path

    @property
    def log_train_path(self) -> Path:
        return self.log_root / "train"

    @property
    def log_eval_path(self) -> Path:
        return self.log_root / "eval"

    @property
    def model_param_config(self) -> MlModelConfig:
        return getattr(self, f"{self.model}_config")

    def load_model(self) -> Any:
        model_struct = self.model_param_config.struct
        model_params = self.model_param_config.model_dump()
        model_params.pop("struct")
        return model_struct(**model_params)
