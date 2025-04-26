import logging
import os
from pathlib import Path

import hydra
import mlflow
import pyrootutils
from omegaconf import DictConfig
from ultralytics import YOLO

log = logging.getLogger(__name__)
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

if os.getenv("DATA_ROOT") is None:
    os.environ["DATA_ROOT"] = ""


def train(cfg: DictConfig):
    PRETRAINED_MODEL_DIR = Path(cfg.paths.pretrained_model_dir)
    MODEL_NAME = f"{PRETRAINED_MODEL_DIR}/{cfg.model.model_name.lower()}.pt"
    log.info(f" Loading pretrained model from {MODEL_NAME}")
    model = YOLO(MODEL_NAME)
    results = model.train(
        data=cfg.datasets.dataset_yaml,
        **cfg.trainer.params,
        **cfg.datasets.augmentations,
    )
    return model, results


@hydra.main(version_base="1.3", config_path=str(root / "configs"), config_name="train.yaml")
def main(cfg: DictConfig):
    mlflow.experiment_name = cfg.task_name
    with mlflow.start_run():
        mlflow.log_params(cfg.trainer.params)
        mlflow.log_params(cfg.datasets.augmentations)
        mlflow.log_params(cfg.datasets.dataset_yaml)
        mlflow.log_param("model_name", cfg.model.model_name)

        model, results = train(cfg)

        mlflow.pytorch.log_model(model, "model")
        mlflow.log_metrics(results)

        metrics = model.validate()
        mlflow.log_metrics(metrics)


if __name__ == "__main__":
    main()
