import logging
import os
from pathlib import Path

import hydra
import mlflow
import pyrootutils
import yaml
from hydra.utils import instantiate
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
    MODEL_NAME = f"{PRETRAINED_MODEL_DIR}/{cfg.models.model_name.lower()}.pt"
    log.info(f" Loading pretrained model from {MODEL_NAME}")
    model = YOLO(MODEL_NAME)
    results = model.train(
        data=cfg.datasets.dataset_yaml,
        **instantiate(cfg.trainer.params),
        **instantiate(cfg.datasets.augmentations),
    )
    return model, results


@hydra.main(version_base="1.3", config_path=str(root / "configs"), config_name="train.yaml")
def main(cfg: DictConfig):
    mlflow.mlflow.set_experiment(experiment_name=cfg.task_name)
    log.info(f"Using MLflow experiment: {cfg.task_name}")
    with mlflow.start_run():
        log.info(f"Started MLflow run ID: {mlflow.active_run().info.run_id}")
        params = instantiate(cfg.trainer.params)
        mlflow.log_params(params)
        mlflow.log_params(instantiate(cfg.datasets.augmentations))

        dataset_yaml_path = hydra.utils.to_absolute_path(cfg.datasets.dataset_yaml)
        mlflow.log_artifact(dataset_yaml_path, artifact_path="configs")
        log.info(f"Logged dataset config artifact: {dataset_yaml_path}")
        mlflow.log_param("model_name", cfg.models.model_name)

        model, results = train(cfg)

        mlflow.pytorch.log_model(model, "model")
        mlflow.log_metrics(results)

        metrics_results = model.val()

        log.info(f"Validation metrics: {metrics_results.results_dict.keys()}")
        mlflow.log_metrics(metrics_results.results_dict)


if __name__ == "__main__":
    main()
