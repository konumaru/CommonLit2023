import pathlib
from typing import Any, Union

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestRegressor

from metric import mcrmse
from utils import timer
from utils.feature import load_feature
from utils.io import load_pickle, save_pickle


def get_model(model_name: str = "rf", seed: int = 42) -> Any:
    return RandomForestRegressor()


def train(
    model_name: str,
    target_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: Union[np.ndarray, None] = None,
    y_valid: Union[np.ndarray, None] = None,
    gruop_train: Union[np.ndarray, None] = None,
    gruop_valid: Union[np.ndarray, None] = None,
    seed=42,
):
    model_dir = pathlib.Path(
        f"./data/model/{model_name}/seed={seed}/target={target_name}"
    )
    model_dir.mkdir(exist_ok=True, parents=True)

    model = get_model(model_name, seed)
    model.fit(X_train, y_train)
    save_pickle(str(model_dir / "model.pkl"), model)


def predict(
    model_name: str,
    target_name: str,
    X_valid: Union[np.ndarray, None] = None,
    y_valid: Union[np.ndarray, None] = None,
    gruop_valid: Union[np.ndarray, None] = None,
    seed: int = 42,
) -> np.ndarray:
    model_dir = pathlib.Path(
        f"./data/model/{model_name}/seed={seed}/target={target_name}"
    )

    pred = np.zeros_like(y_valid)
    model = load_pickle(str(model_dir / "model.pkl"))
    pred = model.predict(X_valid)

    return pred


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    input_dir = pathlib.Path("./data/feature")

    folds = load_pickle(input_dir / "fold.pkl").ravel()
    featres = load_feature(input_dir, cfg.features)

    oof = np.zeros(shape=(len(folds), 2))
    targets = oof.copy()
    for i, target_name in enumerate(["content", "wording"]):
        print("Target:", target_name)
        target = load_pickle(input_dir / f"{target_name}.pkl").ravel()
        targets[:, i] = target
        for fold in range(cfg.n_splits):
            print(f"Fold: {fold}")
            X_train = featres[folds != fold]
            y_train = target[folds != fold]
            X_valid = featres[folds == fold]
            y_valid = target[folds == fold]

            train(
                cfg.model.name,
                target_name,
                X_train,
                y_train,
                X_valid,
                y_valid,
                seed=cfg.seed,
            )

            pred = predict(
                cfg.model.name,
                target_name,
                X_valid,
                y_valid,
                seed=cfg.seed,
            )
            oof[folds == fold, i] = pred

    score = mcrmse(targets, oof)
    print(score)


if __name__ == "__main__":
    with timer("main.py"):
        main()
