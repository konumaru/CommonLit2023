import pathlib
from typing import Union

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from metric import mcrmse
from utils import timer
from utils.feature import load_feature
from utils.io import load_pickle, save_pickle


def fit_rf(
    params,
    X_train: np.ndarray,
    y_train: np.ndarray,
    save_filepath: str,
    seed: int = 42,
) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=100, random_state=seed)
    model.fit(X_train, y_train)
    save_pickle(
        str(pathlib.Path(save_filepath) / f"{save_filepath}.pkl"), model
    )
    return model


def fit_xgb(
    params,
    save_filepath: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: Union[np.ndarray, None] = None,
    y_valid: Union[np.ndarray, None] = None,
    seed: int = 42,
) -> XGBRegressor:
    model = XGBRegressor(**params)
    model.set_params(random_state=seed)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        verbose=50,
    )
    model.save_model(save_filepath + ".json")
    return model


def train(cfg: DictConfig) -> None:
    features_dir = pathlib.Path(cfg.path.features)

    features = load_feature(features_dir, cfg.features)
    folds = load_pickle(features_dir / "fold.pkl").ravel()
    oof = np.zeros(shape=(len(features), 2))
    targets = oof.copy()

    model_dir_suffix = f"{cfg.model.name}/seed={cfg.seed}/"
    model_dir = pathlib.Path(cfg.path.model) / model_dir_suffix
    model_dir.mkdir(exist_ok=True, parents=True)

    for i, target_name in enumerate(["content", "wording"]):
        print("Target:", target_name)
        target = load_pickle(features_dir / f"{target_name}.pkl").ravel()
        targets[:, i] = target
        for fold in range(cfg.n_splits):
            print(f"Fold: {fold}")
            X_train = features[folds != fold]
            y_train = target[folds != fold]
            X_valid = features[folds == fold]
            y_valid = target[folds == fold]

            saved_filename = f"target={target_name}_fold={fold}"
            if cfg.model.name == "rf":
                model = fit_rf(
                    cfg.model.params,
                    X_train,
                    y_train,
                    str(model_dir / saved_filename),
                )
                oof[folds == fold, i] = model.predict(X_valid)
            elif cfg.model.name == "xgb":
                model = fit_xgb(
                    cfg.model.params,
                    str(model_dir / saved_filename),
                    X_train,
                    y_train,
                    X_valid,
                    y_valid,
                )
                oof[folds == fold, i] = model.predict(X_valid)

    output_dir = pathlib.Path(cfg.path.train)
    save_pickle(str(output_dir / "oof.pkl"), oof)


def evaluate(cfg: DictConfig) -> None:
    features_dir = pathlib.Path(cfg.path.features)
    train_output_dir = pathlib.Path(cfg.path.train)

    oof = load_pickle(str(train_output_dir / "oof.pkl"))

    targets = np.zeros_like(oof)
    for i, target_name in enumerate(["content", "wording"]):
        target = load_pickle(features_dir / f"{target_name}.pkl").ravel()
        targets[:, i] = target

    score = mcrmse(targets, oof)
    print(score)


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    train(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    with timer("main.py"):
        main()
