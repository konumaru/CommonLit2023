import pathlib

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import KFold

from utils import timer


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    output_dir = pathlib.Path("./data/preprocessing")

    summaries_train = pd.read_csv("./data/raw/summaries_train.csv")
    prompts_train = pd.read_csv("./data/raw/prompts_train.csv")

    train = pd.merge(
        prompts_train, summaries_train, on="prompt_id", how="right"
    )
    cv = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=42)

    train = train.assign(fold=0)
    for fold, (_, valid_index) in enumerate(cv.split(train)):
        train.loc[valid_index, "fold"] = fold

    train.to_csv(output_dir / "train.csv", index=False)


if __name__ == "__main__":
    with timer("main.py"):
        main()
