import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from utils import timer
from utils.feature import feature

FEATURE_DIR = "./data/feature"


@feature(FEATURE_DIR)
def dummpy_feature() -> np.ndarray:
    data = np.zeros((5, 100))
    return data

@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    feat_funcs = [
        dummpy_feature,
    ]

    for func in feat_funcs:
        func()



if __name__ == "__main__":
    with timer("main.py"):
        main()
