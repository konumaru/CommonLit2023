import pathlib
import re
from typing import List

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from rich.progress import track
from torch.utils.data import DataLoader

from models import CommonLitDataset, EmbeddingEncoder
from utils import timer
from utils.feature import feature, load_feature

FEATURE_DIR = "./data/feature"


@feature(FEATURE_DIR)
def fold(data: pd.DataFrame) -> np.ndarray:
    return data["fold"].to_numpy().reshape(-1, 1)


@feature(FEATURE_DIR)
def content(data: pd.DataFrame) -> np.ndarray:
    return data["content"].to_numpy().reshape(-1, 1)


@feature(FEATURE_DIR)
def wording(data: pd.DataFrame) -> np.ndarray:
    return data["wording"].to_numpy().reshape(-1, 1)


@feature(FEATURE_DIR)
def text_length(data: pd.DataFrame) -> np.ndarray:
    results = data["text"].str.len().to_numpy().reshape(-1, 1)
    return results


@feature(FEATURE_DIR)
def word_count(data: pd.DataFrame) -> np.ndarray:
    results = data["text"].str.split().str.len()
    return results.to_numpy().reshape(-1, 1)


@feature(FEATURE_DIR)
def sentence_count(data: pd.DataFrame) -> np.ndarray:
    results = data["text"].str.split(".").str.len()
    return results.to_numpy().reshape(-1, 1)


@feature(FEATURE_DIR)
def quoted_sentence_count(data: pd.DataFrame) -> np.ndarray:
    results = data["text"].apply(lambda x: len(re.findall(r'"(.*?)"', str(x))))
    return results.to_numpy().reshape(-1, 1)


@feature(FEATURE_DIR)
def consecutive_dots_count(data: pd.DataFrame) -> np.ndarray:
    results = data["text"].apply(lambda x: len(re.findall(r"\.{3,4}", str(x))))
    return results.to_numpy().reshape(-1, 1)


@feature(FEATURE_DIR)
def deberta_embeddings(data: pd.DataFrame) -> np.ndarray:
    model_name = "microsoft/deberta-v3-base"
    dataset = CommonLitDataset(data, model_name, max_len=512)
    dataloader = DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=8
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EmbeddingEncoder(model_name, device=device)

    model.to(device)
    model.eval()
    with torch.no_grad():
        embeddings_batch = []
        for batch in track(dataloader):
            z = model(batch)
            embeddings_batch.append(z.detach().cpu().numpy())
        embeddings = np.concatenate(embeddings_batch, axis=0)

    return embeddings


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    input_dir = pathlib.Path("./data/preprocessing")

    train = pd.read_csv(input_dir / "train.csv")

    funcs = [
        fold,
        content,
        wording,
        text_length,
        word_count,
        sentence_count,
        quoted_sentence_count,
        consecutive_dots_count,
        deberta_embeddings,
    ]

    for func in funcs:
        func(train)

    feature_names = [func.__name__ for func in funcs]
    features = load_feature(FEATURE_DIR, feature_names)
    print(pd.DataFrame(features).head())


if __name__ == "__main__":
    with timer("Create feature"):
        main()
