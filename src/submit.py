import pathlib
import re
from typing import Any, List

import numpy as np
import pandas as pd
import torch
from rich.progress import track
from torch.utils.data import DataLoader
from xgboost import XGBRegressor

from models import CommonLitDataset, EmbeddingEncoder
from utils import timer


def text_length(data: pd.DataFrame) -> np.ndarray:
    results = data["text"].str.len().to_numpy().reshape(-1, 1)
    return results


def word_count(data: pd.DataFrame) -> np.ndarray:
    results = data["text"].str.split().str.len()
    return results.to_numpy().reshape(-1, 1)


def sentence_count(data: pd.DataFrame) -> np.ndarray:
    results = data["text"].str.split(".").str.len()
    return results.to_numpy().reshape(-1, 1)


def quoted_sentence_count(data: pd.DataFrame) -> np.ndarray:
    results = data["text"].apply(lambda x: len(re.findall(r'"(.*?)"', str(x))))
    return results.to_numpy().reshape(-1, 1)


def consecutive_dots_count(data: pd.DataFrame) -> np.ndarray:
    results = data["text"].apply(lambda x: len(re.findall(r"\.{3,4}", str(x))))
    return results.to_numpy().reshape(-1, 1)


def deberta_embeddings(data: pd.DataFrame) -> np.ndarray:
    model_name = "data/external/microsoft-deberta-v3-base"
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


def predict(X: np.ndarray, models: List[Any]) -> np.ndarray:
    pred = [model.predict(X) for model in models]
    return np.mean(pred, axis=0)


def main() -> None:
    N_FOLD = 5
    raw_dir = pathlib.Path("./data/raw")
    input_dir = pathlib.Path("./data/upload")

    test = pd.read_csv(raw_dir / "summaries_test.csv")

    funcs = [
        text_length,
        word_count,
        sentence_count,
        quoted_sentence_count,
        consecutive_dots_count,
        deberta_embeddings,
    ]

    features_tmp = []
    for func in funcs:
        features_tmp.append(func(test))
    features = np.concatenate(features_tmp, axis=1)

    sample_submission = pd.read_csv(raw_dir / "sample_submission.csv")

    for target_name in ["content", "wording"]:
        model = XGBRegressor()
        models = []
        for fold in range(N_FOLD):
            model.load_model(
                str(
                    input_dir
                    / f"xgb/seed=42/target={target_name}_fold={fold}.json"
                )
            )
            models.append(model)
        pred = predict(features, models)
        sample_submission[target_name] = pred

    print(sample_submission.head())


if __name__ == "__main__":
    with timer("main.py"):
        main()
