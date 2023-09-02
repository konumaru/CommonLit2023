import pathlib
import re
from typing import Any, List

import numpy as np
import pandas as pd

from utils import timer
from utils.io import load_pickle


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
    ]

    features_tmp = []
    for func in funcs:
        features_tmp.append(func(test))
    features = np.concatenate(features_tmp, axis=1)

    sample_submission = pd.read_csv(raw_dir / "sample_submission.csv")

    for target_name in ["content", "wording"]:
        models = [
            load_pickle(
                input_dir / f"xgb/{target_name}/seed=42/fold={fold}/model.pkl"
            )
            for fold in range(N_FOLD)
        ]
        pred = predict(features, models)
        sample_submission[target_name] = pred

    print(sample_submission.head())


if __name__ == "__main__":
    with timer("main.py"):
        main()
