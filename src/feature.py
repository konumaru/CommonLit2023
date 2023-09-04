import pathlib
import re
from typing import List

import hydra
import numpy as np
import pandas as pd
import torch
from nltk.corpus import stopwords
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


def word_overlap_counter(row) -> int:
    STOP_WORDS = set(stopwords.words("english"))  # type: ignore

    def check_is_stop_word(word):
        return word in STOP_WORDS

    prompt_words = row["prompt_text"]
    summary_words = row["text"]
    if STOP_WORDS:
        prompt_words = list(filter(check_is_stop_word, prompt_words))
        summary_words = list(filter(check_is_stop_word, summary_words))
    return len(set(prompt_words).intersection(set(summary_words)))


def ngram_co_occurrence_counter(row, n: int = 2) -> int:
    def ngrams(token, n):
        ngrams = zip(*[token[i:] for i in range(n)])
        return [" ".join(ngram) for ngram in ngrams]

    original_tokens = row["prompt_text"].split(" ")
    summary_tokens = row["text"].split(" ")

    original_ngrams = set(ngrams(original_tokens, n))
    summary_ngrams = set(ngrams(summary_tokens, n))

    common_ngrams = original_ngrams.intersection(summary_ngrams)
    return len(common_ngrams)


@feature(FEATURE_DIR)
def word_overlap_count(data: pd.DataFrame) -> np.ndarray:
    results = data.apply(word_overlap_counter, axis=1)
    return results.to_numpy().reshape(-1, 1)


@feature(FEATURE_DIR)
def ngram_co_occurrence_count(data: pd.DataFrame) -> np.ndarray:
    bi_gram = data.apply(ngram_co_occurrence_counter, axis=1, args=(2,))
    tr_gram = data.apply(ngram_co_occurrence_counter, axis=1, args=(3,))

    results = pd.concat([bi_gram, tr_gram], axis=1)
    return results.to_numpy()


def encode_embedding(model_name: str, input_texts: List[str]) -> np.ndarray:
    dataset = CommonLitDataset(input_texts, model_name, max_len=512)
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


@feature(FEATURE_DIR)
def deberta_text_embeddings(data: pd.DataFrame) -> np.ndarray:
    model_name = "microsoft/deberta-v3-base"
    embeddings = encode_embedding(model_name, data["text"].tolist())
    return embeddings


@feature(FEATURE_DIR)
def deberta_prompt_embeddings(data: pd.DataFrame) -> np.ndarray:
    model_name = "microsoft/deberta-v3-base"
    embeddings = encode_embedding(model_name, data["prompt_text"].tolist())
    return embeddings


def create_features(data: pd.DataFrame):
    funcs = [
        fold,
        content,
        wording,
        text_length,
        word_count,
        sentence_count,
        quoted_sentence_count,
        consecutive_dots_count,
        word_overlap_count,
        ngram_co_occurrence_count,
        deberta_text_embeddings,
        deberta_prompt_embeddings,
    ]

    for func in funcs:
        func(data)

    feature_names = [func.__name__ for func in funcs]
    features = load_feature(FEATURE_DIR, feature_names)
    return features


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    input_dir = pathlib.Path("./data/preprocessing")

    train = pd.read_csv(input_dir / "train.csv")

    features = create_features(train)
    print(pd.DataFrame(features).head())


if __name__ == "__main__":
    with timer("Create feature"):
        main()
