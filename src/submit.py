import pathlib
import re
from typing import Any, List

import numpy as np
import pandas as pd
import torch
from nltk.corpus import stopwords
from rich.progress import track
from spellchecker import SpellChecker
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


def quotes_counter(row: pd.Series):
    summary = row["text"]
    text = row["prompt_text"]

    quotes_from_summary = re.findall(r'"([^"]*)"', summary)
    if len(quotes_from_summary) > 0:
        return [quote in text for quote in quotes_from_summary].count(True)
    else:
        return 0


def quotes_count(data: pd.DataFrame) -> np.ndarray:
    results = data.apply(quotes_counter, axis=1)
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


def word_overlap_count(data: pd.DataFrame) -> np.ndarray:
    results = data.apply(word_overlap_counter, axis=1)
    return results.to_numpy().reshape(-1, 1)


def ngram_co_occurrence_count(data: pd.DataFrame) -> np.ndarray:
    bi_gram = data.apply(ngram_co_occurrence_counter, axis=1, args=(2,))
    tr_gram = data.apply(ngram_co_occurrence_counter, axis=1, args=(3,))

    results = pd.concat([bi_gram, tr_gram], axis=1)
    return results.to_numpy()


def spell_miss_count(data: pd.DataFrame) -> np.ndarray:
    def counter(row: pd.Series) -> int:
        words = row["text"].split(" ")
        return len(SpellChecker().unknown(words))

    results = data.apply(counter, axis=1).to_numpy()
    return results.reshape(-1, 1)


def encode_embedding(model_name: str, input_texts: List[str]) -> np.ndarray:
    dataset = CommonLitDataset(input_texts, model_name, max_len=512)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8)

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


def deberta_text_embeddings(data: pd.DataFrame) -> np.ndarray:
    model_name = "microsoft/deberta-v3-base"
    embeddings = encode_embedding(model_name, data["text"].tolist())
    return embeddings


def deberta_prompt_embeddings(data: pd.DataFrame) -> np.ndarray:
    model_name = "microsoft/deberta-v3-base"
    embeddings = encode_embedding(model_name, data["prompt_text"].tolist())
    return embeddings


def create_features(data: pd.DataFrame):
    funcs = [
        text_length,
        word_count,
        sentence_count,
        quoted_sentence_count,
        consecutive_dots_count,
        quotes_count,
        word_overlap_count,
        spell_miss_count,
        ngram_co_occurrence_count,
        deberta_text_embeddings,
    ]

    features_tmp = []
    for func in funcs:
        results = func(data)
        features_tmp.append(results)

    features = np.concatenate(features_tmp, axis=1)
    return features


def predict(X: np.ndarray, models: List[Any]) -> np.ndarray:
    pred = [model.predict(X) for model in models]
    return np.mean(pred, axis=0)


def main() -> None:
    N_FOLD = 4
    raw_dir = pathlib.Path("./data/raw")
    input_dir = pathlib.Path("./data/upload")

    summaries = pd.read_csv(raw_dir / "summaries_test.csv")
    prompts = pd.read_csv(raw_dir / "prompts_test.csv")
    sample_submission = pd.read_csv(raw_dir / "sample_submission.csv")

    test = pd.merge(prompts, summaries, on="prompt_id", how="right")
    features = create_features(test)

    for target_name in ["content", "wording"]:
        model = XGBRegressor()
        models = []
        for fold in range(N_FOLD):
            model.load_model(
                str(input_dir / f"xgb/seed=42/target={target_name}_fold={fold}.json")
            )
            models.append(model)
        pred = predict(features, models)
        sample_submission[target_name] = pred

    print(sample_submission.head())


if __name__ == "__main__":
    with timer("main.py"):
        main()
