import pathlib
import re
from typing import Any, List

import lightgbm as lgb
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from spellchecker import SpellChecker
from transformers import AutoModelForSequenceClassification
from xgboost import XGBRegressor

from finetune import CommonLitDataset
from finetune import predict as predict_finetuned_model
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


def word_overlap_counter(row: pd.Series) -> int:
    STOP_WORDS = set(stopwords.words("english"))  # type: ignore

    def check_is_stop_word(word):
        return word in STOP_WORDS

    prompt_words = row["prompt_text"]
    summary_words = row["text"]
    if STOP_WORDS:
        prompt_words = list(filter(check_is_stop_word, prompt_words))
        summary_words = list(filter(check_is_stop_word, summary_words))
    return len(set(prompt_words).intersection(set(summary_words)))


def ngram_co_occurrence_counter(row: pd.Series, n: int = 2) -> int:
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


def round_to_5(n):
    return round(n / 5) * 5


def target_encoded_word_count(data: pd.DataFrame) -> np.ndarray:
    _data = data.copy()
    encoding_map = pd.read_csv(
        "data/preprocessed/target_encoded_word_count.csv"
    )
    f = _data["text"].str.split().str.len()
    _data["clipped_word_count"] = (
        pd.Series(f.ravel()).clip(25, 200).apply(round_to_5)
    )

    _data = pd.merge(_data, encoding_map, on="clipped_word_count", how="left")
    results = _data[["content", "wording"]].to_numpy()
    return results


def create_features(data: pd.DataFrame) -> np.ndarray:
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
        target_encoded_word_count,
    ]

    features = np.concatenate([func(data) for func in funcs], axis=1)
    return features


def get_finetuned_model_preds(
    data: pd.DataFrame, num_splits: int
) -> np.ndarray:
    model_dir = pathlib.Path("data/external/finetune-debertav3-training")

    preds = []
    for fold in range(num_splits):
        model_path = model_dir / f"finetuned-deberta-v3-base-fold{fold}"
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=2
        )
        dataset = CommonLitDataset(
            data, "test", "microsoft/deberta-v3-base", is_train=False
        )
        pred = predict_finetuned_model(model, dataset)
        preds.append(pred)
    return np.mean(preds, axis=0)


def predict_xgb(X: np.ndarray, models: List[Any]) -> np.ndarray:
    pred = [model.predict(X) for model in models]
    return np.mean(pred, axis=0)


def predict_lgbm(X: np.ndarray, models: List[Any]) -> np.ndarray:
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
    text_features = create_features(test)
    preds_deberta = get_finetuned_model_preds(test, N_FOLD)

    features = np.concatenate([text_features, preds_deberta], axis=1)

    for target_name in ["content", "wording"]:
        model_xgb = XGBRegressor()
        models_xgb = []

        models_lgbm = []

        for fold in range(N_FOLD):
            model_xgb.load_model(
                str(
                    input_dir
                    / f"xgb/seed=42/target={target_name}_fold={fold}.json"
                )
            )
            models_xgb.append(model_xgb)

            model_lgbm = lgb.Booster(
                model_file=str(
                    input_dir
                    / f"lgbm/seed=42/target={target_name}_fold={fold}.txt"
                )
            )
            models_lgbm.append(model_lgbm)

        pred_xgb = predict_xgb(features, models_xgb)
        pred_lgbm = predict_lgbm(features, models_lgbm)

        pred = (pred_xgb + pred_lgbm) / 2

        sample_submission[target_name] = pred

    print(sample_submission.head())


if __name__ == "__main__":
    with timer("main.py"):
        main()
