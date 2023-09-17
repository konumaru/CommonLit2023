import pathlib
import re
from typing import List

import hydra
import nltk
import numpy as np
import pandas as pd
import torch
from nltk.corpus import stopwords
from omegaconf import DictConfig, OmegaConf
from rich.progress import track
from spellchecker import SpellChecker
from torch.utils.data import DataLoader

from models import CommonLitDataset, EmbeddingEncoder
from utils import timer
from utils.feature import BaseFeature, feature
from utils.io import load_pickle

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


def create_target_and_fold(data: pd.DataFrame) -> None:
    _data = data.copy()
    funcs = [fold, content, wording]
    for func in funcs:
        func(_data)


# -- For Feature Engineering --


def quotes_counter(row: pd.Series):
    summary = row["text"]
    text = row["prompt_text"]

    quotes_from_summary = re.findall(r'"([^"]*)"', summary)
    if len(quotes_from_summary) > 0:
        return [quote in text for quote in quotes_from_summary].count(True)
    else:
        return 0


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


def pos_tag_counter(row: pd.Series, pos: str) -> int:
    words = row["text"].split(" ")
    words = [word for word in words if len(word) > 0]
    tags = nltk.pos_tag(words)
    return len([tag for word, tag in tags if tag == pos])


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


def round_to_5(n):
    return round(n / 5) * 5


class CommonLitFeature(BaseFeature):
    def __init__(
        self,
        data: pd.DataFrame,
        use_cache: bool = True,
        is_test: bool = False,
        feature_dir: str | None = None,
        preprocess_dir: str | None = None,
    ) -> None:
        super().__init__(data, use_cache, is_test, feature_dir)
        self.preprocess_dir = pathlib.Path(preprocess_dir)

    @BaseFeature.cache()
    def text_length(self) -> np.ndarray:
        results = self.data["text"].str.len().to_numpy().reshape(-1, 1)
        return results

    @BaseFeature.cache()
    def word_count(self) -> np.ndarray:
        results = self.data["text"].str.split().str.len()
        return results.to_numpy().reshape(-1, 1)

    @BaseFeature.cache()
    def sentence_count(self) -> np.ndarray:
        results = self.data["text"].str.split(".").str.len()
        return results.to_numpy().reshape(-1, 1)

    @BaseFeature.cache()
    def quoted_sentence_count(self) -> np.ndarray:
        results = self.data["text"].apply(
            lambda x: len(re.findall(r'"(.*?)"', str(x)))
        )
        return results.to_numpy().reshape(-1, 1)

    @BaseFeature.cache()
    def consecutive_dots_count(self) -> np.ndarray:
        results = self.data["text"].apply(
            lambda x: len(re.findall(r"\.{3,4}", str(x)))
        )
        return results.to_numpy().reshape(-1, 1)

    @BaseFeature.cache()
    def quotes_count(self) -> np.ndarray:
        results = self.data.apply(quotes_counter, axis=1)
        return results.to_numpy().reshape(-1, 1)

    @BaseFeature.cache()
    def word_overlap_count(self) -> np.ndarray:
        results = self.data.apply(word_overlap_counter, axis=1)
        return results.to_numpy().reshape(-1, 1)

    @BaseFeature.cache()
    def ngram_co_occurrence_count(self) -> np.ndarray:
        bi_gram = self.data.apply(
            ngram_co_occurrence_counter, axis=1, args=(2,)
        )
        tr_gram = self.data.apply(
            ngram_co_occurrence_counter, axis=1, args=(3,)
        )

        results = pd.concat([bi_gram, tr_gram], axis=1)
        return results.to_numpy()

    @BaseFeature.cache()
    def spell_miss_count(self) -> np.ndarray:
        def counter(row: pd.Series) -> int:
            words = row["text"].split(" ")
            return len(SpellChecker().unknown(words))

        results = self.data.apply(counter, axis=1).to_numpy()
        return results.reshape(-1, 1)

    # @BaseFeature.cache()
    # def pos_tag_count(self) -> np.ndarray:
    #     # TODO: pos_tagごとに形態素解析させずに処理したら早くなる
    #     pos_tags = [
    #         "NN",
    #         "NNP",
    #         "NNS",
    #         "PRP",
    #         "VB",
    #         "VBD",
    #         "VBG",
    #         "VBN",
    #         "VBP",
    #         "VBZ",
    #         "JJ",
    #         "JJR",
    #         "JJS",
    #         "RB",
    #         "UH",
    #         "CD",
    #     ]
    #     pos_tags_cnt = [
    #         self.data.apply(pos_tag_counter, axis=1, args=(pos,))
    #         for pos in pos_tags
    #     ]
    #     results = pd.concat(pos_tags_cnt, axis=1).to_numpy()
    #     return results

    # @BaseFeature.cache()
    # def deberta_text_embeddings(self) -> np.ndarray:
    #     model_name = "microsoft/deberta-v3-base"
    #     embeddings = encode_embedding(model_name, self.data["text"].tolist())
    #     return embeddings

    # @BaseFeature.cache()
    # def deberta_prompt_embeddings(self) -> np.ndarray:
    #     model_name = "microsoft/deberta-v3-base"
    #     embeddings = encode_embedding(
    #         model_name, self.data["prompt_text"].tolist()
    #     )
    #     return embeddings

    @BaseFeature.cache()
    def target_encoded_word_count(self) -> np.ndarray:
        _data = self.data.copy()
        word_cnt = self.word_count().ravel()
        _data["clipped_word_count"] = (
            pd.Series(word_cnt).clip(25, 200).apply(round_to_5)
        )

        if self.is_test:
            encoding_map = pd.read_csv(
                self.preprocess_dir / "target_encoded_word_count.csv"
            )
            _data = _data.merge(
                encoding_map, on="clipped_word_count", how="left"
            )
            results = _data[["content", "wording"]].to_numpy()
            return results
        else:
            results = (
                _data.groupby(["fold", "clipped_word_count"])[
                    ["content", "wording"]
                ]
                .transform("mean")
                .to_numpy()
            )
            return results

    # @BaseFeature.cache()
    # def target_encoded_sentence_count(self) -> np.ndarray:
    #     _data = self.data.copy()
    #     f = load_pickle("data/feature/sentence_count.pkl")
    #     _data["sentence_count"] = pd.Series(f.ravel()).clip(None, 20)
    #     results = (
    #         _data.groupby(["fold", "sentence_count"])[["content", "wording"]]
    #         .transform("mean")
    #         .to_numpy()
    #     )
    #     encoding_map = _data.groupby(["sentence_count"])[
    #         ["content", "wording"]
    #     ].mean()

    #     encoding_map.to_csv(
    #         "data/preprocessed/target_encoded_sentence_count.csv"
    #     )
    #     return results


def create_target_encoding_map(cfg: DictConfig, data: pd.DataFrame) -> None:
    feature_dir = pathlib.Path(cfg.path.feature)
    output_dir = pathlib.Path(cfg.path.preprocessed)

    word_cnt = pd.Series(load_pickle(feature_dir / "word_count.pkl").ravel())
    data["clipped_word_count"] = word_cnt.clip(25, 200).apply(round_to_5)
    encoding_map = data.groupby(["clipped_word_count"])[
        ["content", "wording"]
    ].mean()
    encoding_map.to_csv(output_dir / "target_encoded_word_count.csv")


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    input_dir = pathlib.Path(cfg.path.preprocessed)

    train = pd.read_csv(input_dir / "train.csv")
    print(train.shape)

    create_target_and_fold(train)

    features = CommonLitFeature(
        train,
        feature_dir=cfg.path.feature,
        preprocess_dir=cfg.path.preprocessed,
    )
    results = features.create_features()
    print(pd.DataFrame(results).info())

    create_target_encoding_map(cfg, train)


if __name__ == "__main__":
    with timer("Create feature"):
        main()
