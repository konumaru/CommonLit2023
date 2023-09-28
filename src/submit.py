import pathlib
from typing import Any, Dict, List

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from xgboost import XGBRegressor

from feature import CommonLitFeature
from finetune import predict as predict_finetuned_model
from utils import timer


class CommonLitDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        model_name: str,
        targets: List[str] = ["content", "wording"],
        max_len: int = 512,
        is_train: bool = True,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.input_texts = (
            data["prompt_question"]
            + f" {self.tokenizer.sep_token} "
            + data["text"]
        ).tolist()
        self.max_len = max_len
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        if is_train:
            self.targets = torch.from_numpy(data[targets].to_numpy())
        else:
            self.targets = torch.zeros((len(data), len(targets)))

    def __len__(self) -> int:
        return len(self.input_texts)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        text = self.input_texts[index]
        encoded_token = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt",
        )

        outputs = {}
        outputs["input_ids"] = encoded_token[
            "input_ids"
        ].squeeze()  # type: ignore
        outputs["attention_mask"] = encoded_token[
            "attention_mask"
        ].squeeze()  # type: ignore
        outputs["labels"] = self.targets[index, :]

        return outputs


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
            data,
            "microsoft/deberta-v3-base",
            max_len=512,
            is_train=False,
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

    cl_feature = CommonLitFeature(
        test,
        sentence_encoder=SentenceTransformer(
            "all-MiniLM-L6-v2", device="cuda:0"
        ),
        is_test=True,
        preprocess_dir="./data/upload",
    )
    text_features = cl_feature.create_features()
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
