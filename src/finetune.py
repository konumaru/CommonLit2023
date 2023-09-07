import pathlib
from typing import Dict, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from utils import timer


class CommonLitDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        model_name: str,
        max_len: int = 256,
        is_train: bool = True,
    ) -> None:
        self.input_texts = data["text"].tolist()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len

        if is_train:
            self.targets = torch.from_numpy(
                data[["content", "wording"]].to_numpy()
            )
        else:
            self.targets = torch.zeros((len(data), 2))

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
        outputs["labels"] = self.targets[index]

        return outputs


def mcrmse(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    logits, labels = eval_pred

    squared_error = (logits - labels) ** 2
    mean_squared_error = squared_error.mean(axis=0)
    mcrmse = np.sqrt(mean_squared_error).mean()
    return {"mcrmse": mcrmse}


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    input_dir = pathlib.Path(cfg.path.preprocessed)

    data = pd.read_csv(input_dir / "train.csv")

    for fold in range(cfg.n_splits):
        print(f"Fold: {fold}")
        train_dataset = CommonLitDataset(
            data.query(f"fold!={fold}"), "microsoft/deberta-v3-base"
        )
        valid_dataset = CommonLitDataset(
            data.query(f"fold=={fold}"), "microsoft/deberta-v3-base"
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-base", num_labels=2
        )

        training_args = TrainingArguments(
            output_dir=f"./data/train/finetuned-deberta-v3-base-fold{fold}",
            overwrite_output_dir=True,
            load_best_model_at_end=True,
            report_to="none",  # type: ignore
            greater_is_better=False,
            num_train_epochs=5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            learning_rate=1.5e-5,
            weight_decay=0.02,
            seed=cfg.seed,
            metric_for_best_model="mcrmse",
            save_strategy="epoch",
            evaluation_strategy="epoch",
            # save_strategy="steps",
            # evaluation_strategy="steps",
            # eval_steps=50,
            # save_steps=100,
            save_total_limit=1,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            compute_metrics=mcrmse,  # type: ignore
        )

        trainer.train()
        model.save_pretrained(
            f"./data/model/finetuned-deberta-v3-base-fold{fold}"
        )


if __name__ == "__main__":
    with timer("main.py"):
        main()
