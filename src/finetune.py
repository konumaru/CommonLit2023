import os
import pathlib
from typing import Dict

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from metric import mcrmse
from utils import timer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CommonLitDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        target: str,
        model_name: str,
        max_len: int = 256,
        is_train: bool = True,
    ) -> None:
        self.input_texts = data["text"].tolist()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len

        if is_train:
            self.targets = torch.from_numpy(data[target].to_numpy())
        else:
            self.targets = torch.zeros((len(data), 1))

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


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    rmse = mean_squared_error(labels, predictions, squared=False)
    return {"rmse": rmse}


def predict(model: nn.Module, dataset: Dataset) -> np.ndarray:
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    preds = []

    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            z = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            preds.append(z.logits.detach().cpu().numpy())

    preds_all = np.concatenate(preds, axis=0)
    return preds_all


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    input_dir = pathlib.Path(cfg.path.preprocessed)

    data = pd.read_csv(input_dir / "train.csv")

    # --- train ---
    # for target in ["content", "wording"]:
    #     for fold in range(cfg.n_splits):
    #         print(f"Fold: {fold}")
    #         train_dataset = CommonLitDataset(
    #             data.query(f"fold!={fold}"),
    #             target,
    #             "microsoft/deberta-v3-base",
    #         )
    #         valid_dataset = CommonLitDataset(
    #             data.query(f"fold=={fold}"),
    #             target,
    #             "microsoft/deberta-v3-base",
    #         )

    #         model = AutoModelForSequenceClassification.from_pretrained(
    #             "microsoft/deberta-v3-base", num_labels=1
    #         )

    #         output_dir = (
    #             f"./data/train/finetuned-deberta-v3-base-{target}-fold{fold}"
    #         )
    #         training_args = TrainingArguments(
    #             output_dir=output_dir,
    #             overwrite_output_dir=True,
    #             load_best_model_at_end=True,
    #             report_to="none",  # type: ignore
    #             greater_is_better=False,
    #             num_train_epochs=1,
    #             per_device_train_batch_size=16,
    #             per_device_eval_batch_size=16,
    #             learning_rate=1.5e-5,
    #             weight_decay=0.02,
    #             seed=cfg.seed,
    #             metric_for_best_model="rmse",
    #             save_strategy="steps",
    #             evaluation_strategy="steps",
    #             eval_steps=1,
    #             save_steps=1,
    #             save_total_limit=1,
    #         )

    #         trainer = Trainer(
    #             model=model,
    #             args=training_args,
    #             train_dataset=train_dataset,
    #             eval_dataset=valid_dataset,
    #             compute_metrics=compute_metrics,  # type: ignore
    #         )

    #         trainer.train()

    #         save_model_dir = (
    #             f"./data/model/finetuned-deberta-v3-base-{target}-fold{fold}"
    #         )
    #         model.save_pretrained(save_model_dir)

    # --- predict ---
    data[["pred_content", "pred_wording"]] = 0.0
    for fold in range(cfg.n_splits):
        for target in ["content", "wording"]:
            valid_dataset = CommonLitDataset(
                data.query(f"fold=={fold}"),
                target,
                "microsoft/deberta-v3-base",
            )
            # save_model_dir = (
            #     f"./data/model/finetuned-deberta-v3-base-{target}-fold{fold}"
            # )

            external_dir = pathlib.Path(
                f"data/external/finetune-debertav3-fold-{fold}"
            )
            save_model_dir = (
                external_dir / f"finetuned-deberta-v3-base-{target}-fold{fold}"
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                save_model_dir, num_labels=1
            )
            pred = predict(model, valid_dataset)
            data.loc[data["fold"] == fold, f"pred_{target}"] = pred

    score = mcrmse(
        data[["content", "wording"]].to_numpy(),
        data[["pred_content", "pred_wording"]].to_numpy(),
    )

    output_dir = pathlib.Path(cfg.path.train)
    data[["pred_content", "pred_wording"]].to_csv(
        output_dir / "first_stage_pred.csv", index=False
    )
    print(score)


if __name__ == "__main__":
    with timer("main.py"):
        main()
