import os
import pathlib
import warnings

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from metric import mcrmse
from models import CommonLitDataset, CommonLitModel, MCRMSELoss
from models.trainer import PytorchTrainer
from utils import timer

warnings.simplefilter("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("high")


@torch.no_grad()
def predict(model: nn.Module, dataset: Dataset) -> np.ndarray:
    dataloader = DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=8
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()
    preds = []
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        z = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        preds.append(z.logits.detach().cpu().numpy())

    results = np.concatenate(preds, axis=0)
    return results


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    input_dir = pathlib.Path(cfg.path.preprocessed)
    model_dir = pathlib.Path(cfg.path.model) / "deberta-v3-base"

    data = pd.read_csv(input_dir / "train.csv")
    data[["pred_content", "pred_wording"]] = 0.0

    model_name = "microsoft/deberta-v3-base"
    # model_name = "microsoft/deberta-v3-large"

    for fold in range(cfg.n_splits):
        print(f"Fold: {fold}")
        model = CommonLitModel(model_name, num_labels=2)
        train_dataset = CommonLitDataset(
            data.query(f"fold!={fold}"), model_name, max_len=512
        )
        valid_dataset = CommonLitDataset(
            data.query(f"fold=={fold}"), model_name, max_len=512
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=8, shuffle=True, drop_last=True
        )
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=8, shuffle=False
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-4, weight_decay=0.02
        )

        trainer = PytorchTrainer(
            work_dir=str(model_dir / f"fold{fold}"),
            model=model,
            criterion=MCRMSELoss(),
            eval_metric=MCRMSELoss(),
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            optimizer=optimizer,
            save_interval="batch",
            max_epochs=4,
            every_eval_steps=50,
        )
        trainer.train()
        trainer.load_best_model()

        pred = trainer.predict(valid_dataloader)
        data.loc[data["fold"] == fold, ["pred_content", "pred_wording"]] = pred

    score = mcrmse(
        data[["content", "wording"]].to_numpy(),
        data[["pred_content", "pred_wording"]].to_numpy(),
    )
    print(f"Score: {score}")
    data[
        ["prompt_id", "student_id", "fold", "pred_content", "pred_wording"]
    ].to_csv(str(model_dir / "oof.csv"), index=False)


if __name__ == "__main__":
    with timer("main.py"):
        main()
