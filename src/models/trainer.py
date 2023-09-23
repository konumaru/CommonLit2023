import logging
import os
import pathlib
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class PytorchTrainer:
    def __init__(
        self,
        work_dir: Union[str, pathlib.Path],
        model: nn.Module,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        criterion: nn.Module,
        eval_metric: nn.Module,
        optimizer: optim.Optimizer,
        max_epochs: int,
        every_eval_steps: int = 100,
        save_interval="epoch",
        device="cuda",
    ) -> None:
        assert save_interval in [
            "epoch",
            "batch",
        ], "save_interval must be epoch or batch"

        self.work_dir = pathlib.Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.criterion = criterion
        self.eval_metric = eval_metric
        self.optimizer = optimizer

        self.max_epochs = max_epochs
        self.save_interval = save_interval

        self.best_eval_score = float("inf")
        self.every_eval_steps = every_eval_steps
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )

        self.best_model_path = self.work_dir / "best_model.pth"

        self.logger = self._get_logger()

    def _get_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.WARNING)

        file_handler = logging.FileHandler(
            self.work_dir / "train.log",
        )
        os.remove(file_handler.baseFilename)
        format = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        file_handler.setFormatter(format)
        logger.addHandler(file_handler)
        return logger

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.to(self.device)
        self.model.eval()
        preds = []
        targets = []
        for inputs, _targets in self.valid_dataloader:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(inputs)

            preds.append(outputs.detach().cpu())
            targets.append(_targets.detach().cpu())

        preds_all = torch.cat(preds, dim=0)
        targets_all = torch.cat(targets, dim=0)
        score = self.eval_metric(preds_all, targets_all)
        return score.item()

    def train_batch(self, batch):
        inputs, targets = batch
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        targets = targets.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self) -> None:
        self.model.to(self.device)
        self.model.train()

        total_steps = len(self.train_dataloader) * self.max_epochs
        eval_score = None
        train_global_step = 0

        progress_bar = tqdm(
            total=total_steps, desc="Training", dynamic_ncols=True
        )

        for epoch in range(self.max_epochs):
            running_loss = 0.0

            for batch_idx, (inputs, targets) in enumerate(
                self.train_dataloader
            ):
                loss = self.train_batch((inputs, targets))
                running_loss += loss
                avg_loss = running_loss / (batch_idx + 1)

                if (
                    self.save_interval == "batch"
                    and train_global_step % self.every_eval_steps == 0
                ):
                    eval_score = self.evaluate()
                    if eval_score < self.best_eval_score:
                        self.best_eval_score = eval_score
                        self.save_model(self.best_model_path)

                progress_bar.set_postfix(
                    {
                        "Epoch": epoch,
                        "loss": avg_loss,
                        "eval_score": eval_score,
                        "best_score": self.best_eval_score,
                    }
                )

                message = f"Epoch {epoch} | loss={avg_loss:.4f}"
                message += f" | eval_score={eval_score:.4f}"
                message += f" | best_score={self.best_eval_score:.4f}"
                self.logger.info(message)
                progress_bar.update(1)

            if self.save_interval == "epoch":
                eval_score = self.evaluate()
                if eval_score < self.best_eval_score:
                    self.best_eval_score = eval_score
                    self.save_model(self.best_model_path)

    def predict(self, dataloader: DataLoader) -> np.ndarray:
        self.model.to(self.device)
        self.model.eval()

        preds = []
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(inputs)
                preds.append(outputs.detach().cpu().numpy())
        return np.concatenate(preds, axis=0)

    def save_model(self, filename: Union[str, pathlib.Path]) -> None:
        torch.save(self.model.state_dict(), filename)

    def load_best_model(self):
        self.model.load_state_dict(torch.load(self.best_model_path))
        print(f"Loaded best model from {self.best_model_path}")
