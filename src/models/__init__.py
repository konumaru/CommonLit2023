import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer


def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.use_deterministic_algorithms = True


class MCRMSELoss(nn.Module):
    def __init__(self) -> None:
        super(MCRMSELoss, self).__init__()

    def forward(
        self, y_true: torch.Tensor, y_pred: torch.Tensor
    ) -> torch.Tensor:
        colwise_mse = torch.mean(torch.square(y_true - y_pred), dim=0)
        return torch.mean(torch.sqrt(colwise_mse), dim=0)


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

        if is_train:
            self.targets = torch.from_numpy(data[targets].to_numpy())
        else:
            self.targets = torch.zeros((len(data), len(targets)))

    def __len__(self) -> int:
        return len(self.input_texts)

    def __getitem__(
        self, index: int
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        text = self.input_texts[index]
        encoded_token = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt",
        )

        inputs = {}
        inputs["input_ids"] = encoded_token[
            "input_ids"
        ].squeeze()  # type: ignore
        inputs["attention_mask"] = encoded_token[
            "attention_mask"
        ].squeeze()  # type: ignore
        targets = self.targets[index, :]

        return (inputs, targets)


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class CommonLitModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int = 1,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels

        self.config = AutoConfig.from_pretrained(model_name)
        self.config.update(
            {
                "hidden_dropout_prob": 0.0,
                "attention_probs_dropout_prob": 0.007,
            }
        )
        self.model = AutoModel.from_pretrained(model_name, config=self.config)

        self.pooler = MeanPooling()
        self.head = nn.Linear(self.config.hidden_size, self.num_labels)

        self.init_model()

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        output = self.pooler(output.last_hidden_state, attention_mask)
        output = self.head(output)
        return output

    def init_model(self) -> None:
        for i in range(0, 8):
            for _, param in self.model.encoder.layer[i].named_parameters():
                param.requires_grad = False

        for layer in self.model.encoder.layer[-2:]:
            for module in layer.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(
                        mean=0.0, std=self.model.config.initializer_range
                    )
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.Embedding):
                    module.weight.data.normal_(
                        mean=0.0, std=self.model.config.initializer_range
                    )
                    if module.padding_idx is not None:
                        module.weight.data[module.padding_idx].zero_()
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)


def main() -> None:
    data = pd.read_csv("./data/preprocessed/train.csv")

    model_name = "microsoft/deberta-v3-base"
    dataset = CommonLitDataset(data, model_name)
    model = CommonLitModel(model_name, num_labels=2)

    dataloader = DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=8
    )
    inputs, targets = next(iter(dataloader))
    z = model(inputs)
    print(z)

    loss_fn = MCRMSELoss()
    loss = loss_fn(z, targets)
    print(loss)


if __name__ == "__main__":
    main()
