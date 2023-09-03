import pathlib
from typing import Dict

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


class CommonLitDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
    ) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict:
        row = self.data.iloc[index]

        output = {}
        output["text"] = row["text"]
        return output


class Embedding(nn.Module):
    def __init__(
        self,
        model_name: str,
        max_len: int,
        pooling: str = "mean",
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_len = max_len
        self.pooling = pooling
        self.device = device

    def forward(self, x: Dict) -> torch.Tensor:
        encoded_token = self.tokenizer(
            x["text"],
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt",
        )
        if self.device == "cuda":
            encoded_token = {k: v.to("cuda") for k, v in encoded_token.items()}

        outputs = self.model(**encoded_token)

        if self.pooling == "mean":
            mask = (
                torch.tensor(encoded_token["attention_mask"])
                .clone()
                .detach()
                .unsqueeze(-1)
            )
            embeddings = (outputs.last_hidden_state * mask).sum(
                dim=(1)
            ) / mask.sum(dim=1)

        elif self.pooling == "max":
            embeddings = outputs[0].max(dim=1)[0]
        else:
            raise NotImplementedError

        return embeddings


def main() -> None:
    input_dir = pathlib.Path("./data/raw")

    train = pd.read_csv(input_dir / "summaries_train.csv")
    print(train.head())

    dataset = CommonLitDataset(train)
    dataloader = DataLoader(dataset, batch_size=8, num_workers=4)

    model = Embedding("bert-base-uncased", 256)
    model.to("cuda")

    for batch in dataloader:
        z = model(batch)
        print(z.shape)
        break


if __name__ == "__main__":
    main()
