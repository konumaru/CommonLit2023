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
        model_name: str,
        max_len: int = 256,
    ) -> None:
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[index]
        encoded_token = self.tokenizer(
            row["text"],
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

        return outputs


class EmbeddingEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        pooling: str = "mean",
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super(EmbeddingEncoder, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        self.device = device

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = {k: v.to(self.device) for k, v in x.items()}

        outputs = self.model(
            input_ids=x["input_ids"], attention_mask=x["attention_mask"]
        )

        if self.pooling == "mean":
            mask = x["attention_mask"].clone().detach().unsqueeze(-1)
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

    model_name = "bert-base-uncased"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = CommonLitDataset(train, model_name, 256)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    model = EmbeddingEncoder(model_name, device=device)
    model.to(device)

    with torch.no_grad():
        for batch in dataloader:
            z = model(batch)
            print(z)
            print(z.shape)
            break


if __name__ == "__main__":
    main()
