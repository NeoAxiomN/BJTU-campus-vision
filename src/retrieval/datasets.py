from data.manifests import base_manifest_path, query_manifest_path
from utils import load_jsonl

from PIL import Image
from torch.utils.data import Dataset


class RetrievalPairDataset(Dataset):
    def __init__(
    	self,
    	rows: list[dict],
    	transform,
    ):
        self.rows = rows
        self.transform = transform

    def __len__(
    	self,
    ) -> int:
        return len(self.rows)

    def __getitem__(
    	self,
    	index: int,
    ):
        row = self.rows[index]
        image = Image.open(row["image_path"]).convert("RGB")
        return self.transform(image), self.transform(image)


class RetrievalEncodeDataset(Dataset):
    def __init__(
    	self,
    	rows: list[dict],
    	transform,
    ):
        self.rows = rows
        self.transform = transform

    def __len__(
    	self,
    ) -> int:
        return len(self.rows)

    def __getitem__(
    	self,
    	index: int,
    ):
        row = self.rows[index]
        image = Image.open(row["image_path"]).convert("RGB")
        tensor = self.transform(image)
        return tensor, row["id"], row["label"], row["image_path"]


def load_retrieval_rows(
	config,
) -> tuple[list[dict], list[dict]]:
    base_rows = load_jsonl(base_manifest_path(config))
    query_rows = load_jsonl(query_manifest_path(config))
    return base_rows, query_rows


def require_retrieval_manifests(
	config,
) -> tuple[list[dict], list[dict]]:
    try:
        return load_retrieval_rows(config)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Retrieval manifests are missing. Run `prepare-data --profile "
            f"{config.profile}` first."
        ) from exc
