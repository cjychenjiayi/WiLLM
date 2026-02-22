import os
import random
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


def preprocess_labels(labels: torch.Tensor) -> Tuple[torch.Tensor, int]:
    unique_labels = torch.unique(labels)
    label_mapping = {label.item(): idx for idx, label in enumerate(unique_labels)}
    mapped_labels = torch.tensor([label_mapping[label.item()] for label in labels], dtype=torch.long)
    num_classes = len(unique_labels)
    one_hot_labels = F.one_hot(mapped_labels, num_classes=num_classes).float()
    return one_hot_labels, num_classes


def load_baha_processed(data_path: str) -> Tuple[List[Dict], torch.Tensor, torch.Tensor]:
    if not os.path.exists(os.path.join(data_path, "data_list.pth")):
        raise FileNotFoundError(f"Data absent at {data_path}. Please run Baha_preprocess.py.")

    data_list = torch.load(os.path.join(data_path, "data_list.pth"))
    mean = torch.load(os.path.join(data_path, "mean.pth"))
    variance = torch.load(os.path.join(data_path, "variance.pth"))
    return data_list, mean, variance


class BahaDataset(Dataset):
    def __init__(
        self,
        data_list: List[Dict],
        mean: Optional[torch.Tensor] = None,
        variance: Optional[torch.Tensor] = None,
        use_normalize: bool = True,
        crop_size: Optional[int] = None,
    ):
        self.data_list = data_list
        self.mean = mean
        self.variance = variance
        self.use_normalize = use_normalize
        self.crop_size = crop_size

        labels = torch.tensor([item["class"] for item in self.data_list], dtype=torch.long)
        self.labels, self.num_classes = preprocess_labels(labels)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data_list[idx]
        x = item["data"]  # (90, T) torch.Tensor
        y = self.labels[idx]

        if self.use_normalize and self.mean is not None and self.variance is not None:
            x = (x - self.mean) / torch.sqrt(self.variance + 1e-8)

        if self.crop_size is not None:
            T = x.shape[1]
            if T > self.crop_size:
                indices = torch.linspace(0, T - 1, self.crop_size).long()
                x = x.index_select(1, indices)

        return x, y


def Baha_dataloader(
    data_path: str = "/home/chenjiayi/workspace/willm/wifi_data/Baha/baha_processed",
    batch_size: int = 32,
    crop_size: Optional[int] = None,
    seed: int = 42,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    data_list, mean, variance = load_baha_processed(data_path)
    dataset = BahaDataset(data_list, mean, variance, use_normalize=True, crop_size=crop_size)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    g = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=g)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = Baha_dataloader()
    for x, y in train_loader:
        print(x.shape, y.shape)
        break