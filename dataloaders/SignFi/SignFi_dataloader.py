import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import numpy as np
import os
import random
import time


class SignFi_Dataset(Dataset):
    def __init__(
        self,
        csi_abs,
        labels,
        mean=None,
        std=None,
        use_normalize=True,
        crop_size=None,
        random_crop=True,
        noise=False
    ):
        """
        csi_abs: (N, F, T)
        labels:  (N, num_classes) one-hot
        """

        self.csi_abs = csi_abs
        self.labels = labels
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.noise = noise

        if use_normalize and mean is not None and std is not None:
            self.csi_abs = (self.csi_abs - mean) / std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        x = self.csi_abs[idx]   # (F, T)
        y = self.labels[idx]

        # 可选加噪声
        if self.noise:
            x = x * (1 + torch.randn_like(x) * 0.05)

        # 时间维 crop (dim=1)
        if self.crop_size is not None:
            T = x.shape[1]

            if T > self.crop_size:

                if self.random_crop:
                    start = torch.randint(
                        0,
                        T - self.crop_size + 1,
                        (1,)
                    ).item()
                    x = x[:, start:start + self.crop_size]

                else:
                    indices = torch.linspace(
                        0,
                        T - 1,
                        self.crop_size
                    ).long()
                    x = x.index_select(1, indices)

            else:
                pass  # 不 padding

        return x, y


def preprocess_labels(label_tensor):
    unique_labels = torch.unique(label_tensor)
    label_mapping = {label.item(): idx for idx, label in enumerate(unique_labels)}
    mapped = torch.tensor(
        [label_mapping[label.item()] for label in label_tensor],
        dtype=torch.long
    )
    one_hot = F.one_hot(mapped, num_classes=len(unique_labels)).float()
    return one_hot, len(unique_labels)


def signfi_dataset(folder_path, use_normalize=True, crop_size=None):

    file_path = os.path.join(folder_path, "all_processed.npz")
    if not os.path.exists(file_path):
        file_path = os.path.join(folder_path, "all.npz")

    if not os.path.exists(file_path):
        raise FileNotFoundError("Run SignFi_preprocess.py first.")

    print(f"Loading data from {file_path}")
    start_time = time.time()

    data = np.load(file_path)

    if "csi_abs" in data:
        csi_abs = data["csi_abs"]
        label = data["label"]
    else:
        csi = data["csi"]
        label = data["label"]
        csi_abs = np.abs(csi).astype(np.float32)

    print(f"Loaded in {time.time() - start_time:.2f}s")

    # 转 tensor
    csi_abs_tensor = torch.from_numpy(csi_abs)  # (N, F, T)
    label_tensor = torch.from_numpy(label.astype(np.int64))

    # one-hot
    one_hot_labels, num_classes = preprocess_labels(label_tensor)

    # 归一化统计
    if use_normalize:
        mean = torch.mean(csi_abs_tensor)
        std = torch.std(csi_abs_tensor)
    else:
        mean = std = None

    dataset = SignFi_Dataset(
        csi_abs_tensor,
        one_hot_labels,
        mean=mean,
        std=std,
        use_normalize=use_normalize,
        crop_size=crop_size
    )

    return dataset, num_classes


def signfi_dataloader(folder_path, batch_size=32, use_normalize=True, crop_size=None):

    # 固定随机种子
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dataset, num_classes = signfi_dataset(
        folder_path,
        use_normalize=use_normalize,
        crop_size=crop_size
    )

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    g = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=g
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader, num_classes


if __name__ == "__main__":
    root_folder = "/home/chenjiayi/workspace/willm/wifi_data/SignFi"

    print("=== Test without crop ===")
    train_loader, test_loader, num_classes = signfi_dataloader(
        root_folder,
        crop_size=None
    )

    for x, y in train_loader:
        print("No crop:", x.shape, y.shape)
        break

    print("\n=== Test with crop=500 ===")
    train_loader, test_loader, num_classes = signfi_dataloader(
        root_folder,
        crop_size=500
    )

    for x, y in train_loader:
        print("Crop=500:", x.shape, y.shape)
        break