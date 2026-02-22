import torch
import torch.nn.functional as F
import numpy as np
import glob
from torch.utils.data import Dataset, DataLoader
import os


class UTHARTensorDataset(Dataset):
    """
    x: (N, 250, 90)  (T, F)
    crop_size:
      - None: 不裁剪
      - int: 只在时间维T裁剪到crop_size（均匀采样）
    输出：
      - 展平为 (F*T,) 即 (crop_size*90,) 或 (250*90,)
    """
    def __init__(self, x: torch.Tensor, y: torch.Tensor, crop_size=None):
        self.x = x
        self.y = y
        self.crop_size = crop_size

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx]  # (T, F) -> (250, 90)
        y = self.y[idx]  # one-hot: (C,)

        # crop（只裁时间维 T）
        if self.crop_size is not None:
            T = x.shape[0]
            if T > self.crop_size:
                indices = torch.linspace(0, T - 1, self.crop_size).long()
                x = x.index_select(0, indices)
            else:
                # T <= crop_size：按你要求，不padding，不处理
                pass

        # 改成 f*t：直接展平
        x = x.reshape(-1)  # (T*F,)

        return x, y


def UT_HAR_dataset(root_dir):
    data_list = glob.glob(os.path.join(root_dir, 'UT_HAR/data/*.csv'))
    label_list = glob.glob(os.path.join(root_dir, 'UT_HAR/label/*.csv'))

    data_dict = {}
    label_dict = {}

    # 读数据
    for data_dir in data_list:
        data_name = data_dir.split('/')[-1].split('.')[0]
        with open(data_dir, 'rb') as f:
            data = np.load(f)
            # 按你原始逻辑：reshape(len(data), 250, 90)
            data = data.reshape(len(data), 250, 90)
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        data_dict[data_name] = torch.tensor(data_norm, dtype=torch.float32)

    # 读标签
    for label_dir in label_list:
        label_name = label_dir.split('/')[-1].split('.')[0]
        with open(label_dir, 'rb') as f:
            label = np.load(f)
        label_dict[label_name] = torch.tensor(label, dtype=torch.long)

    if not label_dict:
        return {}

    # 更稳的 num_classes：用所有标签的 max + 1
    all_labels = torch.cat([v.reshape(-1) for v in label_dict.values()], dim=0)
    num_classes = int(all_labels.max().item()) + 1

    # one-hot
    processed_labels = {}
    for key, labels in label_dict.items():
        processed_labels[key] = F.one_hot(labels.to(torch.long), num_classes=num_classes).float()

    WiFi_data = {**data_dict, **processed_labels}
    return WiFi_data


def get_UTHAR(root, batch_size, crop_size=None):
    data = UT_HAR_dataset(root)
    if not data:
        return None, None, "UT_HAR"

    # 你原始代码假设这些 key 存在
    x_train = data['X_train']  # (N, 250, 90)
    y_train = data['y_train']  # (N, C)

    x_test = torch.cat((data['X_val'], data['X_test']), dim=0)
    y_test = torch.cat((data['y_val'], data['y_test']), dim=0)


    train_set = UTHARTensorDataset(x_train, y_train, crop_size=crop_size)
    test_set = UTHARTensorDataset(x_test, y_test, crop_size=crop_size)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False
    )
    param = [250, 90, 7]
    return [train_loader, test_loader, "UT_HAR", param]


if __name__ == '__main__':
    root_dir = '/home/chenjiayi/workspace/willm/wifi_data'
    batch_size = 64

    try:
        print("Testing UT_HAR (crop_size=None)...")
        train_loader, test_loader, name = get_UTHAR(root_dir, batch_size, crop_size=None)
        print(f"Successfully loaded {name}")
        if train_loader:
            for x, y in train_loader:
                print(f"Train Batch - X shape: {x.shape}, Y shape: {y.shape}")  # x: (B, 250*90)
                break
    except Exception as e:
        print(f"Error loading UT_HAR: {e}")

    try:
        print("\nTesting UT_HAR (crop_size=100)...")
        train_loader, test_loader, name = get_UTHAR(root_dir, batch_size, crop_size=100)
        print(f"Successfully loaded {name}")
        if train_loader:
            for x, y in train_loader:
                print(f"Train Batch - X shape: {x.shape}, Y shape: {y.shape}")  # x: (B, 100*90)
                break
    except Exception as e:
        print(f"Error loading UT_HAR: {e}")