import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


MAX_SCENARIO = 36


class WidarDataset(Dataset):
    def __init__(self, data, labels, crop_size=None):
        self.data = data
        self.labels = labels
        self.crop_size = crop_size
        self.mean = 14.1664
        self.std = 4.5278

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)

        x = (x - self.mean) / self.std

        if self.crop_size and x.shape[1] > self.crop_size:
            start = (x.shape[1] - self.crop_size) // 2
            x = x[:, start:start + self.crop_size]

        return x, y


# ===============================
# 读取单个场景
# ===============================
def load_single_scenario(root_path, scenario_id):

    csi_path = os.path.join(root_path, f"all_csi_{scenario_id}.mat")
    action_path = os.path.join(root_path, f"action_{scenario_id}.mat")

    data_list = []
    label_list = []

    with h5py.File(csi_path, "r") as f:
        grp = f["csi_result"]

        # keys 是字符串数字，需要排序
        keys = sorted(grp.keys(), key=lambda x: int(x))

        for k in keys:
            data = grp[k][:]
            data_list.append(data)

    with h5py.File(action_path, "r") as f:
        labels = f["action"][:]

    labels = labels.astype(np.int64)
    import pdb; pdb.set_trace()
    return np.array(data_list), labels


# ===============================
# 读取全部场景
# ===============================
def load_all_scenarios(root_path):
    all_data = []
    all_labels = []

    for sid in range(1, MAX_SCENARIO + 1):
        print(f"Loading scenario {sid}")
        data, labels = load_single_scenario(root_path, sid)
        all_data.append(data)
        all_labels.append(labels)

    return np.concatenate(all_data), np.concatenate(all_labels)


# ===============================
# 主接口
# ===============================
def get_widar_dataloader(root_path,
                         scenario_id=0,
                         batch_size=32,
                         crop_size=1000,
                         shuffle=True):

    if scenario_id == 0:
        data, labels = load_all_scenarios(root_path)
    else:
        data, labels = load_single_scenario(root_path, scenario_id)
    import pdb; pdb.set_trace()
    dataset = WidarDataset(data, labels, crop_size)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    root = "/home/chenjiayi/workspace/willm/wifi_data/widar3_raw/processed"

    train_loader, test_loader = get_widar_dataloader(
        root,
        scenario_id=1,   # 0 = all
        batch_size=16,
        crop_size=1000
    )

    for x, y in train_loader:
        print(x.shape, y.shape)
        break