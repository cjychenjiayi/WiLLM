from dataloaders.UT_HAR.UT_HAR_dataloader import get_UTHAR
from dataloaders.NTU_Datasets.NTU_dataloader import get_NTUHAR, get_NTUHumanID
from dataloaders.AI_RAN_Datasets.AI_RAN_dataloader import (
    get_WiCount, get_WiFall_action, get_WiFall_people,
    get_WiGesture_action, get_WiGesture_people
)
from dataloaders.RFNet.RFNet_loader import load_rfnet
from dataloaders.Widar3.widar_dataloader import widar3_x_dataloader, widar3_all_dataloader
from dataloaders.XRF55_repo.XRF55_packed_dataloader import load_packed_dataloader
from dataloaders.SignFi.SignFi_dataloader import signfi_dataset
from dataloaders.Baha.Baha_dataloader import Baha_dataloader

from torch.utils.data import random_split, DataLoader
import os
import torch


# =========================
# Widar (idx=0 → all, else single)
# =========================
def get_Widar(root, batch_size, crop_size=None, idx=0):
    data_root = os.path.join(root, "widar_washed_denoised")

    if idx == 0:
        hdf5_path = os.path.join(data_root, "widar_dataset.h5")

        train_loader, test_loader = widar3_all_dataloader(
            hdf5_path=hdf5_path,
            batch_size=batch_size,
            shuffle=True,
            crop_size=crop_size,
            num_workers=1
        )

        name = "Widar3_all"
    else:
        train_loader, test_loader = widar3_x_dataloader(
            root=data_root,
            idx=idx,
            batch_size=batch_size,
            shuffle=True,
            crop_size=crop_size,
            num_workers=1
        )

        name = f"Widar3_single_{idx}"

    time_len = crop_size if crop_size is not None else 1000
    param = [52, time_len, 22]

    return [train_loader, test_loader, name, param]


# =========================
# RFNet
# =========================
def get_RFNet(root, batch_size, crop_size=100):
    train_dataset, val_dataset = load_rfnet(
        data_path=os.path.join(root, "RF-Net"),
        crop_size=crop_size
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    param = [60, crop_size if crop_size else 512, 6]
    return [train_loader, test_loader, "RFNet", param]


# =========================
# XRF55
# =========================
class XRF55CropWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset, crop_size=None):
        self.base_dataset = base_dataset
        self.crop_size = crop_size

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]

        if self.crop_size is not None:
            T = x.shape[1]
            if T > self.crop_size:
                indices = torch.linspace(0, T - 1, self.crop_size).long()
                x = x.index_select(1, indices)

        return x, y


def get_Xrf55(root, batch_size, crop_size=None):
    train_loader, test_loader = load_packed_dataloader(
        data_dir=os.path.join(root, "xrf55_all"),
        batch_size=batch_size,
        load_to_ram=True
    )

    train_dataset = XRF55CropWrapper(train_loader.dataset, crop_size)
    test_dataset = XRF55CropWrapper(test_loader.dataset, crop_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    param = [270, crop_size if crop_size else 1000, 55]
    return [train_loader, test_loader, "XRF55", param]


# =========================
# SignFi
# =========================
def get_signfi(root, batch_size, crop_size=100):
    folder_path = os.path.join(root, "SignFi")

    dataset, num_classes = signfi_dataset(
        folder_path,
        use_normalize=True,
        crop_size=crop_size
    )

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    x, _ = dataset[0]
    param = [x.shape[0], x.shape[1], num_classes]

    return [train_loader, test_loader, "SignFi", param]


# =========================
# Baha
# =========================
def get_Baha(root, batch_size, crop_size=None):
    data_path = os.path.join(root, "Baha", "baha_processed")

    train_loader, test_loader = Baha_dataloader(
        data_path=data_path,
        batch_size=batch_size,
        crop_size=crop_size
    )

    param = [90, 901, 15]
    return [train_loader, test_loader, "Baha", param]


# =========================
# Dataset Registry
# =========================
dataset_dict = {
    "UTHAR": get_UTHAR,
    "RFNet": get_RFNet,
    "NTUHumanID": get_NTUHumanID,
    "NTUHAR": get_NTUHAR,
    "SignFi": get_signfi,
    "Baha": get_Baha,
    "Xrf55": get_Xrf55,
    "Widar3": get_Widar,
    "WiCount": get_WiCount,
    "WiFallact": get_WiFall_action,
    "WiGestureact": get_WiGesture_action,
    "WiFallid": get_WiFall_people,
    "WiGestureid": get_WiGesture_people,
}


# =========================
# Unified Loader
# =========================
def load_data(dataset_name, root, batch_size=4, crop_size=None, idx=0):
    if dataset_name not in dataset_dict:
        print("Dataset not found:", dataset_name)
        exit(1)

    if dataset_name == "Widar3":
        train_set, test_set, name, param = dataset_dict[dataset_name](
            root, batch_size, crop_size, idx=idx
        )
    else:
        train_set, test_set, name, param = dataset_dict[dataset_name](
            root, batch_size, crop_size
        )

    return train_set, test_set, param


# =========================
# Main Test
# =========================
if __name__ == "__main__":
    root = "/home/chenjiayi/workspace/willm/wifi_data"
    batch_size = 1

    for dataset_name in dataset_dict.keys():
        print("\nTesting:", dataset_name)

        if dataset_name == "Widar3":
            train_set, test_set, param = load_data(
                dataset_name, root, batch_size, crop_size=100, idx=0
            )
        else:
            train_set, test_set, param = load_data(
                dataset_name, root, batch_size, crop_size=100
            )

        for x, y in train_set:
            print(dataset_name, x.shape, y.shape)
            break

        print("Train samples:", len(train_set) * batch_size)

        del train_set, test_set
        torch.cuda.empty_cache()
        import gc; gc.collect()
    
    for idx in range(1, 36):
        print("\nTesting Widar3 single idx:", idx)
        train_set, test_set, param = load_data(
            "Widar3", root, batch_size, crop_size=100, idx=idx
        )
        for x, y in train_set:
            print("Widar3_single", idx, x.shape, y.shape)
            break
        del train_set, test_set
        torch.cuda.empty_cache()
        import gc; gc.collect()