import time
import random
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os

# =========================
# 1) HDF5 Lazy-Loading Dataset (Multi-worker Safe)
# =========================
class WidarHDF5Dataset(Dataset):
    """
    Lazy-load Widar CSI + labels from a single HDF5 file.
    - Does NOT load the whole dataset into RAM.
    - Each DataLoader worker will open its own h5 file handle on first access.
    """

    def __init__(self, hdf5_path: str, crop_size: int | None = None):
        self.hdf5_path = hdf5_path
        self.crop_size = crop_size

        # normalization constants (your original)
        self.mean = 14.1664
        self.std = 4.5278

        # Only read length/shape metadata
        with h5py.File(self.hdf5_path, "r") as f:
            self.length = f["csi"].shape[0]

        # Important: h5py.File handle should NOT be shared across processes
        self._file = None
        self._csi = None
        self._labels = None

    def __len__(self):
        return self.length

    def _ensure_open(self):
        """Open HDF5 file lazily (per worker process)."""
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r")
            self._csi = self._file["csi"]
            self._labels = self._file["labels"]

    def __getitem__(self, idx: int):
        self._ensure_open()

        # Read a single sample (numpy arrays)
        data_np = self._csi[idx]
        label_np = self._labels[idx]

        # Convert to torch (fast path)
        data = torch.from_numpy(data_np).float()
        data = (data - self.mean) / self.std

        # Optional crop
        if self.crop_size is not None and data.shape[1] > self.crop_size:
            indices = torch.linspace(0, data.shape[1] - 1, steps=self.crop_size).long()
            data = data[:, indices]

        label = torch.as_tensor(label_np).long()
        return data, label


# =========================
# 2) Your original MAT loader path (kept intact)
# =========================
class Widar3_X_Dataset(Dataset):
    def __init__(self, data_list, label_list, crop_size=None):
        self.data_list = data_list
        self.label_list = label_list
        self.mean = 14.1664
        self.std = 4.5278
        self.crop_size = crop_size

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = torch.tensor(self.data_list[idx], dtype=torch.float32)
        data = (data - self.mean) / self.std
        label = self.label_list[idx]

        if self.crop_size is not None and data.shape[1] > self.crop_size:
            indices = torch.linspace(0, data.shape[1] - 1, steps=self.crop_size).long()
            data = data[:, indices]
        return data, label


def widar3_x_get_raw_data(root_path, idx):
    st = time.time()
    labels = h5py.File(os.path.join(root_path, f"action_{idx}.mat"), "r")
    labels = labels["action"][()]

    cropped_csi = []
    cropped_labels = []
    threshold = 1000

    with h5py.File(os.path.join(root_path, f"all_csi_{idx}.mat"), "r") as file:
        csi_result_ref = file["csi_result"][:]
        for i, ref in enumerate(csi_result_ref):
            ref_obj = file[ref[0]]
            if isinstance(ref_obj, h5py.Dataset):
                ref_obj = np.array(ref_obj)
            else:
                print(f"第 {i} 个解引用对象的类型不是 Group 或 Dataset，而是: {type(ref_obj)}")

            if ref_obj.shape[1] >= threshold:
                cropped_csi.append(ref_obj[:, 0:threshold])
                cropped_labels.append(labels[i])

    cropped_labels_np = np.array(cropped_labels)
    labels_tensor = torch.from_numpy(cropped_labels_np).long().squeeze(dim=1)

    one_hot_labels = torch.zeros(len(cropped_labels), 22, dtype=torch.float)
    one_hot_labels[torch.arange(len(labels_tensor)), labels_tensor - 1] = 1

    ed = time.time()
    print("Load Widar3x:", ed - st)
    return cropped_csi, one_hot_labels


def widar3_x_dataset(root_path, idx, crop_size=None):
    cropped_csi, label_list = widar3_x_get_raw_data(root_path, idx)
    dataset = Widar3_X_Dataset(cropped_csi, label_list, crop_size=crop_size)
    return dataset


def widar3_x_dataloader(root, idx=1, batch_size=32, shuffle=True, crop_size=None, num_workers=1):
    start_time = time.time()
    dataset = widar3_x_dataset(root, idx, crop_size=crop_size)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )

    end_time = time.time()
    print("Total time in loading Widar (mat path):", end_time - start_time)
    return train_loader, test_loader


# =========================
# 3) HDF5 all-in-one dataloader (the one you need)
# =========================
def widar3_all_dataloader(
    hdf5_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    crop_size: int | None = None,
    num_workers: int = 4,
    seed: int = 42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    start_time = time.time()

    dataset = WidarHDF5Dataset(hdf5_path=hdf5_path, crop_size=crop_size)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    g = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=g)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    print("Total time in building loaders (h5 lazy):", time.time() - start_time)
    return train_loader, test_loader


# =========================
# 4) Unit Tests
# =========================
def unit_test_single_hdf5_sample(hdf5_path: str, crop_size: int | None = None):
    """
    Single-sample unit test for the HDF5 lazy dataset:
    - Checks __len__
    - Fetches one item (idx=0 and random idx)
    - Verifies dtype/shape and finite values
    """
    print("\n[UnitTest] Single sample test (HDF5 lazy dataset)")
    ds = WidarHDF5Dataset(hdf5_path=hdf5_path, crop_size=crop_size)

    assert len(ds) > 0, "Dataset length is 0. Check your HDF5 file content."
    print(f"  - Dataset length: {len(ds)} OK")

    # Test idx=0
    x0, y0 = ds[0]
    assert isinstance(x0, torch.Tensor), "Data is not a torch.Tensor"
    assert isinstance(y0, torch.Tensor), "Label is not a torch.Tensor"
    assert x0.dtype == torch.float32, f"Expected float32, got {x0.dtype}"
    assert torch.isfinite(x0).all(), "Data contains NaN/Inf"
    print(f"  - idx=0 data shape: {tuple(x0.shape)}, label shape: {tuple(y0.shape)} OK")

    # Test random idx
    ridx = random.randint(0, len(ds) - 1)
    xr, yr = ds[ridx]
    assert xr.shape == x0.shape, "Random sample shape differs from first sample (unexpected if fixed shape dataset)"
    assert torch.isfinite(xr).all(), "Random sample has NaN/Inf"
    print(f"  - random idx={ridx} data shape: {tuple(xr.shape)} OK")

    print("[UnitTest] PASS ✅")


def unit_test_single_mat_dataset(root_path: str, idx: int, crop_size: int | None = None):
    """
    Single-dataset unit test for your MAT-based loader (widar3_x_get_raw_data):
    - Loads one dataset id
    - Checks length > 0
    - Fetches one sample
    """
    print("\n[UnitTest] Single dataset test (MAT path)")
    ds = widar3_x_dataset(root_path=root_path, idx=idx, crop_size=crop_size)

    assert len(ds) > 0, "Loaded MAT dataset length is 0. Check threshold/filtering and source files."
    print(f"  - Dataset length: {len(ds)} OK")

    x0, y0 = ds[0]
    assert isinstance(x0, torch.Tensor), "Data is not a torch.Tensor"
    assert isinstance(y0, torch.Tensor), "Label is not a torch.Tensor"
    assert x0.dtype == torch.float32, f"Expected float32, got {x0.dtype}"
    assert torch.isfinite(x0).all(), "Data contains NaN/Inf"
    print(f"  - idx=0 data shape: {tuple(x0.shape)}, label shape: {tuple(y0.shape)} OK")

    print("[UnitTest] PASS ✅")


# =========================
# 5) Quick run
# =========================
if __name__ == "__main__":
    hdf5_path = "/home/chenjiayi/workspace/willm/wifi_data/widar_washed_denoised/widar_dataset.h5"
    root_path = "/home/chenjiayi/workspace/willm/wifi_data/widar_washed_denoised/"
    mat_idx = 1

    # --- Unit tests ---
    unit_test_single_hdf5_sample(hdf5_path=hdf5_path, crop_size=None)
    # If you still use MAT path:
    # unit_test_single_mat_dataset(root_path=root_path, idx=mat_idx, crop_size=None)

    # --- DataLoader test ---
    print("\nStart loading Widar3 dataset (lazy HDF5)...")
    train_loader, test_loader = widar3_all_dataloader(
        hdf5_path=hdf5_path,
        batch_size=32,
        shuffle=False,
        crop_size=None,
        num_workers=1,
        seed=42,
    )

    print("Dataset loaded successfully!")

    data, label = next(iter(train_loader))
    print("Train batch:")
    print("Data shape:", data.shape)
    print("Label shape:", label.shape)

    data, label = next(iter(test_loader))
    print("Test batch:")
    print("Data shape:", data.shape)
    print("Label shape:", label.shape)

    print("Load test finished.")