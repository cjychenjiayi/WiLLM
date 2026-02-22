import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

DEFAULT_DATA_DIR = "/home/chenjiayi/workspace/willm/wifi_data/xrf55_all"

actions_dict = {
    1: "Carrying Weight",
    2: "Mopping the Floor",
    3: "Cutting",
    4: "Wearing Hat",
    5: "Using a Phone",
    6: "Throw Something",
    7: "Put Something on the Table",
    8: "Put on Clothing",
    9: "Picking",
    10: "Drinking",
    11: "Smoking",
    12: "Eating",
    13: "Brushing Teeth",
    14: "Blow Dry Hair",
    15: "Brush Hair",
    16: "Shake Hands",
    17: "Hugging",
    18: "Hand Something to Someone",
    19: "Kick Someone",
    20: "Hit Someone with Something",
    21: "Choke Someone's Neck",
    22: "Push Someone",
    23: "Body Weight Squats",
    24: "Tai Chi",
    25: "Boxing",
    26: "Weightlifting",
    27: "Hula Hooping",
    28: "Jump Rope",
    29: "Jumping Jack",
    30: "High Leg Lift",
    31: "Waving",
    32: "Clap Hands",
    33: "Fall on the Floor",
    34: "Jumping",
    35: "Running",
    36: "Sitting Down",
    37: "Standing Up",
    38: "Turning",
    39: "Walking",
    40: "Stretch Oneself",
    41: "Pat on Shoulder",
    42: "Playing Erhu",
    43: "Playing Ukulele",
    44: "Playing Drum",
    45: "Stomping",
    46: "Shaking Head",
    47: "Nodding",
    48: "Draw Circles",
    49: "Draw a Cross",
    50: "Pushing",
    51: "Pulling",
    52: "Swipe Left",
    53: "Swipe Right",
    54: "Swipe Up",
    55: "Swipe Down"
}

class XRF55PackedDataset(Dataset):
    def __init__(self, data_source, labels, indices=None):
        self.class_dict = actions_dict
        self.num_classes = len(actions_dict)
        self.mean = 8.7824
        self.std = 4.8665

        # If data_source is a path string, open as memmap (Disk Mode)
        # If data_source is an array, use it directly (RAM Mode)
        if isinstance(data_source, str):
            if not os.path.exists(data_source):
                raise FileNotFoundError(f"Data not found: {data_source}")
            self.data = np.memmap(data_source, dtype='float32', mode='r', shape=(len(labels), 270, 1000))
        else:
            self.data = data_source
            
        self.labels = labels
        # Use provided indices or all
        self.indices = indices if indices is not None else np.arange(len(labels))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        # Fast read + copy
        sample = self.data[real_idx].copy() 
        label = self.labels[real_idx]
        
        # To Tensor
        sample_tensor = torch.from_numpy(sample).float()
        label_tensor = torch.tensor(label).long()
        
        # Normalize
        sample_tensor = (sample_tensor - self.mean) / self.std

        return sample_tensor, label_tensor

def load_packed_dataloader(data_dir=DEFAULT_DATA_DIR, batch_size=32, load_to_ram=True):
    data_path = os.path.join(data_dir, "xrf55_data.mmap")
    labels_path = os.path.join(data_dir, "xrf55_labels.npy")
    
    if not os.path.exists(data_path):
        raise RuntimeError(f"Packed file not found: {data_path}\nRun XRF55_packed_preprocess.py first.")

    # 1. Load Labels & Determine Size
    all_labels = np.load(labels_path)
    total_len = len(all_labels)
    
    # 2. Prepare Data Source (RAM or Disk Path)
    if load_to_ram:
        print("[XRF55] Loading entire dataset to RAM...")
        # Open mmap then copy to RAM array
        mmap_ref = np.memmap(data_path, dtype='float32', mode='r', shape=(total_len, 270, 1000))
        data_source = np.array(mmap_ref)
        print(f"[XRF55] Loaded {data_source.nbytes / 1e9:.2f} GB.")
    else:
        # Pass path string, Dataset will open memmap on demand
        data_source = data_path

    # 3. Create Splits (Deterministic)
    indices = np.arange(total_len)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    test_ratio = 0.2
    split_point = int(total_len * (1 - test_ratio))
    
    train_idx = indices[:split_point]
    test_idx = indices[split_point:]
    
    print(f"[XRF55] Train: {len(train_idx)}, Test: {len(test_idx)}")

    # 4. Instantiate Datasets with different indices
    train_ds = XRF55PackedDataset(data_source, all_labels, indices=train_idx)
    test_ds = XRF55PackedDataset(data_source, all_labels, indices=test_idx)
    
    # 5. Create Loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    return train_loader, val_loader

if __name__ == "__main__":
    from tqdm import tqdm
    import time
    
    st = time.time()
    
    print("--- Testing XRF55 Dataloader ---")
    tr, te = load_packed_dataloader()
    
    print(f"Finished in {time.time()-st:.2f}s")
    st = time.time()
    for x, y in tqdm(tr, desc="Iterating Train"):
        pass
    print(f"Finished in {time.time()-st:.2f}s")

