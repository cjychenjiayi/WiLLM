import os
import numpy as np
import torch
import torch.nn.functional as F
import time
from torch.utils.data import Dataset

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

class XRF55(Dataset):
    def __init__(self, data, labels, crop_size=None):
        import time
        st = time.time()
        print("Dataset Initialization START")
        self.num_classes = len(actions_dict)
        
        # Ensure labels are a tensor
        if isinstance(labels, list):
            labels = torch.tensor(labels, dtype=torch.long)
        elif isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).long()
        elif isinstance(labels, torch.Tensor):
            labels = labels.long()
            
        self.label_tensor = F.one_hot(labels - 1, num_classes=self.num_classes).float()
        ed = time.time()
        print(f"Label processing: {ed - st:.4f}s")
        
        st = ed
        if isinstance(data, np.ndarray):
            data_tensor = torch.from_numpy(data).float()
        else:
            data_tensor = data.float()

        ed = time.time()
        print(f"Data tensor conversion: {ed - st:.4f}s")
        
        st = ed
        # Original logic maintained
        data_tensor = data_tensor.transpose(-1, -2)
        ed = time.time()
        print(f"Transpose 1: {ed - st:.4f}s")
        
        st = ed
        self.mean = 8.7824
        self.std = 4.8665
        
        data_tensor = (data_tensor - self.mean) / self.std
        ed = time.time()
        print(f"Normalization: {ed - st:.4f}s")
        
        st = ed
        data_tensor = data_tensor.transpose(1, 2)
        ed = time.time()
        print(f"Transpose 2: {ed - st:.4f}s")
        
        self.data = data_tensor
        self.crop_size = crop_size

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.label_tensor[idx] # Use pre-computed label tensor
        if self.crop_size is not None:
            original_size = sample.size(1)
            # if original_size > self.crop_size: # Logic as per original code ?
            # Re-checking original logic: "if original_size > self.crop_size:" -> index_select
            # But what if smaller? The original code didn't handle it (maybe assumed larger).
            if original_size > self.crop_size:
                indices = torch.linspace(0, original_size - 1, steps=self.crop_size).long()
                sample = sample.index_select(dim=1, index=indices)
        return sample, label

def load_xrf55(load_path="/home/chenjiayi/workspace/willm/wifi_data/xrf55_all", crop_size=None):
    """
    Loads the preprocessed XRF55 dataset from .pth files.
    """
    csi_path = os.path.join(load_path, "csi.pth")
    action_path = os.path.join(load_path, "action.pth")
    
    if not os.path.exists(csi_path) or not os.path.exists(action_path):
        raise FileNotFoundError(f"Processed data not found at {load_path}. Please run preprocess_xrf55.py first.")
        
    print(f"Loading data from {csi_path}...")
    import time
    st = time.time()
    # Using weights_only=False because we are loading tensor data...
    csi = torch.load(csi_path, weights_only=False)
    action = torch.load(action_path, weights_only=False)
    print(f"Data loading time: {time.time() - st:.4f}s")
    
    print("Creating dataset instance...")
    dataset = XRF55(csi, action, crop_size=crop_size)
    return dataset

if __name__ == "__main__":
    try:
        dataset = load_xrf55()
        print(f"Dataset loaded. Size: {len(dataset)}")
    except Exception as e:
        print(f"Error: {e}")
