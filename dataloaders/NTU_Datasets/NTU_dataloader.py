import torch
import torch.nn.functional as F
import numpy as np
import glob
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import os

class NTU_Dataset(Dataset):
    def __init__(self, root_dir, modal='CSIamp', crop_size=None):
        self.root_dir = root_dir
        self.modal = modal
        self.data_list = glob.glob(os.path.join(root_dir, '*/*.mat'))
        self.folder = glob.glob(os.path.join(root_dir, '*/'))
        self.category = {self.folder[i].split('/')[-2]: i for i in range(len(self.folder))}
        self.num_classes = len(self.category)
        self.crop_size = crop_size

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_dir = self.data_list[idx]
        y = self.category[sample_dir.split('/')[-2]]
        try:
            x = sio.loadmat(sample_dir)[self.modal]
        except:
            print(f"Error loading {sample_dir}") # Basic error handling
            return torch.zeros(1), torch.zeros(1) # Placeholder

        x = (x - 42.3199) / 4.9802
        x = x[:114, ::8]
        x = x.reshape(114, -1)
        if self.crop_size is not None:
            original_size = x.shape[1]
            if original_size > self.crop_size:
                indices = np.linspace(0, original_size - 1, self.crop_size, dtype=int)
                x = x[:, indices]
            else:
                pass
        x = torch.FloatTensor(x)
        label = F.one_hot(torch.tensor(y, dtype=torch.long), num_classes=self.num_classes).float()
        return x, label

def get_NTUHAR(root, batch_size, crop_size=100):
    train_loader = DataLoader(
        dataset=NTU_Dataset(os.path.join(root, 'NTU-Fi_HAR/train_amp/'), crop_size=crop_size),
        batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        dataset=NTU_Dataset(os.path.join(root, 'NTU-Fi_HAR/test_amp/'), crop_size=crop_size),
        batch_size=batch_size, shuffle=False
    )
    param = [114, 250, 6]
    if crop_size is not None:
        param[1] = crop_size
    return [train_loader, test_loader, "NTU_HAR", param]

def get_NTUHumanID(root, batch_size, crop_size=100):
    train_loader = DataLoader(
        dataset=NTU_Dataset(os.path.join(root, 'NTU-Fi-HumanID/train_amp/'), crop_size=crop_size),
        batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        dataset=NTU_Dataset(os.path.join(root, 'NTU-Fi-HumanID/test_amp/'), crop_size=crop_size),
        batch_size=batch_size, shuffle=False
    )
    param = [114, 250, 14]
    if crop_size is not None:
        param[1] = crop_size
    return [train_loader, test_loader, "NTU_HumanID", param]

if __name__ == '__main__':
    # Unit test
    root_dir = '/home/chenjiayi/workspace/willm/wifi_data'
    batch_size = 64
    try:
        print("Testing NTU_HAR...")
        train_loader, test_loader, name, param = get_NTUHAR(root_dir, batch_size)
        print(f"Successfully loaded {name}")
        for x, y in train_loader:
             print(f"Train Batch - X shape: {x.shape}, Y shape: {y.shape}")
             break
    except Exception as e:
        print(f"Error loading NTU_HAR: {e}")
        
    try:
        print("Testing NTU_HumanID...")
        train_loader, test_loader, name, param = get_NTUHumanID(root_dir, batch_size)
        print(f"Successfully loaded {name}")
    except Exception as e:
        print(f"Error loading NTU_HumanID: {e}")
