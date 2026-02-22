# https://github.com/di0002ya/RFNet/blob/master/data_loader.py
# https://arxiv.org/pdf/2111.04566
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import random

def load_rfnet(data_path = "/home/chenjiayi/workspace/willm/wifi_data/RF-Net", test_ratio=0.2, seed=42, normalize=True, crop_size = None):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    x = torch.load(f'{data_path}/X_100_scenarios.pth',weights_only=True)
    y = torch.load(f'{data_path}/Y_100_scenarios.pth', weights_only=True)
    # x: n_scenes, n_classes, shots_per_class, time, channel
    n_scenes, n_classes, shots_per_class, time, channel = x.shape
    x = x.reshape(-1, channel, time)
    y = y.reshape(-1, n_classes)
    mean, std = torch.mean(x), torch.std(x)
    
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=test_ratio, random_state=seed, shuffle=True)
    # : 
    train_dataset = RFNet_Dataset(x_train, y_train, mean, std, normalize=normalize, crop_size = crop_size)
    val_dataset = RFNet_Dataset(x_val, y_val, mean, std, normalize=normalize, crop_size = crop_size)
    
    return train_dataset, val_dataset

class RFNet_Dataset(Dataset):
    def __init__(self, x, y, mean, std, normalize=True, crop_size = None):
        self.x = x
        self.y = y
        self.mean = mean
        self.std = std
        self.normalize = normalize
        self.class_meaning = ["wiping", "walking", "moving", "rotating", "sitting", "standing up"]
        if self.normalize:
            self.x = (self.x - self.mean) / self.std
        self.crop_size = crop_size
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_data = self.x[idx]
        if self.crop_size is not None:
            original_size = x_data.shape[1]
            if original_size > self.crop_size:
                indices = np.linspace(0, original_size - 1, self.crop_size, dtype=int)
                x_data =x_data[:, indices]
            else:
                pass
        return x_data, self.y[idx]

if __name__ == "__main__":
    train_rfnet, test_rfnet = load_rfnet()
    print(train_rfnet[0])
    x, y = train_rfnet[0]
    print(x.shape)
    print(y.shape)
    print(y)