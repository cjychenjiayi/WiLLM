import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os

class AI_RAN_Dataset(Dataset):
    def __init__(self, magnitudes, label, crop_size=None):
        super().__init__()
        self.magnitudes = magnitudes
        self.mean = np.mean(self.magnitudes)
        self.std = np.std(self.magnitudes)
        self.magnitudes = (self.magnitudes - self.mean) / self.std
        num_classes = len(set(label))
        self.label = F.one_hot(torch.tensor(label), num_classes=num_classes).float()
        self.crop_size = crop_size

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        x = self.magnitudes[index].reshape(52, 100)
        if self.crop_size is not None:
            original_size = x.shape[1]
            if original_size > self.crop_size:
                indices = np.linspace(0, original_size - 1, self.crop_size, dtype=int)
                x = x[:, indices]
            else:
                pass
        return x, self.label[index]

def load_airan_data(magnitudes_path, label_path, crop_size=None):
    magnitude = np.load(magnitudes_path).astype(np.float32)
    label = np.load(label_path).astype(np.int64)
    return AI_RAN_Dataset(magnitude, label, crop_size)

def get_local_dataset(mag_path, label_path, batch_size, crop_size=100):
    seed = 42
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dataset = load_airan_data(mag_path, label_path, crop_size=crop_size)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_set, test_set = train_test_split(dataset, test_size=test_size, random_state=seed, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def get_WiCount(root, batch_size, crop_size=100):
    folder_path = os.path.join(root, "AI_RAN_Datasets/WiCount")
    mag_path = os.path.join(folder_path, "magnitude_linear.npy")
    label_path = os.path.join(folder_path, "people.npy")
    train_loader, test_loader = get_local_dataset(mag_path, label_path, batch_size, crop_size=crop_size)
    param = [52, 100, 4]
    if crop_size is not None:
        param[1] = crop_size
    return [train_loader, test_loader, "WiCount", param]

def get_WiFall_action(root, batch_size, crop_size=100):
    folder_path = os.path.join(root, "AI_RAN_Datasets/WiFall")
    mag_path = os.path.join(folder_path, "magnitude_linear.npy")
    label_path = os.path.join(folder_path, "action.npy")
    train_loader, test_loader = get_local_dataset(mag_path, label_path, batch_size, crop_size=crop_size)
    param = [52, 100, 5]
    if crop_size is not None:
        param[1] = crop_size
    return [train_loader, test_loader, "WiFall", param]

def get_WiFall_people(root, batch_size, crop_size=100):
    folder_path = os.path.join(root, "AI_RAN_Datasets/WiFall")
    mag_path = os.path.join(folder_path, "magnitude_linear.npy")
    label_path = os.path.join(folder_path, "people.npy")
    train_loader, test_loader = get_local_dataset(mag_path, label_path, batch_size, crop_size=crop_size)
    param = [52, 100, 10]
    if crop_size is not None:
        param[1] = crop_size
    return [train_loader, test_loader, "WiFall", param]

def get_WiGesture_action(root, batch_size, crop_size=100):
    folder_path = os.path.join(root, "AI_RAN_Datasets/WiGesture")
    mag_path = os.path.join(folder_path, "magnitude_linear.npy")
    label_path = os.path.join(folder_path, "action.npy")
    train_loader, test_loader = get_local_dataset(mag_path, label_path, batch_size, crop_size=crop_size)
    param = [52, 100, 6]
    if crop_size is not None:
        param[1] = crop_size
    return [train_loader, test_loader, "WiGesture", param]

def get_WiGesture_people(root, batch_size, crop_size=100):
    folder_path = os.path.join(root, "AI_RAN_Datasets/WiGesture")
    mag_path = os.path.join(folder_path, "magnitude_linear.npy")
    label_path = os.path.join(folder_path, "people.npy")
    train_loader, test_loader = get_local_dataset(mag_path, label_path, batch_size, crop_size=crop_size)
    param = [52, 100, 8]
    if crop_size is not None:
        param[1] = crop_size
    return [train_loader, test_loader, "WiGesture", param]

if __name__ == '__main__':
    # Unit test
    root_dir = '/home/chenjiayi/workspace/willm/wifi_data'
    batch_size = 64
    try:
        print("Testing WiCount...")
        train_loader, test_loader, name, param = get_WiCount(root_dir, batch_size)
        print(f"Successfully loaded {name}")
        for x, y in train_loader:
             print(f"Train Batch - X shape: {x.shape}, Y shape: {y.shape}")
             break
    except Exception as e:
        print(f"Error loading WiCount: {e}")
