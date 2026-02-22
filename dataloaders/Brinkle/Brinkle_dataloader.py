import os
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from typing import Optional, Tuple, Union

class BrinkleDataset(Dataset):
    def __init__(self, data_list: list, label_dict: dict):
        self.data_list = data_list
        self.label_dict = label_dict
        self.num_classes = len(label_dict)
        self.classes = sorted(label_dict.keys())
        
        # Ensure consistency in label dictionary
        if len(set(label_dict.values())) != self.num_classes:
            print("Warning: Label dictionary values are not unique.")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # Extract CSI magnitude, expected shape: (92, 3, 3, 30) or similar
        x = torch.tensor(item['csi_magnitude'], dtype=torch.float32)
        
        # Flatten spatial dimensions while keeping time dimension
        # (T, Ntx, Nrx, Sub) -> (T, Features)
        # e.g., (92, 3, 3, 30) -> (92, 270)
        x = x.reshape(x.shape[0], -1) 
        
        label_str = item['activity']
        label_idx = self.label_dict[label_str]
        
        # Return one-hot encoded label (float32)
        y = F.one_hot(torch.tensor(label_idx, dtype=torch.long), num_classes=self.num_classes).float()
        
        return x, y

def Brinkle_dataloader(
    root_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    split_ratio: Optional[float] = 0.8
) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    """
    Loads the Brinkle dataset from 'dataset.pkl'.
    
    Args:
        root_path (str): Directory containing dataset.pkl
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle training data
        num_workers (int): Number of DataLoader workers
        split_ratio (float, optional): If < 1.0, splits into train/test. 
                                       If None or 1.0, returns single DataLoader.
    
    Returns:
        DataLoader or (train_loader, test_loader)
    """
    dataset_path = os.path.join(root_path, 'dataset.pkl')
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Run Brinkle_preprocess.py first.")

    with open(dataset_path, 'rb') as f:
        saved_data = pickle.load(f)

    if not isinstance(saved_data, dict) or 'data_list' not in saved_data:
        raise ValueError("Invalid dataset format. Re-run preprocessing.")

    full_dataset = BrinkleDataset(saved_data['data_list'], saved_data['label_dict'])

    # Single Loader
    if split_ratio is None or split_ratio >= 1.0:
        return DataLoader(
            full_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers,
            pin_memory=True
        )

    # Train/Test Split
    train_size = int(split_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    train_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42) # Ensure reproducible splits
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader

if __name__ == "__main__":
    # Test execution
    root = "/home/chenjiayi/workspace/willm/wifi_data/Brinkle"
    if os.path.exists(os.path.join(root, 'dataset.pkl')):
        try:
            print("Loading dataset...")
            tr_loader, te_loader = Brinkle_dataloader(root, batch_size=4, split_ratio=0.8)
            print(f"Train batches: {len(tr_loader)}, Test batches: {len(te_loader)}")
            
            for x, y in tr_loader:
                print(f"Sample Batch - X: {x.shape}, Y: {y.shape}")
                break
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Dataset missing at {root}")


