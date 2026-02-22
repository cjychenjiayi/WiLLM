import torch
import os
import numpy as np
import random
from tqdm import tqdm
from Baha_utils import build_dataset, actions_dict, experiment_dict

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def preprocess_data(root_path, save_dir, require_align=True):
    print("Building dataset from:", root_path)
    # build_dataset iterates over E1, E2, E3 folders inside root_path
    data_list = build_dataset(root_path)
    
    # Filter based on length > 900 as in original request
    original_count = len(data_list)
    data_list = [item for item in data_list if item["data"].size(1) > 900]
    print(f"Filtered dataset from {original_count} to {len(data_list)} (length > 900)")

    # Create class mapping
    # (C, A) pair determines the class
    unique_pairs = sorted(list({(item["C"], item["A"]) for item in data_list}))
    class_mapping = {pair: idx + 1 for idx, pair in enumerate(unique_pairs)}
    
    # Align data if required
    if require_align and len(data_list) > 0:
        time_length = min(item["data"].size(1) for item in data_list)
        print(f"Aligning all data to min length: {time_length}")
    else:
        time_length = None

    all_data_tensor = []
    
    processed_list = []
    for item in tqdm(data_list, desc="Processing items"):
        # Assign class label
        item["class"] = class_mapping.get((item["C"], item["A"]))
        
        # Add action description from utils title (actions_dict)
        # Assuming action A maps to actions_dict key
        if item["A"] in actions_dict:
            item["action_description"] = actions_dict[item["A"]]
        
        if require_align and time_length:
            item["data"] = item["data"][:, :time_length]
            
        all_data_tensor.append(item["data"].unsqueeze(0))
        processed_list.append(item)

    if not processed_list:
        print("No data found!")
        return

    # Calculate mean and variance
    all_data_cat = torch.cat(all_data_tensor, dim=0) # (N, 90, T)
    # Mean and variance across (N, T) for each channel (90)
    # Or mean/var across whole dataset?
    # Original code: mean = all_data.mean(dim=0) -> (90, T)
    # This implies normalizing per time-step per channel across dataset?
    # Standard practice is usually normalizing per channel.
    # But let's stick to the user's previous logic:
    # all_data: (N, 90, T)
    # mean(dim=0): (90, T)
    # This assumes time alignment is perfect and meaningful.
    
    mean = all_data_cat.mean(dim=0)
    variance = all_data_cat.var(dim=0, unbiased=False)
    
    # Save
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print(f"Saving processed data to {save_dir}")
    torch.save(processed_list, os.path.join(save_dir, "data_list.pth"))
    torch.save(class_mapping, os.path.join(save_dir, "class_mapping.pth"))
    torch.save(mean, os.path.join(save_dir, "mean.pth"))
    torch.save(variance, os.path.join(save_dir, "variance.pth"))
    
    print("Preprocessing complete.")

if __name__ == "__main__":
    set_seed()
    # Path to MAT files
    root_path = "/home/chenjiayi/workspace/willm/wifi_data/Baha/Dataset-for-Wi-Fi-based-human-activity-recognition-in-LOS-and-NLOS-indoor-environments/MAT"
    
    # Directory to save processed .pth files
    save_dir = "/home/chenjiayi/workspace/willm/wifi_data/Baha/baha_processed"
    
    preprocess_data(root_path, save_dir)
