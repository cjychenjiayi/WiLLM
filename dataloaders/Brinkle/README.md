# Brinkle (Jeroen Klein Brinke's CSI Dataset)

This directory provides a PyTorch dataloader and preprocessing script for the dataset created by **Jeroen Klein Brinke** at the University of Twente (Pervasive Systems).

## 1. Original Paper & Dataset
The dataset originates from the following work:

> **Comparison of channel state information-based indoor localization with different input representations**  
> *Jeroen Klein Brinke, Nirvana Meratnia.*  
> Proceedings of the 2nd Workshop on Data Acquisition To Analysis (DATA '19), November 2019.  
> [ACM Digital Library](https://dl.acm.org/doi/10.1145/3359427.3361913)

It is also part of his thesis work:
> **"Interwoven Waves: Enhancing the Scalability and Robustness of Wi-Fi Channel State Information for Human Activity Recognition"**  
> *Jeroen Klein Brinke, University of Twente, 2019/2024.*

### Dataset Details
*   **Activities (6 types):** `clapping`, `waving`, `falling`, `walking`, `jumping`, `nothing`
*   **Format:** The raw data consists of `.mat` files containing CSI traces recorded using the Linux 802.11n CSI Tool (Intel 5300 NIC).

## 2. Directory Structure
Please ensure your raw data is placed in the following structure (reflecting how the preprocessing script expects it):

```
wifi_data/
└── Brinkle/
    ├── 1_clapping_1.mat
    ├── 1_waving_2.mat
    └── ... (flat list of .mat files is supported by the script)
```

> **Note:** The original dataset might be organized into subfolders like `day_1/`, `day_2/`, etc. Our script recursively scans for `.mat` files, so subfolders are fine, but the filenames must contain the activity (e.g., `1_clapping_1.mat`).

## 3. Preprocessing

Before training, you must run the preprocessing script to parse the `.mat` files, interpolate missing CSI data, and save the dataset into a single `dataset.pkl` file.

### Run Preprocessing

```bash
# Activate your environment
conda activate willm

# Run the preprocessing script
python datas/dataloaders/Brinkle/Brinkle_preprocess.py --root_path /home/chenjiayi/workspace/willm/wifi_data/Brinkle --workers 16
```

**Workflow:**
1.  **Scanning:** Recursively finds all `.mat` files.
2.  **Parsing:** robustness improvements allow handling nested struct arrays common in this dataset.
3.  **Interpolation:** Missing CSI packets are interpolated.
4.  **Filtering:** Samples with length outside `[92, 110)` are discarded; valid samples are truncated to **92** time steps.
5.  **Saving:** Output saved to `dataset.pkl`.

## 4. Usage

Use the provided `Brinkle_dataloader` in your PyTorch training script:

```python
from datas.dataloaders.Brinkle.Brinkle_dataloader import Brinkle_dataloader

# Load dataset (returns train and test loaders by default with 80/20 split)
train_loader, test_loader = Brinkle_dataloader(
    root_path="/home/chenjiayi/workspace/willm/wifi_data/Brinkle",
    batch_size=32,
    split_ratio=0.8
)

# Example iteration
for batch_idx, (data, label) in enumerate(train_loader):
    # data shape: (Batch, 92, 270) -> Time x Features (3*3*30 flattened)
    print(f"Batch {batch_idx}: Data {data.shape}, Label {label.shape}")
```

### Dataloader Arguments
*   `root_path`: Path to the directory containing `dataset.pkl`.
*   `batch_size`: Batch size (default: 32).
*   `split_ratio`: Ratio for train/test split (default: 0.8). If `None` or `1.0`, returns a single dataloader.
*   `num_workers`: Number of subprocesses for data loading.
