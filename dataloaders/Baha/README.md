# A Dataset for Wi-Fi-based Human Activity Recognition (Baha Dataset)

This folder provides a PyTorch dataloader and preprocessing script for the **Baha** dataset (Wi-Fi-based Human Activity Recognition in LOS and NLOS Indoor Environments). We do **not** provide the raw data; please download it from the official source.

## 1. Download Raw Data

The dataset was proposed in the paper:
> **"A dataset for Wi-Fi-based human activity recognition in line-of-sight and non-line-of-sight indoor environments"**  
> Baha’ A. Alsaify, Mahmoud M. Almazari, Rami Alazrai, Mohammad I. Daoud.  
> *Data in Brief*, Volume 33, 2020.

You can download the raw `.mat` files from the official repository:
*   **Official Repository**: [https://github.com/lcsig/Dataset-for-Wi-Fi-based-human-activity-recognition-in-LOS-and-NLOS-indoor-environments](https://github.com/lcsig/Dataset-for-Wi-Fi-based-human-activity-recognition-in-LOS-and-NLOS-indoor-environments)
*   **Mendeley Data**: [https://data.mendeley.com/datasets/v38wjmz6f6/1](https://data.mendeley.com/datasets/v38wjmz6f6/1)

Please place the downloaded `MAT` folder (containing subfolders `E1`, `E2`, `E3`) into the following directory structure:

```
wifi_data/
└── Baha/
    └── Dataset-for-Wi-Fi-based-human-activity-recognition-in-LOS-and-NLOS-indoor-environments/
        └── MAT/
            ├── E1/
            │   ├── E1_S01_C01_A01_T01.mat
            │   └── ...
            ├── E2/
            └── E3/
```

*(Note: The exact path can be configured in the preprocessing script if your structure differs.)*

## 2. Preprocessing

The raw `.mat` files need to be processed to extract CSI amplitude, normalize, and organize into a unified PyTorch-friendly format. We provide a script to do this automatically using **multiprocessing** for efficiency.

Run the preprocessing script:

```bash
python datas/dataloaders/Baha/Baha_preprocess.py
```

This script will:
1.  Read all `.mat` files from the specified source directory.
2.  Extract CSI data, compute amplitude (in dB), and handle packet alignment.
3.  Compute global mean and variance for normalization.
4.  Save the processed data (data list, mean, variance) to `wifi_data/baha_processed/`.

## 3. Usage

After preprocessing, you can use the dataloader in your project:

```python
from datas.dataloaders.Baha.Baha_dataloader import Baha_dataloader

# Create train and test loaders
# By default, loads from wifi_data/baha_processed
train_loader, test_loader = Baha_dataloader(
    data_path="wifi_data/baha_processed",
    batch_size=32,
    crop_size=None  # Optional: crop/resize time dimension
)

for x, y in train_loader:
    print(f"Data shape: {x.shape}")   # Expected: (Batch, 90, Time)
    print(f"Label shape: {y.shape}")  # One-hot encoded labels
```

## 4. Dataset Details

*   **Activities**: 6 main activities (Sitting, Walking, Moving, Rotating, etc.) or 12 fine-grained actions depending on mapping.
*   **Subjects**: 30 volunteers.
*   **Environments**: 3 different indoor environments (E1, E2, E3).
*   **Data Format**: CSI Amplitude (Magnitude) in dB.

## Citation

If you use this dataset, please cite the original paper:

```bibtex
@article{alsaify2020dataset,
  title={A dataset for Wi-Fi-based human activity recognition in line-of-sight and non-line-of-sight indoor environments},
  author={Alsaify, Baha’ A and Almazari, Mahmoud M and Alazrai, Rami and Daoud, Mohammad I},
  journal={Data in Brief},
  volume={33},
  pages={106534},
  year={2020},
  publisher={Elsevier}
}
```

---
*Note: This repository only provides the data loading and preprocessing utilities for PyTorch. All rights to the dataset belong to the original authors.*
