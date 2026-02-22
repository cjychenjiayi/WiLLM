# UT_HAR: University of Toronto Human Activity Recognition Dataset

This folder contains the PyTorch dataloader for the **UT_HAR** dataset.

## 1. Dataset Information

*   **Classes (7)**: `bed`, `run`, `walk`, `pickup`, `stand2sit`, `sit2stand`, `empty`.
*   **Input Shape**: `[Batch, 250, 90]`
    *   **Time Steps**: 250
    *   **Subcarriers x Antennas**: 90 (30 subcarriers * 3 antennas)
*   **Source**: Collected using Intel 5300 NICs.

## 2. Directory Structure

Ensure the dataset is placed in the following structure:

```
wifi_data/
└── UT_HAR/
    ├── data/
    │   ├── X_train.csv
    │   ├── X_test.csv
    │   └── X_val.csv
    └── label/
        ├── y_train.csv
        ├── y_test.csv
        └── y_val.csv
```

## 3. Usage

```python
from datas.dataloaders.UT_HAR.UT_HAR_dataloader import get_UTHAR

# root: path to wifi_data folder
train_loader, test_loader, dataset_name = get_UTHAR(root='path/to/wifi_data', batch_size=64)
```

## 4. Reference
Please cite the relevant paper if you use this dataset in your work:

> **"A Survey on Behavior Recognition Using WiFi Channel State Information"**
> Siamak Yousefi, Hirokazu Narui, Sankalp Dayal, Stefano Ermon, Shahrokh Valaee.
> *IEEE Communications Magazine*, 2017.

*   **Paper Link**: [IEEE Xplore](https://ieeexplore.ieee.org/document/8067693)
*   **Official Repository**: [ermongroup/Wifi_Activity_Recognition](https://github.com/ermongroup/Wifi_Activity_Recognition)

## 5. Download Instructions

You can download the dataset from the official repository or directly via the provided Google Drive link in their README.
After downloading, organize the files as shown in Section 2.
