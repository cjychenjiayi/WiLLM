# NTU-Fi: NTU Human Activity Recognition & Human ID Dataset

This folder contains the dataloader for the **NTU-Fi** datasets, including **NTU-HAR** (Activity Recognition) and **NTU-HumanID** (Authentication).

## 1. Dataset Information

The data was collected at **Nanyang Technological University (NTU)**.

### NTU_HAR (Human Activity Recognition)
*   **Classes (6)**: `boxing`, `circular motion`, `clean`, `fall`, `run`, `walk`.
*   **Input Shape**: `[Batch, 114, 100]` (Default cropped size) or `[Batch, 114, 500/Full]`
    *   **Subcarriers x Antennas**: 114 (3 transmitting * 3 receiving * 114 subcarriers? Actually 114 is often 3x3x50?? Or specific to their setup. In code it is 114 channels).
    *   **Time**: Variable, cropped to `100` by default.

### NTU_HumanID
*   **Classes (14)**: 14 distinct subjects.
*   **Input Shape**: Same as HAR.

## 2. Directory Structure

```
wifi_data/
├── NTU-Fi_HAR/
│   ├── train_amp/  (.mat files)
│   └── test_amp/   (.mat files)
└── NTU-Fi-HumanID/
    ├── train_amp/  (.mat files)
    └── test_amp/   (.mat files)
```

## 3. Usage

```python
from datas.dataloaders.NTU_Datasets.NTU_dataloader import get_NTUHAR, get_NTUHumanID

# Load HAR
train_loader, test_loader, name, params = get_NTUHAR(root='path/to/wifi_data', batch_size=64, crop_size=100)

# Load HumanID
train_loader, test_loader, name, params = get_NTUHumanID(root='path/to/wifi_data', batch_size=64, crop_size=100)
```

## 4. Reference
These datasets are part of the **SenseFi** benchmark and associated works. Please cite the relevant papers:

**Benchmark Paper:**
> **"SenseFi: A Library and Benchmark on Deep-Learning-Empowered WiFi Human Sensing"**
> Jianfei Yang, Xinyan Chen, Dazhuo Wang, Han Zou, Chris Xiaoxuan Lu, Sumei Sun, Lihua Xie.
> *Patterns*, 2023.

**NTU-HAR Source:**
> **"EfficientFi: Towards Large-Scale Lightweight WiFi Sensing via CSI Compression"**
> *IEEE Internet of Things Journal*, 2022.

**NTU-HumanID Source:**
> **"CAUTION: A Robust WiFi-based Human Authentication System via Few-shot Open-set Gait Recognition"**
> *IEEE Internet of Things Journal*, 2022.

*   **Paper Link**: [SenseFi (arXiv)](https://arxiv.org/abs/2207.07859)
*   **Official Repository**: [xyanchen/WiFi-CSI-Sensing-Benchmark](https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark)

## 5. Download Instructions

The dataset can be downloaded from the **SenseFi** repository or its associated links.
*   **GitHub**: [xyanchen/WiFi-CSI-Sensing-Benchmark](https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark)
*   **Direct Download**: Check the "Datasets" section in the SenseFi README for Google Drive/Baidu Netdisk links.

After downloading, extract the `.mat` files and organize them into `train_amp` and `test_amp` folders as shown in Section 2.
