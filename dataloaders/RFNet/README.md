# RF-Net: A Unified Meta-Learning Framework for RF-based One-shot Human Activity Recognition

This folder contains the **integrated** PyTorch dataloader for the **RF-Net** dataset. This work is based on the original implementation provided in the [RFNet repository](https://github.com/di0002ya/RFNet). Our contribution mainly focuses on integrating this dataset into the **csi-llm** framework for unified training and evaluation.

We do **not** provide the raw data; please download it from the official source or use the provided links.

## 1. Download Raw Data

The RF-Net dataset (specifically the 100 scenarios dataset used here) is associated with the paper:
> **"RF-Net: A Unified Meta-Learning Framework for RF-based One-shot Human Activity Recognition"**  
> Shuya Ding, Zhe Chen, Tianyue Zheng, Jun Luo.  
> *Proceedings of the 19th ACM Conference on Embedded Networked Sensor Systems (SenSys)*, 2021.

Resources:
*   **Paper**: [https://arxiv.org/pdf/2111.04566](https://arxiv.org/pdf/2111.04566)
*   **Official Repository**: [https://github.com/di0002ya/RFNet](https://github.com/di0002ya/RFNet)

Please place the dataset files (`X_100_scenarios.pth`, `Y_100_scenarios.pth`) into the following directory structure:

```
wifi_data/
└── RF-Net/
    ├── X_100_scenarios.pth
    └── Y_100_scenarios.pth
```

## 2. Data Details

The dataset contains CSI data for 6 human activities collected in 100 different scenarios.

*   **Classes (6)**: `wiping`, `walking`, `moving`, `rotating`, `sitting`, `standing up`
*   **Input Shape**: `(Channels: 60, Time: 512)`
*   **Format**: The raw `.pth` files contain tensors of shape `(n_scenes, n_classes, shots_per_class, time, channel)`. The dataloader reshapes this to `(N_samples, 60, 512)`.

## 3. Usage

You can use the dataloader in your project as follows:

```python
from datas.dataloaders.RFNet.RFNet_loader import load_rfnet

# Create train and test datasets
# By default, it loads from wifi_data/RF-Net
train_dataset, val_dataset = load_rfnet(
    data_path="wifi_data/RF-Net", 
    test_ratio=0.2, 
    seed=42
)

# Access a sample
x, y = train_dataset[0]
print(f"Data shape: {x.shape}")  # Expected: (60, 512)
print(f"Label: {y}")            # Expected: One-hot vector, e.g., [0, 0, 1, 0, 0, 0]
```

## 4. References

This work is based on [RF-Net: A Unified Meta-Learning Framework for RF-based One-shot Human Activity Recognition](https://dl.acm.org/doi/10.1145/3485730.3485946).

If you find this dataset or code useful, please cite the original paper:

```bibtex
@inproceedings{ding2021rf,
  title={Rf-net: A unified meta-learning framework for rf-based one-shot human activity recognition},
  author={Ding, Shuya and Chen, Zhe and Zheng, Tianyue and Luo, Jun},
  booktitle={Proceedings of the 19th ACM Conference on Embedded Networked Sensor Systems},
  pages={217--230},
  year={2021}
}
```
