# RF-Net

A PyTorch adapter for the prepared RF-Net 100-scenario tensors used for human activity recognition.

[All datasets](../../DATASETS.md) · [Loader source](RFNet_loader.py)

## Obtain and organize the data

Start with the [original RF-Net repository](https://github.com/di0002ya/RFNet) for data access and preparation. This adapter expects these prepared files:

```text
wifi_data/
└── RF-Net/
    ├── X_100_scenarios.pth
    └── Y_100_scenarios.pth
```

`X_100_scenarios.pth` is expected to contain a tensor with axes described by the loader as `[scenarios, classes, shots, time, channels]`. `Y_100_scenarios.pth` supplies one-hot labels. The wrapper metadata assumes 60 channels, 512 time steps, and 6 classes.

## Load a batch

`load_rfnet` returns two **datasets**, so wrap them in PyTorch DataLoaders:

```python
from torch.utils.data import DataLoader
from dataloaders.RFNet.RFNet_loader import load_rfnet

train_dataset, test_dataset = load_rfnet(
    data_path="wifi_data/RF-Net",
    test_ratio=0.2,
    normalize=True,
    crop_size=100,
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

x, y = next(iter(train_loader))
print(x.shape)  # [batch, 60, 100] for the expected prepared tensors
print(y.shape)  # [batch, 6]
```

Run this example from the repository root. The registry key is `RFNet`.

## Processing and split behavior

The checked-in loader reshapes the input to `[samples, channels, time]`, computes a global mean and standard deviation, then creates a random sample split. Its default test fraction is 0.2. It fixes the random seed to 42 internally, including when another `seed` argument is supplied.

`crop_size=None` keeps the prepared sequence length. A smaller crop size selects evenly spaced time indices. Labels are returned as stored in the prepared label tensor.

The sample split mixes the flattened scenarios; it is not a held-out-scenario evaluation. The normalization statistics are computed before splitting. Also, the input axis conversion uses `reshape`, not a transpose; verify that your prepared tensor ordering matches the representation you intend to use.

## Source and citation

Please cite the original RF-Net work when using the dataset:

> Shuya Ding, Zhe Chen, Tianyue Zheng, and Jun Luo. *RF-Net: A Unified Meta-Learning Framework for RF-enabled One-Shot Human Activity Recognition*. SenSys, 2021. [Project](https://github.com/di0002ya/RFNet) · [Paper](https://arxiv.org/abs/2111.04566).
