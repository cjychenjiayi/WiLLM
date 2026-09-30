# NTU-Fi HAR and HumanID

PyTorch loaders for the prepared NTU-Fi amplitude datasets: human activity recognition (`NTUHAR`) and human identification (`NTUHumanID`).

[All datasets](../../DATASETS.md) · [Loader source](NTU_dataloader.py)

## Obtain and organize the data

Download the processed datasets linked in the [SenseFi repository](https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark#run). Keep the train/test split and the class subdirectories:

```text
wifi_data/
├── NTU-Fi_HAR/
│   ├── train_amp/<class_name>/*.mat
│   └── test_amp/<class_name>/*.mat
└── NTU-Fi-HumanID/
    ├── train_amp/<class_name>/*.mat
    └── test_amp/<class_name>/*.mat
```

Each `.mat` sample must contain a `CSIamp` array. The loader reads files one class directory below `train_amp` or `test_amp`; files placed directly in those directories will not be discovered.

## Load a batch

Run from the repository root:

```python
from dataloaders.NTU_Datasets.NTU_dataloader import get_NTUHAR, get_NTUHumanID

train_loader, test_loader, name, params = get_NTUHAR(
    root="wifi_data", batch_size=32, crop_size=100,
)

x, y = next(iter(train_loader))
print(name, params)  # NTU_HAR [114, 100, 6]
print(x.shape)       # [batch, 114, 100]
print(y.shape)       # [batch, 6]

id_train, id_test, id_name, id_params = get_NTUHumanID(
    root="wifi_data", batch_size=32, crop_size=100,
)
```

Both functions return four values: `train_loader, test_loader, name, params`. Their direct-call default is `crop_size=100`. Pass `None` to keep the sequence after the built-in temporal subsampling.

## Processing and labels

- The loader normalizes with fixed constants: `(CSIamp - 42.3199) / 4.9802`.
- It selects the first 114 rows and every eighth time step, then reshapes to `[114, T]`.
- If `T > crop_size`, it selects evenly spaced time indices. It does not pad shorter samples.
- Labels are floating-point one-hot vectors. The wrapper declares 6 HAR classes and 14 HumanID classes; the actual label mapping is built from the class folders.
- Train and test mappings are constructed separately using filesystem glob order. Check that `train_loader.dataset.category == test_loader.dataset.category` before training or evaluating.

The uncropped wrapper metadata is `[114, 250, K]`; inspect the actual sample shape when using a different prepared release.

## Source and citation

The [SenseFi project](https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark) provides the processed data links and benchmark context. Cite the datasets' original work as specified there and the benchmark when using its prepared resources:

> Jianfei Yang et al. *SenseFi: A Library and Benchmark on Deep-Learning-Empowered WiFi Human Sensing*. Patterns, 2023. [Paper](https://arxiv.org/abs/2207.07859).
