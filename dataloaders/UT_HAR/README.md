# UT-HAR

A PyTorch loader for the prepared UT-HAR activity recognition data, using 7-class metadata and 250 time steps with 90 CSI features before optional temporal subsampling.

[All datasets](../../DATASETS.md) · [Loader source](UT_HAR_dataloader.py)

## Obtain and organize the data

Use the processed UT-HAR data linked from [SenseFi](https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark#run). The [original project](https://github.com/ermongroup/Wifi_Activity_Recognition) provides the dataset's research context.

```text
wifi_data/
└── UT_HAR/
    ├── data/
    │   ├── X_train.csv
    │   ├── X_val.csv
    │   └── X_test.csv
    └── label/
        ├── y_train.csv
        ├── y_val.csv
        └── y_test.csv
```

Despite the `.csv` suffix, these files must contain NumPy binary arrays readable with `numpy.load`. Do not convert them to text CSV.

## Load a batch

Run from the repository root:

```python
from dataloaders.UT_HAR.UT_HAR_dataloader import get_UTHAR

train_loader, test_loader, name, params = get_UTHAR(
    root="wifi_data", batch_size=32, crop_size=100,
)

x, y = next(iter(train_loader))
print(name, params)  # UT_HAR [250, 90, 7]
print(x.shape)       # [batch, 9000], flattened 100 x 90 samples
print(y.shape)       # [batch, 7] for the standard prepared labels

# Restore time and feature axes when your model needs them.
x_time_first = x.reshape(x.shape[0], -1, 90)
x_channels_first = x_time_first.transpose(1, 2)
```

## Output and split behavior

The function returns **four values** when the expected files exist. Its registry key is `UTHAR`.

- Without cropping, each sample is flattened from `[250, 90]` to `[22500]`.
- With `crop_size=100`, uniform temporal subsampling precedes flattening, producing `[9000]`.
- `params` remains `[250, 90, 7]` even when cropping is enabled. Use the actual tensor shape to size your model.
- Labels are one-hot encoded with width inferred from the maximum label plus one.
- The evaluation loader combines the supplied validation and test sets.
- Min-max normalization is computed separately for each loaded data array. The training loader uses `drop_last=True`.

All six files are required. The missing-data path returns only `None, None, "UT_HAR"`; confirm the directory layout before unpacking four values.

## Source and citation

See the [original project](https://github.com/ermongroup/Wifi_Activity_Recognition) and [SenseFi](https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark) for dataset attribution and the prepared distribution.

> Siamak Yousefi et al. *A Survey on Behavior Recognition Using WiFi Channel State Information*. IEEE Communications Magazine, 2017. [Paper](https://ieeexplore.ieee.org/document/8067693).
