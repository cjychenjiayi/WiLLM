# SignFi

Preprocessing and PyTorch loading utilities for WiFi sign language recognition. This integration loads CSI amplitude and derives its class count from the labels present in the prepared files.

[All datasets](../../DATASETS.md) · [Loader source](SignFi_dataloader.py) · [Preprocessing source](SignFi_preprocess.py)

## Obtain the data

The [official SignFi project](https://github.com/yongsen/SignFi) provides data downloads, usage terms, and the original MATLAB example. It includes 276-sign and 150-sign collections. Select a collection appropriate to your experiment rather than mixing every release by default.

Place compatible `.mat` files in one directory, for example:

```text
wifi_data/
└── SignFi/
    └── dataset_lab_276_dl.mat
```

## Preprocess

Run from the repository root:

```bash
python dataloaders/SignFi/SignFi_preprocess.py --root_folder wifi_data/SignFi
```

The script scans `.mat` files directly in this directory. It selects non-metadata keys without the letter `u`, then collects the selected keys beginning with `csi` and `label`. It saves amplitude, phase, labels, and normalization statistics to `all_processed.npz`.

The loader prefers `all_processed.npz` and falls back to `all.npz`. It reads `csi_abs` and `label`, or computes amplitude from a `csi` array if `csi_abs` is absent. The prepared CSI array should have shape `[N, C, T]`, and labels should be a one-dimensional integer array with one entry per sample.

The preprocessing script uses reshaping to reorder the source CSI dimensions and concatenates labels without flattening them. Check the saved array ordering and label shape before training, especially when combining source files with different label layouts.

## Load a batch

```python
from dataloaders.SignFi.SignFi_dataloader import signfi_dataloader

train_loader, test_loader, num_classes = signfi_dataloader(
    folder_path="wifi_data/SignFi",
    batch_size=32,
    use_normalize=True,
    crop_size=None,
)

x, y = next(iter(train_loader))
print(x.shape)       # [batch, channels, time], typically [batch, 90, 200]
print(y.shape)       # [batch, num_classes], one-hot labels
print(num_classes)  # Derived from the prepared labels
```

The function returns **three values**. It does not accept a `num_classes` argument and does not impose a fixed 50-class limit. Its registry key is `SignFi`.

For access to the complete unsplit dataset, use `signfi_dataset(folder_path, use_normalize=True, crop_size=None)`, which returns `dataset, num_classes`.

## Processing and split behavior

Labels are remapped to consecutive class indices and converted to one-hot vectors. The loader creates an 80/20 random sample split with seed 42. Normalization uses the mean and standard deviation of the entire loaded amplitude array; the loader recomputes these values rather than reading the saved statistics.

With `crop_size=None`, it keeps the full time dimension. With a smaller crop size, the dataset uses a random contiguous crop by default. Train and test subsets share this behavior, so test inputs can change across evaluations when cropping is enabled. No padding is applied to shorter sequences.

## Source and citation

The original paper's citation is available from the [SignFi project](https://github.com/yongsen/SignFi):

> Yongsen Ma, Gang Zhou, Shuangquan Wang, Hongyang Zhao, and Woosub Jung. *SignFi: Sign Language Recognition Using WiFi*. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 2(1), Article 23, 2018. [DOI](https://doi.org/10.1145/3191755).
