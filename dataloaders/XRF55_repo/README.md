# XRF55 WiFi data adapter

This directory contains WiFi preprocessing and loading utilities used by WiLLM, alongside XRF55 upstream experiment and hardware materials. The packed adapter described here uses the WiFi modality for 55-class human action recognition.

[All datasets](../../DATASETS.md) · [Packed loader](XRF55_packed_dataloader.py) · [Packing script](XRF55_packed_preprocess.py) · [Upstream Q&A](XRF55-QA.md)

## Obtain and prepare the data

Use the [official XRF55 project page](https://aiotgroup.github.io/XRF55/) for data downloads and upstream instructions. This packing script consumes prepared WiFi `.npy` samples; it does not decode raw CSI capture files.

It recursively scans directories whose final component is exactly `WiFi`. Filenames must contain the one-based action ID as their second underscore-separated component, such as `01_55_05.npy`.

```text
wifi_data/
└── xrf55_all/
    └── xrf55/
        └── <scene_or_split>/
            └── WiFi/
                ├── 01_01_01.npy
                └── 01_55_05.npy
```

The expected sample shape is `[270, 1000]`. The packer also accepts arrays with 270,000 elements by reshaping them, so verify channel/time ordering before packing.

## Pack the samples

The packing script uses the module constants `DATA_ROOT`, `OUTPUT_DIR`, and `NUM_WORKERS`. To supply paths without modifying its source, run this from a Python script at the repository root:

```python
from dataloaders.XRF55_repo import XRF55_packed_preprocess as packer

if __name__ == "__main__":
    packer.DATA_ROOT = "wifi_data/xrf55_all/xrf55"
    packer.OUTPUT_DIR = "wifi_data/xrf55_all"
    packer.NUM_WORKERS = 4
    packer.main()
```

The output is:

```text
wifi_data/xrf55_all/
├── xrf55_data.mmap    # float32, [N, 270, 1000]
└── xrf55_labels.npy   # int64, [N], labels 0 through 54
```

The memory map is allocated for every discovered filename. The current script does not remove allocated rows when a file fails loading or shape validation. Validate the input files before packing, and keep the `.mmap` and labels from the same run together. Re-running in the same output directory overwrites the packed files.

## Load a batch

Use disk-backed loading when you do not want to copy the full array into RAM. Run from a Python script at the repository root:

```python
from dataloaders.XRF55_repo.XRF55_packed_dataloader import load_packed_dataloader

if __name__ == "__main__":
    train_loader, test_loader = load_packed_dataloader(
        data_dir="wifi_data/xrf55_all",
        batch_size=32,
        load_to_ram=False,
    )

    x, y = next(iter(train_loader))
    print(x.shape)  # [batch, 270, 1000]
    print(y.shape)  # [batch], integer class indices
```

The function returns two DataLoaders. The checked-in implementation uses eight workers and normalizes samples with `(x - 8.7824) / 4.8665`. It has no `crop_size` argument. The root wrapper, registered as `Xrf55`, adds optional temporal subsampling and selects RAM loading.

Each float32 sample occupies 1,080,000 bytes before other storage and loader overhead. The `.npy` label count determines the memory-map shape.

## Split and label conventions

The packed loader shuffles all packed sample indices with seed 42 and creates an 80/20 split. It does not preserve source train/test directory membership or the original XRF55 evaluation split. Document this protocol when using the adapter.

Labels are zero-based integers, not one-hot vectors. The `actions_dict` display dictionary uses one-based keys; look up an integer label with `actions_dict[int(label) + 1]`.

## Source and citation

> Fei Wang, Yizhe Lv, Mengdie Zhu, Han Ding, and Jinsong Han. *XRF55: A Radio Frequency Dataset for Human Indoor Action Analysis*. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 8(1), 2024. [Project and paper](https://aiotgroup.github.io/XRF55/).

Refer to the original project for the other sensing modalities and hardware tutorials. Upstream materials retain their own attribution and applicable terms.
