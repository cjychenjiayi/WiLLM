# Baha

Preprocessing and PyTorch loading utilities for WiFi human activity recognition in line-of-sight (LOS) and non-line-of-sight (NLOS) indoor environments.

[All datasets](../../DATASETS.md) · [Loader source](Baha_dataloader.py) · [Preprocessing source](Baha_preprocess.py)

## Obtain and organize the data

Use the [original dataset repository](https://github.com/lcsig/Dataset-for-Wi-Fi-based-human-activity-recognition-in-LOS-and-NLOS-indoor-environments) or its [Mendeley Data record](https://data.mendeley.com/datasets/v38wjmz6f6/1).

The preprocessing function expects a directory containing `E1`, `E2`, and `E3`:

```text
wifi_data/
└── Baha/
    ├── MAT/
    │   ├── E1/E1_S01_C01_A01_T01.mat
    │   ├── E2/...
    │   └── E3/...
    └── baha_processed/          # Generated output
```

If the downloaded archive contains another parent directory, pass the actual `MAT` directory to the preprocessing function.

## Preprocess

The script's `__main__` block contains machine-specific paths. To use your paths without editing the source, run the following from a Python script saved at the repository root:

```python
import sys
from pathlib import Path

# The preprocessing module imports Baha_utils as a sibling module.
sys.path.insert(0, str(Path("dataloaders/Baha").resolve()))
from Baha_preprocess import preprocess_data, set_seed

if __name__ == "__main__":
    set_seed(42)
    preprocess_data(
        root_path="wifi_data/Baha/MAT",
        save_dir="wifi_data/Baha/baha_processed",
        require_align=True,
    )
```

The pipeline extracts CSI amplitude in dB, retains recordings longer than 900 time steps, and truncates them to the shortest retained sequence when alignment is enabled. Class IDs are based on the observed `(C, A)` pairs in the filenames.

It writes:

- `data_list.pth`: processed sample records with data and labels.
- `class_mapping.pth`: the mapping from `(C, A)` pairs to prepared class IDs.
- `mean.pth` and `variance.pth`: statistics across samples at each channel and time position.

The alignment length depends on the input recordings. A length of 901 is wrapper metadata, not a guaranteed preprocessing output.

## Load a batch

```python
from dataloaders.Baha.Baha_dataloader import Baha_dataloader

train_loader, test_loader = Baha_dataloader(
    data_path="wifi_data/Baha/baha_processed",
    batch_size=32,
    crop_size=100,
    num_workers=0,
)

x, y = next(iter(train_loader))
print(x.shape)  # [batch, 90, 100] when the aligned sequence is longer than 100
print(y.shape)  # [batch, number_of_observed_classes]
```

The standalone loader returns two loaders and remaps the prepared labels to one-hot vectors. Its default split is 80/20 with seed 42. It applies the saved normalization statistics before optional uniform temporal subsampling. The statistics are prepared from all retained samples before splitting.

The registry key is `Baha`. Its wrapper declares `[90, 901, 15]`; inspect the batch and class mapping when using a different subset or a non-default crop size.

## Source and citation

> Baha' A. Alsaify, Mahmoud M. Almazari, Rami Alazrai, and Mohammad I. Daoud. *A dataset for Wi-Fi-based human activity recognition in line-of-sight and non-line-of-sight indoor environments*. Data in Brief, 33, 106534, 2020. [DOI](https://doi.org/10.1016/j.dib.2020.106534).

Dataset attribution and access terms are available from the [original repository](https://github.com/lcsig/Dataset-for-Wi-Fi-based-human-activity-recognition-in-LOS-and-NLOS-indoor-environments).
