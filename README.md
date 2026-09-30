# WiLLM

**WiFi CSI datasets, preprocessing tools, and PyTorch data loaders.**

WiLLM brings together data loading utilities for WiFi channel state information (CSI) research, covering human activity recognition, sign language recognition, human identification, and people counting. Each dataset guide explains where to obtain the data, how to prepare it, and what its loader returns.

This public repository focuses on the data utilities developed for WiLLM experiments. Dataset files are obtained separately from their original providers. Model weights and the full pretraining framework are not included in this release.

[Dataset catalog](DATASETS.md) · [Quick start](#quick-start) · [Loader directory](dataloaders/README.md) · [Usage notes](DATASETS.md#usage-notes)

## Dataset overview

| Dataset | Task | Data expected by this integration | Guide |
| --- | --- | --- | --- |
| RF-Net | Human activity recognition | CSI and label tensors in `.pth` files | [RF-Net](dataloaders/RFNet/README.md) |
| NTU-Fi HAR | Human activity recognition | Per-sample `CSIamp` arrays in `.mat` files | [NTU-Fi](dataloaders/NTU_Datasets/README.md) |
| NTU-Fi HumanID | Human identification | Per-sample `CSIamp` arrays in `.mat` files | [NTU-Fi](dataloaders/NTU_Datasets/README.md) |
| UT-HAR | Human activity recognition | NumPy binary arrays distributed with `.csv` filenames | [UT-HAR](dataloaders/UT_HAR/README.md) |
| SignFi | Sign language recognition | CSI and labels in `.mat`, prepared as `.npz` | [SignFi](dataloaders/SignFi/README.md) |
| Baha | Human activity recognition in LOS and NLOS environments | CSI recordings in `.mat`, prepared as `.pth` | [Baha](dataloaders/Baha/README.md) |
| XRF55 | Human action recognition using the WiFi modality | WiFi `.npy` samples packed into a memory-mapped file | [XRF55](dataloaders/XRF55_repo/README.md) |
| Brinkle | Human activity recognition | `csi_trace` recordings in `.mat`, prepared as `.pkl` | [Brinkle](dataloaders/Brinkle/README.md) |
| WiCount, WiFall, WiGesture | Counting, activity recognition, and identification | Locally prepared magnitude and label `.npy` files | [AI RAN adapters](dataloaders/AI_RAN_Datasets/README.md) |

The [catalog](DATASETS.md) lists exact registry keys, tensor layouts, labels, and data availability. The Widar3 entry in the root registry depends on a local loader that is not distributed in this repository.

## Quick start

### 1. Set up the environment

Run the examples from the repository root in a Python environment with PyTorch installed.

```bash
git clone https://github.com/cjychenjiayi/WiLLM.git
cd WiLLM
python -m venv .venv
source .venv/bin/activate
python -m pip install torch numpy scipy scikit-learn
```

For the preprocessing scripts, also install:

```bash
python -m pip install tqdm pandas matplotlib
```

These commands list the dependencies used by the data utilities; this release does not include a pinned environment. GPU-specific PyTorch installation is optional for data loading.

### 2. Prepare a dataset

For a first example, obtain the processed NTU-Fi HAR data linked from [SenseFi](https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark#run) and preserve its class subdirectories:

```text
wifi_data/
└── NTU-Fi_HAR/
    ├── train_amp/
    │   └── <class_name>/*.mat
    └── test_amp/
        └── <class_name>/*.mat
```

`wifi_data/` is an example location on your machine. Pass its path explicitly when calling the loader. See the [NTU-Fi guide](dataloaders/NTU_Datasets/README.md) for the HumanID variant.

### 3. Load a batch

Import the dataset-specific module directly:

```python
from dataloaders.NTU_Datasets.NTU_dataloader import get_NTUHAR

train_loader, test_loader, name, params = get_NTUHAR(
    root="wifi_data",
    batch_size=32,
    crop_size=100,
)

x, y = next(iter(train_loader))
print(name, params)  # NTU_HAR [114, 100, 6]
print(x.shape)       # [batch, 114, 100] for the standard prepared data
print(y.shape)       # [batch, 6], one-hot labels
print(len(train_loader.dataset), len(test_loader.dataset))
```

Direct imports allow each public dataset loader to be used independently. The root `dataloader.py` imports the unpublished Widar3 module at import time, so its unified `load_data` entry point requires that module to be present locally. See [Unified entry point](DATASETS.md#unified-entry-point) for its interface.

## Repository layout

```text
WiLLM/
├── README.md                       # Project overview and first example
├── DATASETS.md                     # Dataset catalog and API notes
├── dataloader.py                   # Registry and unified loading interface
├── dataloaders/
│   ├── README.md                   # Links to the individual loader guides
│   ├── RFNet/
│   ├── NTU_Datasets/
│   ├── UT_HAR/
│   ├── SignFi/
│   ├── Baha/
│   ├── XRF55_repo/                 # WiFi adapters and upstream XRF55 materials
│   ├── Brinkle/
│   └── AI_RAN_Datasets/
├── single_dataset_baseline.py      # Legacy experiment entry point
├── run_single_dataset_baseline.sh  # Legacy experiment launcher
└── LICENSE
```

The legacy baseline scripts depend on `models.basic_model`, which is not part of this public release. Use the loader examples above to integrate the data into your own training code.

## Working with the data

- **Inspect shapes and labels.** Most loaders yield `[batch, channels, time]` with one-hot labels. UT-HAR, Brinkle, and XRF55 have different conventions documented in the catalog.
- **Keep the evaluation protocol explicit.** Several adapters create random sample splits. These splits should not be described as subject-independent or environment-independent evaluations.
- **Prepare only the datasets you need.** Download links, preprocessing steps, filenames, and path arguments are listed in each dataset guide.

## Attribution and license

Please cite the original dataset papers when using their data. Source links and citations are provided in the individual guides. WiLLM integrates these resources; the datasets remain the work of their original authors.

The repository includes an [MIT license](LICENSE). Upstream code and dataset terms remain applicable; this license does not grant rights to redistribute third-party datasets.

## Feedback

For a loader issue, please include the dataset name, the function you called, your Python and PyTorch versions, and the input shape or traceback in a [GitHub issue](https://github.com/cjychenjiayi/WiLLM/issues). This helps distinguish a data preparation issue from a loader issue.
