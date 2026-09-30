# WiLLM dataset catalog

This catalog describes the data adapters included in the public repository. Shapes and return values below follow the checked-in implementation. Raw dataset specifications and this repository's preprocessing choices may differ.

[Project overview](README.md) · [Loader directory](dataloaders/README.md)

## Data availability

The repository provides loader source code, selected preprocessing scripts, and documentation. Obtain full datasets separately through the original sources linked in the guides.

| Dataset family | Source and preparation guide | Availability in this repository |
| --- | --- | --- |
| RF-Net | [Guide](dataloaders/RFNet/README.md) | Loader for the prepared 100-scenario tensors |
| NTU-Fi HAR and HumanID | [Guide](dataloaders/NTU_Datasets/README.md) | Loaders for the prepared amplitude data |
| UT-HAR | [Guide](dataloaders/UT_HAR/README.md) | Loader for the prepared train, validation, and test arrays |
| SignFi | [Guide](dataloaders/SignFi/README.md) | MAT-to-NPZ preprocessing and loader |
| Baha | [Guide](dataloaders/Baha/README.md) | MAT preprocessing and loader |
| XRF55 | [Guide](dataloaders/XRF55_repo/README.md) | WiFi packing script, packed loader, and upstream materials |
| Brinkle | [Guide](dataloaders/Brinkle/README.md) | MAT preprocessing and standalone loader; not in the root registry |
| WiCount, WiFall, WiGesture | [Guide](dataloaders/AI_RAN_Datasets/README.md) | Adapters for local prepared arrays; no public data download is provided here |
| Widar3 | Root registry entry only | Loader source and processed data are not included |

## Shapes and labels

The table shows **one sample**, before batching, with `crop_size=None`. `C` is the feature/channel dimension, `T` is time, and `K` is the number of classes. A PyTorch DataLoader adds a batch dimension. Class counts marked as metadata come from the root wrapper and should be checked against the prepared data.

| Registry key | Task | Sample layout | Labels | Class count |
| --- | --- | --- | --- | --- |
| `RFNet` | Activity recognition | `[60, 512]` for the expected prepared tensors | One-hot `[K]` | 6 |
| `NTUHAR` | Activity recognition | `[114, T]`; metadata expects `T=250` | One-hot `[K]` | 6 in metadata; folders determine labels |
| `NTUHumanID` | Human identification | `[114, T]`; metadata expects `T=250` | One-hot `[K]` | 14 in metadata; folders determine labels |
| `UTHAR` | Activity recognition | Flat `[250 × 90]`, or `[22500]` | One-hot `[K]` | 7 in metadata; inferred from labels |
| `SignFi` | Sign language recognition | `[C, T]` from the prepared array; typically `[90, 200]` | One-hot `[K]` | Inferred from distinct labels |
| `Baha` | Activity recognition | `[90, T]`, aligned during preprocessing | One-hot `[K]` | Inferred from prepared labels; wrapper declares 15 |
| `Xrf55` | Action recognition | `[270, 1000]` | Integer scalar, `0…54` | 55 |
| `WiCount` | People counting | `[52, 100]` | One-hot `[K]` | 4 in metadata |
| `WiFallact` | Activity recognition | `[52, 100]` | One-hot `[K]` | 5 in metadata |
| `WiFallid` | Human identification | `[52, 100]` | One-hot `[K]` | 10 in metadata |
| `WiGestureact` | Gesture recognition | `[52, 100]` | One-hot `[K]` | 6 in metadata |
| `WiGestureid` | Human identification | `[52, 100]` | One-hot `[K]` | 8 in metadata |
| Not registered: Brinkle | Activity recognition | `[92, F]`, time first; `F` depends on the recording | One-hot `[K]` | Inferred from activity names |

Registry keys are case-sensitive: use `Xrf55`, `UTHAR`, and `NTUHAR` exactly as written. SignFi has no fixed 50-class limit in the checked-in loader. Baha is an activity dataset, not a respiration or heart-rate dataset.

## Local data layout

Only create the directories for the datasets you use. Names and capitalization below match the loaders.

```text
wifi_data/
├── RF-Net/
│   ├── X_100_scenarios.pth
│   └── Y_100_scenarios.pth
├── NTU-Fi_HAR/
│   ├── train_amp/<class_name>/*.mat
│   └── test_amp/<class_name>/*.mat
├── NTU-Fi-HumanID/
│   ├── train_amp/<class_name>/*.mat
│   └── test_amp/<class_name>/*.mat
├── UT_HAR/
│   ├── data/{X_train,X_val,X_test}.csv
│   └── label/{y_train,y_val,y_test}.csv
├── SignFi/
│   └── all_processed.npz
├── Baha/
│   └── baha_processed/
│       ├── data_list.pth
│       ├── class_mapping.pth
│       ├── mean.pth
│       └── variance.pth
├── xrf55_all/
│   ├── xrf55_data.mmap
│   └── xrf55_labels.npy
├── Brinkle/
│   └── dataset.pkl
└── AI_RAN_Datasets/
    ├── WiCount/{magnitude_linear,people}.npy
    ├── WiFall/{magnitude_linear,action,people}.npy
    └── WiGesture/{magnitude_linear,action,people}.npy
```

Braces in this diagram abbreviate separate filenames. UT-HAR's `.csv` files are read with `numpy.load`; ordinary text CSV files are not compatible.

## Dataset-specific entry points

These modules can be imported independently of the unpublished Widar3 loader. Follow each guide for complete examples and required input files.

| Module under `dataloaders/` | Function | Return value |
| --- | --- | --- |
| `RFNet/RFNet_loader.py` | `load_rfnet(...)` | `train_dataset, test_dataset` |
| `NTU_Datasets/NTU_dataloader.py` | `get_NTUHAR(...)`, `get_NTUHumanID(...)` | `train_loader, test_loader, name, params` |
| `UT_HAR/UT_HAR_dataloader.py` | `get_UTHAR(...)` | `train_loader, test_loader, name, params` when files are present |
| `SignFi/SignFi_dataloader.py` | `signfi_dataloader(...)` | `train_loader, test_loader, num_classes` |
| `SignFi/SignFi_dataloader.py` | `signfi_dataset(...)` | `dataset, num_classes` |
| `Baha/Baha_dataloader.py` | `Baha_dataloader(...)` | `train_loader, test_loader` |
| `XRF55_repo/XRF55_packed_dataloader.py` | `load_packed_dataloader(...)` | `train_loader, test_loader` |
| `Brinkle/Brinkle_dataloader.py` | `Brinkle_dataloader(...)` | Two loaders by default; one loader when `split_ratio=None` or `>=1` |
| `AI_RAN_Datasets/AI_RAN_dataloader.py` | `get_WiCount(...)`, `get_WiFall_action(...)`, `get_WiFall_people(...)`, `get_WiGesture_action(...)`, `get_WiGesture_people(...)` | `train_loader, test_loader, name, params` |

## Unified entry point

The root interface is:

```python
load_data(dataset_name, root, batch_size=4, crop_size=None, idx=0)
```

It returns `train_loader, test_loader, params`. In most wrappers, `params` describes `[channels, time_steps, num_classes]`, but it is metadata rather than a validated tensor schema. UT-HAR uses `[250, 90, 7]` while emitting flattened samples; Baha and Widar3 also have hardcoded metadata. Inspect a batch before constructing a model.

**Availability:** `dataloader.py` unconditionally imports `dataloaders.Widar3.widar_dataloader`, which was removed from the public repository. A fresh clone cannot import the unified entry point without that local module. Use the dataset-specific entry points above for the public loaders.

For an existing local setup that includes the Widar3 module:

```python
from dataloader import load_data

train_loader, test_loader, params = load_data(
    dataset_name="RFNet",
    root="wifi_data",
    batch_size=32,
    crop_size=100,
)
```

The current registry uses `Widar3` with an `idx` argument: `idx=0` selects the combined file and a nonzero `idx` is forwarded to the local shard loader. Names such as `Widar3_all` and `Widar3_1` are not registered keys. The public wrapper does not establish the valid shard range or validate the private loader's output shape.

## Usage notes

### Cropping and tensor layout

`crop_size` is generally applied on the time axis only when the original sequence is longer than the requested size. RF-Net, NTU-Fi, UT-HAR, Baha, and the AI RAN adapters use evenly spaced indices. SignFi uses a random contiguous crop by default, including for its test subset. The XRF55 root wrapper adds optional uniform subsampling; the standalone packed loader has no `crop_size` argument.

These loaders generally do not pad short sequences. Brinkle uses a separate fixed-length preprocessing rule. Metadata can differ from actual length when the requested crop is larger than a sample.

### Label conventions

Most loaders return floating-point one-hot labels; XRF55 returns integer class indices. A training loop that accepts either can convert labels as follows:

```python
targets = y.argmax(dim=-1) if y.ndim > 1 else y
targets = targets.long()
```

NTU-Fi constructs its class mapping independently from the folders in each split. Check that the train and test `dataset.category` mappings agree. The AI RAN adapter expects contiguous zero-based labels. Baha and SignFi remap their observed labels.

### Splits and normalization

RF-Net, SignFi, Baha, XRF55's packed loader, Brinkle, and the AI RAN adapters use random sample splits, typically 80/20. NTU-Fi uses the supplied train/test directories. UT-HAR uses the supplied training set and combines validation and test arrays into its evaluation loader.

RF-Net, SignFi, Baha, and AI RAN normalization statistics are computed or prepared using the full dataset before splitting. UT-HAR applies min-max normalization separately to each loaded data array. NTU-Fi and the packed XRF55 loader use fixed normalization constants. Record these choices when reporting results; the adapters do not implement a common subject-held-out or environment-held-out benchmark protocol.

### Memory and paths

Pass explicit paths rather than relying on the original author's machine-specific defaults. `root` in the registry is the parent directory of the dataset folders; `data_path`, `folder_path`, and `data_dir` in standalone loaders usually point to a specific dataset directory.

For XRF55, `load_to_ram=False` opens the packed data as a memory map. The root wrapper currently requests RAM loading. Other adapters may load their prepared arrays fully into memory.

### Scope of the legacy experiment scripts

The root baseline entry point imports `models.basic_model`, which is not included in the public repository. Its label conversion also assumes one-hot labels. It is retained as an experiment reference, not as the quick-start path for all public loaders.

For help with an adapter, open an [issue](https://github.com/cjychenjiayi/WiLLM/issues) with the exact loader call, prepared file layout, and traceback.
