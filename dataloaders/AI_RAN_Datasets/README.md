# WiCount WiFall and WiGesture adapters

These adapters load local prepared CSI magnitude arrays for people counting, activity recognition, and human identification. This repository provides the adapter code and expected file schema; it does not provide a public download for these local datasets.

[All datasets](../../DATASETS.md) · [Loader source](AI_RAN_dataloader.py)

## File layout

```text
wifi_data/
└── AI_RAN_Datasets/
    ├── WiCount/
    │   ├── magnitude_linear.npy
    │   └── people.npy
    ├── WiFall/
    │   ├── magnitude_linear.npy
    │   ├── action.npy
    │   └── people.npy
    └── WiGesture/
        ├── magnitude_linear.npy
        ├── action.npy
        └── people.npy
```

Each magnitude sample must contain `52 × 100` values. Label files must contain one integer per sample, in the same sample order. Labels must be contiguous and zero-based, because the adapter sets the one-hot width to the number of unique labels.

## Available tasks

| Registry key | Function | Label file | Wrapper class count |
| --- | --- | --- | --- |
| `WiCount` | `get_WiCount` | `WiCount/people.npy` | 4 |
| `WiFallact` | `get_WiFall_action` | `WiFall/action.npy` | 5 |
| `WiFallid` | `get_WiFall_people` | `WiFall/people.npy` | 10 |
| `WiGestureact` | `get_WiGesture_action` | `WiGesture/action.npy` | 6 |
| `WiGestureid` | `get_WiGesture_people` | `WiGesture/people.npy` | 8 |

Wrapper class counts describe the expected local collections. The one-hot label width is inferred from the actual label file.

## Load a batch

From the repository root, with the prepared arrays in place:

```python
from dataloaders.AI_RAN_Datasets.AI_RAN_dataloader import get_WiCount

train_loader, test_loader, name, params = get_WiCount(
    root="wifi_data", batch_size=32, crop_size=100,
)

x, y = next(iter(train_loader))
print(name, params)  # WiCount [52, 100, 4]
print(x.shape)       # [batch, 52, 100]
print(y.shape)       # [batch, 4] for the expected labels
```

Each task function returns four values. `root` points to the parent of `AI_RAN_Datasets`, not to an individual task directory.

## Processing and split behavior

The adapter converts magnitudes to float32, standardizes them with the full array's mean and standard deviation, and creates an 80/20 random sample split with seed 42. Samples are reshaped to `[52, 100]` before optional uniform temporal subsampling.

With a crop size below 100, the time axis is reduced. A larger crop size does not pad the sample, although the wrapper reports the requested size in `params`.

For data access and attribution, refer to the provider of your local arrays. The task names in this directory alone do not identify a downloadable release or a paper citation.
