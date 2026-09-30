# Brinkle

A standalone preprocessing pipeline and PyTorch loader for CSI activity recordings. The directory retains the name `Brinkle` used in the experiment code.

[All datasets](../../DATASETS.md) · [Loader source](Brinkle_dataloader.py) · [Preprocessing source](Brinkle_preprocess.py)

## Data source and input format

The University of Twente hosts Jeroen Klein Brinke's [Channel state information (WiFi traces) for 6 activities](https://research.utwente.nl/en/datasets/channel-state-information-wifi-traces-for-6-activities/). Use the dataset record for download access and attribution.

This adapter expects `.mat` files containing a `csi_trace` structure whose frames have a `csi` field. It derives activity names from the second underscore-separated component of each filename:

```text
wifi_data/
└── Brinkle/
    ├── 1_clapping_1.mat
    ├── 1_waving_2.mat
    └── ...
```

Nested folders are also scanned. The actual class vocabulary is derived from accepted filenames rather than a hardcoded activity list.

## Preprocess

From the repository root:

```bash
python dataloaders/Brinkle/Brinkle_preprocess.py \
    --root_path wifi_data/Brinkle \
    --output dataset.pkl \
    --workers 4
```

The script extracts CSI magnitude, aligns compatible antenna/subcarrier shapes, and interpolates missing values where possible. It retains recordings with lengths in `[92, 110)` and truncates them to 92 time steps. Files that fail parsing or filtering are skipped.

The output `wifi_data/Brinkle/dataset.pkl` contains `data_list` and `label_dict`. Review the printed valid-sample count and class dictionary before training.

## Load a batch

```python
from dataloaders.Brinkle.Brinkle_dataloader import Brinkle_dataloader

train_loader, test_loader = Brinkle_dataloader(
    root_path="wifi_data/Brinkle",
    batch_size=32,
    split_ratio=0.8,
    num_workers=0,
)

x, y = next(iter(train_loader))
print(x.shape)  # [batch, 92, features], time first
print(y.shape)  # [batch, num_classes], one-hot labels

# Convert to channels first if required by your model.
x_channels_first = x.transpose(1, 2)
```

Spatial dimensions are flattened per time step: a `[92, 3, 3, 30]` recording becomes `[92, 270]`. The feature count depends on the source recording. All samples in a batch must have matching feature dimensions.

The default split is 80/20 using seed 42. With `split_ratio=None` or `split_ratio>=1`, the function returns a **single** DataLoader instead of a pair. There is no additional normalization in the loader.

Brinkle is not registered in the root `dataloader.py`; use this module directly.

## Source and citation

Please use the citation associated with the [University of Twente dataset record](https://research.utwente.nl/en/datasets/channel-state-information-wifi-traces-for-6-activities/) and its linked publication, *Dataset: Channel State Information for Different Activities, Participants and Days*, by Jeroen Klein Brinke and Nirvana Meratnia, 2019.
