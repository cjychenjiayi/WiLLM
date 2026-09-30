# WiFi CSI data loaders

This directory contains dataset-specific PyTorch loaders and preprocessing utilities. Start with the guide for the dataset you want to use, then import that module directly from the repository root.

[Project overview](../README.md) · [Dataset catalog and API notes](../DATASETS.md)

## Choose a dataset

| Directory | Loader | Preprocessing | Guide |
| --- | --- | --- | --- |
| `RFNet` | [RFNet_loader.py](RFNet/RFNet_loader.py) | Uses prepared `.pth` tensors | [RF-Net](RFNet/README.md) |
| `NTU_Datasets` | [NTU_dataloader.py](NTU_Datasets/NTU_dataloader.py) | Uses prepared `CSIamp` files | [NTU-Fi HAR and HumanID](NTU_Datasets/README.md) |
| `UT_HAR` | [UT_HAR_dataloader.py](UT_HAR/UT_HAR_dataloader.py) | Uses prepared arrays | [UT-HAR](UT_HAR/README.md) |
| `SignFi` | [SignFi_dataloader.py](SignFi/SignFi_dataloader.py) | [SignFi_preprocess.py](SignFi/SignFi_preprocess.py) | [SignFi](SignFi/README.md) |
| `Baha` | [Baha_dataloader.py](Baha/Baha_dataloader.py) | [Baha_preprocess.py](Baha/Baha_preprocess.py) | [Baha](Baha/README.md) |
| `XRF55_repo` | [XRF55_packed_dataloader.py](XRF55_repo/XRF55_packed_dataloader.py) | [XRF55_packed_preprocess.py](XRF55_repo/XRF55_packed_preprocess.py) | [XRF55 WiFi](XRF55_repo/README.md) |
| `Brinkle` | [Brinkle_dataloader.py](Brinkle/Brinkle_dataloader.py) | [Brinkle_preprocess.py](Brinkle/Brinkle_preprocess.py) | [Brinkle](Brinkle/README.md) |
| `AI_RAN_Datasets` | [AI_RAN_dataloader.py](AI_RAN_Datasets/AI_RAN_dataloader.py) | Expects local prepared `.npy` files | [WiCount, WiFall, and WiGesture](AI_RAN_Datasets/README.md) |

## Example

After preparing SignFi as described in its guide:

```python
from dataloaders.SignFi.SignFi_dataloader import signfi_dataloader

train_loader, test_loader, num_classes = signfi_dataloader(
    folder_path="wifi_data/SignFi",
    batch_size=32,
    crop_size=None,
)

csi, labels = next(iter(train_loader))
print(csi.shape, labels.shape, num_classes)
```

Each adapter has its own return signature and tensor convention. See the [API table](../DATASETS.md#dataset-specific-entry-points) before substituting one loader for another.

## Public repository scope

Brinkle is a standalone adapter and is not registered in the root `dataloader.py`. Widar3's implementation is not distributed, although its import and registry entry remain in the root module. WiMANS is not included in this public tree. The guides above cover the loader directories that are actually present.

Dataset downloads and citations are linked from each guide. Local raw data and generated caches should be stored in your chosen data directory.
