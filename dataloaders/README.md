数据集与数据加载器
=================

概述
---
本目录包含各个数据集的加载器（dataloader）与预处理脚本，方便在模型训练/评估时直接调用。

可用数据集（子目录）
---
- RFNet: RFNet_loader.py（路径：dataloaders/RFNet）
- Brinkle: Brinkle_dataloader.py / Brinkle_preprocess.py（路径：dataloaders/Brinkle）
- SignFi: SignFi_dataloader.py / SignFi_preprocess.py（路径：dataloaders/SignFi）
- UT_HAR: UT_HAR_dataloader.py（路径：dataloaders/UT_HAR）
- Baha: Baha_dataloader.py（路径：dataloaders/Baha）
- NTU_Datasets: NTU_dataloader.py（路径：dataloaders/NTU_Datasets）
- AI_RAN_Datasets:（路径：dataloaders/AI_RAN_Datasets）
- XRF55_repo: xrf55_dataloader.py 与说明文件（路径：dataloaders/XRF55_repo）
- WiMANS: 见子目录（路径：dataloaders/WiMANS）

说明与使用
---
- 每个子目录内通常包含一个主 dataloader（以 *_dataloader.py 或 *_loader.py 命名）和可能的预处理脚本。
- 在训练代码中直接 import 对应模块并构造 Dataset/Loader 即可。例如（示例代码仅供参考）：

```python
from dataloaders.SignFi.SignFi_dataloader import SignFiDataset
dataset = SignFiDataset(root='/path/to/SignFi', split='train')
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```

- 若子目录内有 README.md（如 RFNet、Brinkle、UT_HAR、SignFi、Baha 等），请优先阅读子目录说明，里面通常包含数据下载、解压及预处理的详细步骤。

常见文件
---
- `*_dataloader.py` / `*_loader.py`: 构造 Dataset 类并实现 __len__/__getitem__。
- `*_preprocess.py`: 原始数据到训练格式的预处理脚本。
- `README.md`（子目录）: 数据来源、格式与下载说明。

如果需要
---
- 我可以：
  - 为某个具体数据集补充更详细的 README（包含下载链接与示例命令）。
  - 将示例改写为不依赖 PyTorch 的通用加载示例。
