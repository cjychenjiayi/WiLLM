# Simplified Widar dataloader
import os
from typing import List, Optional, Tuple
import numpy as np
import h5py

# 全局默认路径（指向包含日期子文件夹的 widar 原始数据根目录）
# 可以在外部脚本中覆盖此变量，例如在训练脚本内设置：
# from datas.dataloaders.Widar3.dataloader import WIDAR_ROOT
# WIDAR_ROOT = '/my/custom/path'
WIDAR_ROOT = os.path.join(os.path.expanduser('~'), 'workspace', 'willm', 'wifi_data', 'widar3_raw')

try:
    # when used as a package
    from .widar_process import parse_filename, process_single_file, map_action
except Exception:
    # fallback to absolute import if ran as a script from workspace root
    from datas.dataloaders.Widar3.widar_process import parse_filename, process_single_file, map_action


class WidarDataset:
    """
    简化的 Widar dataloader。

    参数:
    - root_path: widar 原始数据根目录（包含日期子文件夹）
    - scenario_id: int, 0 表示加载所有；否则只加载 parse_filename 中 `id` 等于该值的文件
    - sub_folders: 可选的子用户列表（例如 ['user1','user2']）；默认遍历文件夹下所有子目录
    - convert_map: 可选的动作映射（传入 process 文件中使用的 convert 列表），用于将原始标签映射到目标标签
    - min_len: 处理时的最小帧数（与 process 中逻辑保持一致）

    用法示例:
        ds = WidarDataset('/path/to/widar3_raw', scenario_id=0)
        for amp, action in ds.iterate():
            ...
    """

    def __init__(self,
                 root_path: str,
                 scenario_id: int = 0,
                 sub_folders: Optional[List[str]] = None,
                 convert_map: Optional[List[int]] = None,
                 min_len: int = 50,
                 preloaded: Optional[List[Tuple[np.ndarray, int]]] = None):
        self.root_path = root_path
        self.scenario_id = int(scenario_id)
        self.sub_folders = sub_folders
        self.convert_map = convert_map
        self.min_len = min_len
        self.files: List[Tuple[str, Optional[int]]] = []
        self._preloaded = preloaded
        # if preloaded data is provided, use it directly
        if self._preloaded is None:
            self._collect_files()
        else:
            # files will be indexed into preloaded array
            self.files = [("__inmemory__", i) for i in range(len(self._preloaded))]

    def _collect_files(self):
        if not os.path.isdir(self.root_path):
            return

        date_folders = [d for d in os.listdir(self.root_path) if os.path.isdir(os.path.join(self.root_path, d))]
        for date in sorted(date_folders):
            date_path = os.path.join(self.root_path, date)
            subs = self.sub_folders or [d for d in os.listdir(date_path) if os.path.isdir(os.path.join(date_path, d))]
            for sub in sorted(subs):
                sub_path = os.path.join(date_path, sub)
                if not os.path.isdir(sub_path):
                    continue
                for fname in sorted(os.listdir(sub_path)):
                    if not fname.endswith('.dat'):
                        continue
                    parsed = parse_filename(fname)
                    if parsed is None:
                        continue
                    # scenario filter
                    if self.scenario_id != 0 and parsed.get('id') != self.scenario_id:
                        continue
                    full = os.path.join(sub_path, fname)
                    act = None
                    if self.convert_map is not None:
                        act = map_action(parsed.get('a', None), self.convert_map)
                    self.files.append((full, act))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Optional[Tuple[np.ndarray, Optional[int]]]:
        path, meta = self.files[idx]
        # in-memory preloaded data
        if path == "__inmemory__":
            amp, act = self._preloaded[meta]
            return amp, act

        amp = process_single_file(path)
        if amp is None:
            return None
        if meta is None:
            parsed = parse_filename(os.path.basename(path))
            act = parsed.get('a') if parsed else None
        else:
            act = meta
        return amp, act

    @staticmethod
    def _load_mat_pair(csi_mat_path: str, action_mat_path: str) -> List[Tuple[np.ndarray, int]]:
        """Load a processed pair of .mat files (all_csi_X.mat + action_X.mat).

        Returns a list of (amp, action) tuples.
        """
        items: List[Tuple[np.ndarray, int]] = []
        if not (os.path.exists(csi_mat_path) and os.path.exists(action_mat_path)):
            return items
        try:
            with h5py.File(csi_mat_path, 'r') as f:
                # expect group 'csi_result' with datasets '0','1',...
                if 'csi_result' in f:
                    grp = f['csi_result']
                    # datasets stored as numeric keys
                    keys = sorted(list(grp.keys()), key=lambda x: int(x) if x.isdigit() else x)
                    for k in keys:
                        amp = np.array(grp[k])
                        items.append((amp, None))
                else:
                    # fallback: dataset named 'csi_result'
                    if 'csi_result' in f:
                        arr = np.array(f['csi_result'])
                        for i in range(len(arr)):
                            items.append((arr[i], None))
            with h5py.File(action_mat_path, 'r') as fa:
                if 'action' in fa:
                    actions = np.array(fa['action']).squeeze()
                else:
                    # try common alternatives
                    actions = np.array(fa.get('y') or fa.get('label') or [])
            # attach actions
            if len(actions) > 0:
                # ensure length matches
                if len(actions) == len(items):
                    items = [(amp, int(actions[i])) for i, (amp, _) in enumerate(items)]
                else:
                    # if lengths mismatch, attach as many as possible
                    minlen = min(len(actions), len(items))
                    items = [(items[i][0], int(actions[i])) for i in range(minlen)]
        except Exception:
            return []
        return items

    def iterate(self):
        """按文件顺序惰性返回处理后的 (amp, action) 对。若某文件处理失败则跳过。"""
        for i in range(len(self.files)):
            item = self[i]
            if item is None:
                continue
            yield item

    def info(self) -> str:
        return f"WidarDataset(root={self.root_path}, scenario_id={self.scenario_id}, files={len(self)})"


def demo_example():
    """简单示例：仅供快速手动测试。"""
    import pprint
    root = os.path.join(os.path.expanduser('~'), 'workspace', 'willm', 'wifi_data', 'widar3_raw')
    ds = WidarDataset(root, scenario_id=0)
    print(ds.info())
    cnt = 0
    for amp, act in ds.iterate():
        cnt += 1
        if cnt >= 3:
            break
    print('demo done, read', cnt, 'samples')


if __name__ == '__main__':
    demo_example()


def widar_dataset(folder_path: Optional[str] = None, scenario_id: int = 0, sub_folders: Optional[List[str]] = None, convert_map: Optional[List[int]] = None) -> WidarDataset:
    """兼容 SignFi 风格的构造函数：返回一个 `WidarDataset` 实例。

    - folder_path: 指定数据根路径；为 None 时使用 `WIDAR_ROOT`。
    - scenario_id: 0 表示全部，否则只加载该 id。
    - sub_folders: 可选子用户列表。
    - convert_map: 可选动作映射。
    """
    root = folder_path or WIDAR_ROOT
    # prefer processed .mat files if present
    proc_dir = os.path.join(root, 'processed')
    preloaded: List[Tuple[np.ndarray, int]] = []
    if os.path.isdir(proc_dir):
        if scenario_id == 0:
            # load all pairs
            for fname in sorted(os.listdir(proc_dir)):
                if fname.startswith('all_csi_') and fname.endswith('.mat'):
                    sid = fname[len('all_csi_'):-4]
                    csi_mat = os.path.join(proc_dir, fname)
                    action_mat = os.path.join(proc_dir, f'action_{sid}.mat')
                    preloaded.extend(WidarDataset._load_mat_pair(csi_mat, action_mat))
        else:
            csi_mat = os.path.join(proc_dir, f'all_csi_{scenario_id}.mat')
            action_mat = os.path.join(proc_dir, f'action_{scenario_id}.mat')
            preloaded = WidarDataset._load_mat_pair(csi_mat, action_mat)

    if len(preloaded) > 0:
        return WidarDataset(root, scenario_id=scenario_id, sub_folders=sub_folders, convert_map=convert_map, preloaded=preloaded)
    return WidarDataset(root, scenario_id=scenario_id, sub_folders=sub_folders, convert_map=convert_map)


def widar_dataloader(folder_path: Optional[str] = None, test_ratio: float = 0.2, scenario_id: int = 0, seed: int = 42):
    """返回 (dataset, train_idx, test_idx) 三元组。

    说明：为了保持简单并避免一次性把所有变长样本加载到内存，函数只负责收集样本路径并按索引切分。
    使用者可以基于返回的 `dataset`（支持 `__getitem__`/`iterate`）和索引构建自己的 DataLoader。
    """
    ds = widar_dataset(folder_path, scenario_id=scenario_id)
    n = len(ds)
    if n == 0:
        return ds, [], []
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    split = int(np.floor((1 - test_ratio) * n))
    train_idx = idx[:split].tolist()
    test_idx = idx[split:].tolist()
    return ds, train_idx, test_idx
