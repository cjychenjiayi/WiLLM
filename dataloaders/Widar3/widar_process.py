# preprocess.py

import os
import re
import numpy as np
from scipy.io import savemat
import hdf5storage
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from csi5300_utils import scale_csi_5300
from csiread_utils import read_csi_file
from csiprocess_utils import (
    remove_empty_csi,
    get_amplitude,
    amp_hampel,
    amp_DWT
)
import h5py

def save_mat_files(save_dir, save_id, csi_result, action):
    """
    等价：
        save(all_csi_x.mat, 'csi_result', '-v7.3')
        save(action_x.mat, 'action', '-v7.3')
    """

    csi_path = os.path.join(save_dir, f"all_csi_{save_id}.mat")
    action_path = os.path.join(save_dir, f"action_{save_id}.mat")

    # 删除旧文件（防止格式冲突）
    if os.path.exists(csi_path):
        os.remove(csi_path)
    if os.path.exists(action_path):
        os.remove(action_path)

    # === 保存 csi_result ===
    with h5py.File(csi_path, "w") as f:
        grp = f.create_group("csi_result")
        for i, amp in enumerate(csi_result):
            grp.create_dataset(str(i), data=amp.astype(np.float32, copy=False))

    # === 保存 action ===
    with h5py.File(action_path, "w") as f:
        f.create_dataset("action", data=np.array(action, dtype=np.int32))

    print(f"Saved save_id={save_id}")
    
def parse_filename(filename):

    pattern = r'(\w+)-(\d+)-(\d+)-(\d+)-(\d+)-r(\d+)\.dat'
    match = re.match(pattern, filename)

    if not match:
        return None

    g = match.groups()
    id_number = int(re.findall(r'\d+', g[0])[0])

    return {
        "id": id_number,
        "a": int(g[1]),
        "b": int(g[2]),
        "c": int(g[3]),
        "d": int(g[4]),
        "Rx": int(g[5])
    }


def map_action(original_label, convert_map):
    if 1 <= original_label <= len(convert_map):
        return convert_map[original_label - 1]
    return np.nan

def process_single_file(file_path):

    csi = read_csi_file(file_path)
    csi = remove_empty_csi(csi)

    if len(csi) < 50:
        return None

    csi = scale_csi_5300(csi)

    amp = get_amplitude(csi)
    amp = amp_hampel(amp, 20)
    amp = amp_DWT(amp)

    PA = amp.shape[0]
    amp = amp.reshape(PA, -1)

    return amp



def worker(args):
    folder, file, convert_map = args

    parsed = parse_filename(file)
    if parsed is None:
        return None

    file_path = os.path.join(folder, file)

    try:
        amp = process_single_file(file_path)
    except Exception:
        return None

    if amp is None:
        return None

    act = map_action(parsed["a"], convert_map)

    return amp, act
        
def process_dataset(main_path, sub_folders, convert_map, save_dir, save_id):

    for sub in sub_folders:
        folder = os.path.join(main_path, sub)

        if not os.path.isdir(folder):
            print("Path does not exist:", folder)
            continue

        files = sorted([f for f in os.listdir(folder) if f.endswith(".dat")])
        print(len(files), "files in", folder)

        csi_result = []
        action = []
        cnt = 0
        with ProcessPoolExecutor(max_workers=150) as executor:
            args = [(folder, file, convert_map) for file in files]

            for result in tqdm(executor.map(worker, args), total=len(files)):
                if result is None:
                    continue
                amp, act = result
                csi_result.append(amp)
                action.append(act)
                cnt += 1
                # if cnt > 100:
                #     break

        csi_result = np.array(csi_result, dtype=object)
        action = np.array(action)
        # import pdb; pdb.set_trace()
        # # tmp = np.array(csi_result, dtype=object)
        # # print("Size GB:", tmp.nbytes / 1024**3)
        # # import pdb; pdb.set_trace()
        # # MATLAB：每个 sub_folder 都保存一次
        save_mat_files(
            save_dir,
            save_id,
            csi_result,
            action
        )
        # savemat(os.path.join(save_dir, f"all_csi_{save_id}.mat"),
        #         {"csi_result": csi_result})

        # savemat(os.path.join(save_dir, f"action_{save_id}.mat"),
        #         {"action": action})
        
        
        # hdf5storage.savemat(
        #     os.path.join(save_dir, f"all_csi_{save_id}.mat"),
        #     {"csi_result": np.array(csi_result, dtype=object)},
        #     format='7.3'
        # )

        # hdf5storage.savemat(
        #     os.path.join(save_dir, f"action_{save_id}.mat"),
        #     {"action": np.array(action)},
        #     format='7.3'
        # )

        print("Save finish save_id =", save_id)

        # MATLAB：每个 sub_folder 保存后 save_id++ 
        save_id += 1
    return save_id

def main():

    root_path = "/home/chenjiayi/workspace/willm/wifi_data/widar3_raw"
    save_dir = os.path.join(root_path, "processed")
    os.makedirs(save_dir, exist_ok=True)

    datasets = [
        # 20181109
        {"folder_name": "20181109", "sub_folders": ["user1", "user2", "user3"], "convert": [1, 2, 3, 4, 10, 11]},
        # 20181112
        {"folder_name": "20181112", "sub_folders": ["user1", "user2"], "convert": [13, 14, 15, 16, 17, 18, 19, 20, 21, 22]},
        # 20181115
        {"folder_name": "20181115", "sub_folders": ["user1"], "convert": [1, 2, 3, 12, 10, 11]},
        # 20181116
        {"folder_name": "20181116", "sub_folders": ["user1"], "convert": [13, 14, 15, 16, 17, 18, 19, 20, 21, 22]},
        # 20181117
        {"folder_name": "20181117", "sub_folders": ["user4"], "convert": [1, 2, 3, 12, 10, 11]},
        # 20181118
        {"folder_name": "20181118", "sub_folders": ["user2", "user3"], "convert": [1, 2, 3, 12, 10, 11]},
        # 20181121
        {"folder_name": "20181121", "sub_folders": ["user1", "user2", "user3"], "convert": [4, 6, 9, 5, 8, 7]},
        # 20181127
        {"folder_name": "20181127", "sub_folders": ["user2", "user5"], "convert": [4, 6, 9, 5, 8, 7]},
        # 20181128
        {"folder_name": "20181128", "sub_folders": ["user6"], "convert": [1, 2, 3, 6, 9, 5]},
        # 20181130_user5_10_11
        {"folder_name": "20181130_user5_10_11", "sub_folders": ["user5", "user10", "user11"], "convert": [1, 2, 3, 4, 6, 9, 5, 8, 7]},
        # 20181130_user12_13_14
        {"folder_name": "20181130_user12_13_14", "sub_folders": ["user12", "user13", "user14"], "convert": [1, 2, 3, 4, 6, 9, 5, 8, 7]},
        # 20181130_user15_16_17
        {"folder_name": "20181130_user15_16_17", "sub_folders": ["user15", "user16", "user17"], "convert": [1, 2, 3, 4, 6, 9, 5, 8, 7]},
        # 20181204
        {"folder_name": "20181204", "sub_folders": ["user1"], "convert": [1, 2, 3, 4, 6, 9, 5, 8, 7]},
        # 20181205 user2
        {"folder_name": "20181205", "sub_folders": ["user2"], "convert": [6, 9, 5, 8, 7]},
        # 20181205 user3
        {"folder_name": "20181205", "sub_folders": ["user3"], "convert": [4, 6, 9, 5, 8, 7]},
        # 20181208 user2
        {"folder_name": "20181208", "sub_folders": ["user2"], "convert": [1, 2, 3, 4]},
        # 20181208 user3
        {"folder_name": "20181208", "sub_folders": ["user3"], "convert": [1, 2, 3]},
        # 20181209 user2
        {"folder_name": "20181209", "sub_folders": ["user2"], "convert": [1]},
        # 20181209 user6
        {"folder_name": "20181209", "sub_folders": ["user6"], "convert": [1, 2, 3, 4, 6, 9]},
        # 20181211
        {"folder_name": "20181211", "sub_folders": ["user3", "user7", "user8", "user9"], "convert": [1, 2, 3, 4, 6, 9]},
    ]
    
    save_id = 1

    for ds in tqdm(datasets, desc = "Total Progress"):
        main_path = os.path.join(root_path, ds["folder_name"])
        save_id = process_dataset(
            main_path,
            ds["sub_folders"],
            ds["convert"],
            save_dir,
            save_id
        )
        print("Finish dataset:", ds["folder_name"], "next save_id =", save_id)
    
if __name__ == "__main__":
    main()