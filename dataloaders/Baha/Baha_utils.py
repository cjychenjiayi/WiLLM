# https://www.sciencedirect.com/science/article/pii/S2352340920314165?ref=cra_js_challenge&fr=RR-1
# https://github.com/lcsig/Dataset-for-Wi-Fi-based-human-activity-recognition-in-LOS-and-NLOS-indoor-environments
# https://data.mendeley.com/datasets/v38wjmz6f6/1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import tqdm
import torch
import pickle
import concurrent.futures


import scipy.io
import concurrent.futures

def parse_complex(value):
    value = value.replace("+-", "-").replace("i", "j")
    return complex(value)

def load_and_process_mat(path_info):
    try:
        mat = scipy.io.loadmat(path_info["path"])
        if 'data' not in mat:
            return None
        
        data_struct = mat['data']
        # Extract CSI
        csi_list = []
        # Accessing data_struct elements can be slow if done one by one in a loop
        # But data_struct is (N, 1) struct array.
        
        # Optimization: Check if we can vectorized access.
        # Scipy loadmat returns numpy structured arrays for structs.
        # data_struct[i, 0]['csi']
        
        for i in range(data_struct.shape[0]):
             # packet structure check
             packet = data_struct[i, 0]
             # check field existence
             # packet.dtype.names is consistent across array, so checking once is enough ideally, 
             # but let's stick to per-packet for robustness or check first packet.
             if 'csi' in packet.dtype.names:
                 csi = packet['csi'][0, 0] # (1, 3, 30)
                 csi_list.append(csi.reshape(3, 30))
        
        if not csi_list:
            return None
            
        csi_tensor = torch.tensor(np.array(csi_list), dtype=torch.cfloat) # (T, 3, 30)
        amplitude = torch.abs(csi_tensor)
        amplitude_db = 20 * torch.log10(amplitude + 1e-8)
        # (T, 3, 30) -> (T, 90) -> (90, T)
        data = amplitude_db.reshape(amplitude_db.shape[0], -1).permute(1, 0)
        
        path_info["data"] = data.float()
        # Return path_info with data
        return path_info
    except Exception as e:
        # print(f"Error {path_info['path']}: {e}")
        return None

def build_dataset(root_file="/"):
    csi_file_paths = []
    
    # ... file collection logic ... 
    # Logic to collect file paths should remain.
    
    for E in range(1, 4):
        env_dir = os.path.join(root_file, f"E{E}")
        if not os.path.exists(env_dir):
            continue
            
        try:
            files = os.listdir(env_dir)
        except OSError:
            continue
            
        for file in files:
            if not file.endswith(".mat"):
                continue
            
            # Parse filename: E1_S01_C01_A01_T01.mat
            try:
                parts = file.replace(".mat", "").split("_")
                if len(parts) >= 5:
                    e_val = int(parts[0][1:])
                    s_val = int(parts[1][1:])
                    c_val = int(parts[2][1:])
                    a_val = int(parts[3][1:])
                    t_val = int(parts[4][1:])
                    
                    full_path = os.path.join(env_dir, file)
                    csi_file_paths.append({
                        "E": e_val,
                        "S": s_val,
                        "C": c_val,
                        "A": a_val,
                        "T": t_val,
                        "path": full_path
                    })
            except Exception:
                pass

    # Process files
    processed_paths = []
    
    # Use ProcessPoolExecutor for CPU-bound tasks (parsing mat files and tensor ops)
    # Adjust max_workers as needed. 
    # If using ThreadPoolExecutor, GIL limits speed. ProcessPoolExecutor bypasses GIL.
    
    with tqdm.tqdm(total=len(csi_file_paths), desc="Data Preprocess Baha") as pbar:
        # Using ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Helper to just call function
            futures = [executor.submit(load_and_process_mat, p) for p in csi_file_paths]
            
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res is not None:
                    processed_paths.append(res)
                pbar.update(1)

    return processed_paths

def get_amplitude_sig(csi_file_path):

    csi_file = pd.read_csv(csi_file_path)
    timestamps = csi_file['timestamp_low'].values - csi_file['timestamp_low'].values[0]
    num_packets = len(csi_file)
    
    csi_columns = [col for col in csi_file.columns if col.startswith('csi')]
    col_indices = [(int(col.split('_')[2]) - 1, int(col.split('_')[3]) - 1) for col in csi_columns]
    num_streams = max(antenna for antenna, _ in col_indices) + 1
    num_subcarriers = max(subcarrier for _, subcarrier in col_indices) + 1

    result = {
        'timestamp': (timestamps / 1000000).tolist(),
        'streams': [np.zeros((num_packets, num_subcarriers)) for _ in range(num_streams)]
    }

    for col, (antenna, subcarrier) in zip(csi_columns, col_indices):
        complex_values = csi_file[col].apply(parse_complex).values
        amplitudes = np.abs(complex_values)
        db_values = np.where(amplitudes > 0, 20 * np.log10(amplitudes + 1e-8), 0)
        result['streams'][antenna][:, subcarrier] = db_values

    res_data = {
        'timestamp': result['timestamp'],
        **{f'stream{stream+1}': result['streams'][stream].tolist() for stream in range(num_streams)}
    }
    
    return res_data

def vis_2d(activity, save_path = None):
    plt.plot(activity['timestamp'], activity['stream1'].iloc[:, 0], linewidth=0.5)
    plt.title('Turning Activity of The First Subject at Transmitter Side')
    plt.xlabel('Time [s]')
    plt.ylabel('SNR [dB]')
    plt.ylim([1, 40]) 
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlim(int(activity['timestamp'].min()), int(activity['timestamp'].max()+0.01))
    plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()

def vis_3d(activity, save_path = None):
    x, y = np.meshgrid(activity['timestamp'], np.arange(1, 31))
    z = np.array(activity['stream1']).T
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, z, cmap='jet', edgecolor='none', shade=True)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Subcarrier Indices")
    ax.set_zlabel("SNR [dB]")
    fig.colorbar(surf)
    ax.view_init(elev=20., azim=60)
    plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()
    
def csv_to_pkl(root_file = "/"):
    csi_file_paths = []
    for E in range(1, 4):
        for S in range(1, 31):
            for C in range(1, 6):
                for A in range(1, 13):
                    for T in range(1, 20):
                        base_name = f"Environment_{E}/Subject_{S}/E{E}_S{str(S).zfill(2)}_C{str(C).zfill(2)}_A{str(A).zfill(2)}_T{str(T).zfill(2)}.csv"
                        csv_file_path = os.path.join(root_file, base_name)
                        if os.path.exists(csv_file_path):
                            csi_file_paths.append({"E":E, "S":S, "C":C, "A":A, "T":T, "path":csv_file_path})

    def process_and_save_file(path):
        activity = get_amplitude_sig(path["path"])
        with open(path["path"].replace("csv", "pkl"), "wb") as f:
            pickle.dump(activity, f)
            
    with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
        with tqdm.tqdm(total=len(csi_file_paths), desc="Data Preprocess Baha") as pbar:
            futures = [executor.submit(process_and_save_file, path) for path in csi_file_paths]
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)



def load_and_process_mat(path_info):
    try:
        mat = scipy.io.loadmat(path_info["path"])
        if 'data' not in mat:
            return None
        
        data_struct = mat['data']
        # Extract CSI
        csi_list = []
        for i in range(data_struct.shape[0]):
             # packet structure check
             packet = data_struct[i, 0]
             if 'csi' in packet.dtype.names:
                 csi = packet['csi'][0, 0] # (1, 3, 30)
                 csi_list.append(csi.reshape(3, 30))
        
        if not csi_list:
            return None
            
        csi_tensor = torch.tensor(np.array(csi_list), dtype=torch.cfloat) # (T, 3, 30)
        amplitude = torch.abs(csi_tensor)
        amplitude_db = 20 * torch.log10(amplitude + 1e-8)
        # (T, 3, 30) -> (T, 90) -> (90, T)
        data = amplitude_db.reshape(amplitude_db.shape[0], -1).permute(1, 0)
        
        path_info["data"] = data.float()
        return path_info
    except Exception as e:
        # print(f"Error {path_info['path']}: {e}")
        return None

def build_dataset(root_file="/"):
    csi_file_paths = []
    
    # Iterate over files logic (simplified/corrected)
    # The previous logic was:
    # E1/file.mat
    
    for E in range(1, 4):
        # Assuming folder structure is exactly: root_file/MAT/E{num}/file.mat 
        # But user input might be root_file/E{num}/file.mat if they point directly to MAT dir 
        # Or root_file points to parent of MAT.
        # User passed path ending in MAT in Baha_preprocess.py.
        # So root_file/E1 should be correct.
        
        env_dir = os.path.join(root_file, f"E{E}")
        if not os.path.exists(env_dir):
            continue
            
        try:
            files = os.listdir(env_dir)
        except OSError:
            continue
            
        for file in files:
            if not file.endswith(".mat"):
                continue
                
            # Parse filename: E1_S11_C01_A01_T01.mat
            try:
                parts = file.replace(".mat", "").split("_")
                if len(parts) >= 5:
                    e_val = int(parts[0][1:])
                    s_val = int(parts[1][1:])
                    c_val = int(parts[2][1:])
                    a_val = int(parts[3][1:])
                    t_val = int(parts[4][1:])
                    
                    full_path = os.path.join(env_dir, file)
                    csi_file_paths.append({
                        "E": e_val,
                        "S": s_val,
                        "C": c_val,
                        "A": a_val,
                        "T": t_val,
                        "path": full_path
                    })
            except Exception:
                pass


    processed_paths = []
    # Use ProcessPoolExecutor to bypass GIL for CPU bound tasks
    with tqdm.tqdm(total=len(csi_file_paths), desc="Data Preprocess Baha") as pbar:
        # max_workers=None defaults to cpu_count(), which is good.
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(load_and_process_mat, p) for p in csi_file_paths]
            
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res is not None:
                    processed_paths.append(res)
                pbar.update(1)

    return processed_paths

    # for path in tqdm.tqdm(csi_file_paths, desc="Data Preprocess Baha"):
    #     with open(path["path"], "rb") as f:
    #         activity = pickle.load(f)
    #         stream_data = [activity[f'stream{stream+1}'] for stream in range(len(activity) - 1)]
    #         activity = torch.tensor(stream_data).permute(0, 2, 1).flatten(start_dim=0, end_dim=1)
    #         path["data"] = activity
    return csi_file_paths
    
experiment_dict = {
    1: "Falling from sitting position",
    2: "Falling from standing position",
    3: "Walking",
    4: "Sit down and stand up",
    5: "Pick a pen from the ground"
}
actions_dict = {
    1: "sit still on a chair",
    2: "falling down",
    3: "lie down",
    4: "stand still",
    5: "falling down",
    6: "walking from transmitter to receiver",
    7: "turning",
    8: "walking from receiver to transmitter",
    9: "turning",
    10: "standing up",
    11: "sitting down",
    12: "pick a pen from the ground"
}

# 运行 main 函数
if __name__ == "__main__":
    root_path = "/mntcephfs/lab_data/guangxuzhu/chenjiayi/wifi/baha/"
    # csv_to_pkl(root_path)
    result = build_dataset(root_path)
    print(result)
