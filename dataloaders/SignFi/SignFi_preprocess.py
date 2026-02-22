import numpy as np
import scipy.io as sio
import os
import argparse

def data_preprocess(folder_path):
    tot_label = []
    tot_csi_abs = []
    tot_csi_ang = []
    
    print(f"Processing .mat files in {folder_path}...")
    
    files_processed = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".mat"):
            file_path = os.path.join(folder_path, filename)
            try:
                data = sio.loadmat(file_path)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
                
            show_key = [key for key in data.keys() if (not key.startswith("__")) and ("u" not in key)]
            label = []
            csi = []
            for key in show_key:
                if key.startswith("label"):
                    label.append(data[key])
                if key.startswith("csi"):
                    csi.append(data[key])
            if len(label) == 0 or len(csi) == 0:
                continue
           
            print(f"Processing {filename}...")
            label = np.concatenate(label, axis=-1)
            csi = np.concatenate(csi, axis=-1)
            time_dim, subcarry, num_att, repeat = csi.shape
            csi = csi.reshape(repeat, subcarry*num_att, time_dim)
            
            # Compute abs and ang here to save loading time later
            # Convert to float32 to save space
            csi_abs = np.abs(csi).astype(np.float32)
            csi_ang = np.angle(csi).astype(np.float32)

            tot_label.append(label)
            tot_csi_abs.append(csi_abs)
            tot_csi_ang.append(csi_ang)
            files_processed += 1

    if files_processed == 0:
        print("No .mat files processed.")
        return

    print("Concatenating data...")
    tot_label = np.concatenate(tot_label, axis=0)
    tot_csi_abs = np.concatenate(tot_csi_abs, axis=0)
    tot_csi_ang = np.concatenate(tot_csi_ang, axis=0)

    # Compute normalization statistics
    print("Computing normalization statistics...")
    # Compute mean and std across all samples (axis 0), subcarriers (axis 1), and time steps (axis 2)
    # Using keepdims=True to maintain broadcastable shape if needed, but usually for global norm we want scalars or per-feature.
    # The original code did: axis=(0, 1, 2)
    mean_abs = np.mean(tot_csi_abs, axis=(0, 1, 2), keepdims=True)
    std_abs = np.std(tot_csi_abs, axis=(0, 1, 2), keepdims=True)
    mean_ang = np.mean(tot_csi_ang, axis=(0, 1, 2), keepdims=True)
    std_ang = np.std(tot_csi_ang, axis=(0, 1, 2), keepdims=True)
    
    save_path = os.path.join(folder_path, "all_processed.npz")
    print(f"Saving processed data to {save_path}...")
    np.savez_compressed(
        save_path, 
        label=tot_label, 
        csi_abs=tot_csi_abs, 
        csi_ang=tot_csi_ang,
        mean_abs=mean_abs,
        std_abs=std_abs,
        mean_ang=mean_ang,
        std_ang=std_ang
    )
    print("Pre-processing completed with statistics saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess SignFi data")
    parser.add_argument("--root_folder", type=str, default="/home/chenjiayi/workspace/willm/wifi_data/SignFi", help="Root folder containing .mat files")
    args = parser.parse_args()
    
    data_preprocess(args.root_folder)
