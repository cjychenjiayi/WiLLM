import numpy as np
import scipy.io
import os
import tqdm
import pickle
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import traceback

def inter_csi(csi_list):
    """Interpolates CSI samples to consistent shape."""
    try:
        if not csi_list: return None
        
        # Robust shape extraction - handle scalars or weird arrays
        ntx = []
        nrx = []
        for d in csi_list:
            if hasattr(d, 'shape') and len(d.shape) >= 2:
                ntx.append(d.shape[0])
                nrx.append(d.shape[1])
            else:
                return None # Invalid csi item

        if not ntx: return None

        # Check consistency (>90%)
        most_common_ntx = Counter(ntx).most_common(1)[0][0]
        most_common_nrx = Counter(nrx).most_common(1)[0][0]
        
        # Debugging loose condition
        if ntx.count(most_common_ntx) / len(csi_list) < 0.9: return None
        if nrx.count(most_common_nrx) / len(csi_list) < 0.9: return None
        
        # Get common subcarrier count
        # Filter for shapes that match TX/RX
        valid_sub_shapes = [d.shape[2] for d in csi_list if d.shape[0] == most_common_ntx and d.shape[1] == most_common_nrx and len(d.shape) > 2]
        
        if not valid_sub_shapes:
            # print("No valid subcarrier shapes found")
            return None

        n_carriers = Counter(valid_sub_shapes).most_common(1)[0][0]
        
        # Align all shapes
        aligned_csi = []
        for d in csi_list:
            target_shape = (most_common_ntx, most_common_nrx, n_carriers)
            if d.shape == target_shape:
                aligned_csi.append(d)
            else:
                aligned_csi.append(np.full(target_shape, np.nan))
        
        all_csi = np.array(aligned_csi) # (T, Ntx, Nrx, Sub)
        T, Ntx, Nrx, S = all_csi.shape
        
        # Interpolate NaNs
        final_csi = np.abs(all_csi) # Use magnitude for interpolation basis
        
        # Vectorized interpolation is hard with varying NaNs
        for s in range(S):
            for t in range(Ntx):
                for r in range(Nrx):
                    y = final_csi[:, t, r, s]
                    nans = np.isnan(y)
                    if np.all(nans): continue
                    if np.any(nans):
                        x = np.arange(T)
                        final_csi[nans, t, r, s] = np.interp(x[nans], x[~nans], y[~nans])
        
        return final_csi # Returns magnitude already interpolated
    except Exception:
        # traceback.print_exc()
        return None

def process_file_task(file_path):
    """Worker task to process a single mat file."""
    try:
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        # Expecting at least 2 parts typically for activity? 
        # But consistent with user code: 3 parts
        if len(parts) < 2: 
             return None
             
        # Activity is usually index 1
        activity = parts[1]

        try:
            mat = scipy.io.loadmat(file_path)
        except Exception:
            return None

        if "csi_trace" not in mat or mat["csi_trace"].size == 0:
             return None
        
        # Robust extraction
        csi_frames = []
        trace_data = mat["csi_trace"]
        trace_flat = trace_data.reshape(-1) # Flatten struct array

        for frame in trace_flat:
            try:
                # Check for 'csi' field
                if 'csi' in frame.dtype.names:
                    val = frame['csi']
                    # Handle object arrays or nested cells
                    if val.size > 0:
                        # If it's a 1x1 object array (common in MATLAB cells)
                        if val.dtype == 'O' and val.shape == (1, 1):
                             # Extract inner content
                             inner = val[0, 0]
                             csi_frames.append(inner)
                        elif val.size == 1 and val.dtype.names is None: 
                             # Maybe direct array?
                             csi_frames.append(val)
                        else:
                             # Just append whatever, let inter_csi filter bad shapes
                             csi_frames.append(val)
            except: 
                continue
            
        if not csi_frames: 
             return None

        # Preprocess/Interpolate
        processed_mag = inter_csi(csi_frames)
        if processed_mag is None: 
             return None
        
        # Filter Length [92, 110)
        # Directly truncate here to save memory passing back to main process
        T = processed_mag.shape[0]
        if 92 <= T < 110:
            return {
                'activity': activity,
                'csi_magnitude': processed_mag[:92], # Truncate to 92
                'file': filename
            }
        else:
             # Debug length issue - maybe most files fail here?
             # print(f"Length mismatch {filename}: {T}")
             pass
             
        return None
    except Exception:
        # traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Brinkle Preprocessing (Multi-threaded)")
    parser.add_argument("--root_path", type=str, default="/home/chenjiayi/workspace/willm/wifi_data/Brinkle", help="Data root")
    parser.add_argument("--output", type=str, default="dataset.pkl")
    # Reduce default workers slightly to avoid potential IO/Memory thrashing if that was an issue
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Parallel workers")
    args = parser.parse_args()

    if not os.path.exists(args.root_path):
        print(f"Error: {args.root_path} not found.")
        return

    # Find files
    mat_files = []
    print(f"Scanning {args.root_path}...")
    for root, _, files in os.walk(args.root_path):
        for f in files:
            if f.endswith('.mat'):
                mat_files.append(os.path.join(root, f))
    
    print(f"Found {len(mat_files)} files. Processing with {args.workers} workers...")

    valid_data = []
    
    # Parallel Processing
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_file_task, f): f for f in mat_files}
        
        for future in tqdm.tqdm(as_completed(futures), total=len(mat_files), desc="Processing"):
            res = future.result()
            if res: 
                valid_data.append(res)
    
    print(f"Valid samples: {len(valid_data)}")
    
    if not valid_data:
        print("No valid data generated. Check filtered length criteria [92, 110).")
        return

    # Generate Labels
    classes = sorted(list(set(d['activity'] for d in valid_data)))
    label_dict = {cls: i for i, cls in enumerate(classes)}
    print(f"Classes: {label_dict}")

    # Save
    save_path = os.path.join(args.root_path, args.output)
    print(f"Saving to {save_path}...")
    with open(save_path, 'wb') as f:
        pickle.dump({'data_list': valid_data, 'label_dict': label_dict}, f)
    print("Preprocess pipeline complete.")

if __name__ == "__main__":
    main()

