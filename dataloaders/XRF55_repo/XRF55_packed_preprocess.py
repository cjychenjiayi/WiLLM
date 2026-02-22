import os
import glob
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- CONFIGURATION (Hardcoded for simplicity) ---
# Where the raw .npy files are (DATA_ROOT/train, DATA_ROOT/test)
# User specified: /home/chenjiayi/workspace/willm/wifi_data/xrf55_all/xrf55
DATA_ROOT = "/home/chenjiayi/workspace/willm/wifi_data/xrf55_all/xrf55"  

# Where to save the packed .mmap file
# User specified: /home/chenjiayi/workspace/willm/wifi_data/xrf55_all/
OUTPUT_DIR = "/home/chenjiayi/workspace/willm/wifi_data/xrf55_all"

NUM_WORKERS = 16 

def load_file_check_shape(args):
    """
    Worker function: reads ONE file, returns (index, data, label)
    """
    idx, file_path, label = args
    try:
        raw = np.load(file_path)
        # Expected shape (270, 1000)
        if raw.shape == (270, 1000):
            return idx, raw.astype(np.float32), label
            
        # Try simplistic fix if just dims are wrong (e.g. (1, 270, 1000))
        if raw.size == 270000:
             return idx, raw.reshape(270, 1000).astype(np.float32), label
             
        return None # Bad shape
    except:
        return None

def main():
    if not os.path.exists(DATA_ROOT):
        print(f"Error: Raw data folder not found: {DATA_ROOT}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"I: Scanning files in {DATA_ROOT}...")

    # 1. Collect file paths recursively
    # Structure: .../SceneX/SceneX/WiFi/Subject_Action_Instance.npy
    # We want any .npy file inside a WiFi folder
    all_files = [] 
    
    # Walk through DATA_ROOT
    for root, dirs, files in os.walk(DATA_ROOT):
        # We are only interested if we are in a 'WiFi' directory
        if os.path.basename(root) == 'WiFi':
            print(f"I: Found WiFi folder: {root}")
            for f in files:
                if f.endswith('.npy'):
                    # Filename format: Subject_Action_Instance.npy (e.g., 01_55_05.npy)
                    # We need the middle number (Action)
                    try:
                        parts = f.replace('.npy', '').split('_')
                        if len(parts) >= 2:
                            # Label is the second part (1-based index 1..55)
                            label_1based = int(parts[1])
                            # Convert to 0-based index 0..54
                            label_0based = label_1based - 1
                            
                            full_path = os.path.join(root, f)
                            all_files.append((full_path, label_0based))
                    except Exception as e:
                        # print(f"Skipping {f}: {e}")
                        pass

    total = len(all_files)
    print(f"I: Found {total} valid .npy files.")
    
    if total == 0:
        return

    # 2. Setup Memmap
    mmap_path = os.path.join(OUTPUT_DIR, "xrf55_data.mmap")
    labels_path = os.path.join(OUTPUT_DIR, "xrf55_labels.npy")
    
    # Create empty mmap file on disk
    print(f"I: Allocating memmap {mmap_path} (~{total * 270 * 1000 * 4 / 1e9:.2f} GB)...")
    fp = np.memmap(mmap_path, dtype='float32', mode='w+', shape=(total, 270, 1000))
    
    # Array for labels
    labels_arr = np.zeros(total, dtype=np.int64)
    
    # 3. Parallel Load & Write
    print(f"I: Starting multiprocessing with {NUM_WORKERS} workers...")
    
    # Prepare arguments (maintain index so we write to correct slot)
    tasks = []
    for i, (fpath, lbl) in enumerate(all_files):
        # Pass the full path
        tasks.append((i, fpath, lbl))
        
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(load_file_check_shape, t) for t in tasks]
        
        for future in tqdm(as_completed(futures), total=total, desc="Packing"):
            res = future.result()
            if res:
                idx, data, lbl = res
                fp[idx] = data
                labels_arr[idx] = lbl
                
    # Persist
    fp.flush()
    np.save(labels_path, labels_arr)
    
    print("I: Success. Packed data saved.")
    print(f"   Data:   {mmap_path}")
    print(f"   Labels: {labels_path}")

if __name__ == "__main__":
    main()
