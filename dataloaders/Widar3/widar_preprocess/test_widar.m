% 指定文件路径
file_path = "F:\washed_widar\all_csi_1.pkl";
pickle = py.importlib.import_module('pickle');
file = py.open(file_path, 'rb');
csi_result_idx = pickle.load(file);
file.close();
csi_result_idx_mat = cell(csi_result_idx);
