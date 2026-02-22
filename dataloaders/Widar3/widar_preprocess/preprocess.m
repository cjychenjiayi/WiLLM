clc; clear; close all;

%% ================= 基本设置 =================
save_dir = '/home/chenjiayi/workspace/willm/wifi_data/widar3_raw/processed';
main_path = '/home/chenjiayi/workspace/willm/wifi_data/widar3_raw/20181109';
sub_folders = {'user1','user2','user3'};
save_id = 1;

% 20181109 动作映射
convert = [1,2,3,4,10,11];

% 只处理前 N 个文件（正式跑改成 Inf）
MAX_FILES = 5;

% CSI 最小长度
MIN_LEN = 50;

% 确保保存目录存在
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

if exist('db','file') ~= 2
    function out = db(x)
        out = 20*log10(abs(x) + eps); 
    end
end

%% ================= 添加路径 =================
script_dir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(script_dir,'preprocess_tool')));
addpath(fullfile(script_dir,'linux-80211n-csitool'));

%% ================= 主循环 =================
for ii = 1:length(sub_folders)

    sub_folder = sub_folders{ii};
    path = fullfile(main_path, sub_folder);
    csi_files = dir(fullfile(path, '*.dat'));

    % ===== 强制按 Windows 字典序排序 =====
    file_names = {csi_files.name};
    [~, idx] = sort(lower(file_names));   % 不区分大小写排序
    csi_files = csi_files(idx);
    
    total_files = length(csi_files);

    fprintf('\n%s: %d files found\n', sub_folder, total_files);

    if total_files == 0
        continue;
    end

    nFiles = min(total_files, MAX_FILES);

    % 预分配
    csi_result = cell(1, nFiles);
    action = zeros(1, nFiles);

    csi_result_idx = 1;

    for i1 = 1:nFiles

        fname = csi_files(i1).name;
        csi_file = fullfile(csi_files(i1).folder, fname);

        % 解析文件名
        nums = sscanf(fname, '%*[^-]-%d-%d-%d-%d-r%d');
        if numel(nums) < 5
            fprintf('跳过（文件名异常）: %s\n', fname);
            continue;
        end

        a = nums(1);

        if a < 1 || a > length(convert)
            fprintf('跳过（动作编号异常）: %s\n', fname);
            continue;
        end


        csi = read_bf_file(csi_file);
        csi = remove_empty_csi(csi);

        if length(csi) < MIN_LEN
            fprintf('跳过（CSI太短）: %s\n', fname);
            continue;
        end

        csi = scale_csi_5300(csi,"5300");
        amp = get_amplitude(csi);
        amp = amp_hampel_v2(amp,20);
        amp = amp_DWT(amp);
        amp = reshape(amp, size(amp,1), []);
        amp = reshape(amp, size(amp,1),[]);
        

        % 保存结果
        csi_result{csi_result_idx} = amp;
        action(csi_result_idx) = convert(a);
        csi_result_idx = csi_result_idx + 1;

        fprintf('Processed %d/%d: %s\n', i1, nFiles, fname);
    end

    % 裁剪无效部分
    csi_result = csi_result(1:csi_result_idx-1);
    action = action(1:csi_result_idx-1);

    % 保存（关键：-v7.3 让 Python h5py 能读取）
    save(fullfile(save_dir, sprintf('all_csi_%d.mat', save_id)), ...
         'csi_result', '-mat');

    save(fullfile(save_dir, sprintf('action_%d.mat', save_id)), ...
         'action', '-mat');

    fprintf('Saved user %s (id=%d)\n', sub_folder, save_id);

    save_id = save_id + 1;
    break
end

fprintf('\nAll done.\n');