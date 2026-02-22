clc;clear;close all;
addpath(genpath('D:\Widar_preprocess\preprocess_tool'));
addpath('D:\Widar_preprocess/linux-80211n-csitool');

%%
save_dir = "E:\widar_washed_denoised\";
save_id = 1;

%% actions
% 1: Push&Pull; 
% 2: Sweep; 
% 3: Clap; 
% 4: Slide; 
% 5: Draw N(H);
% 6: Draw-O(H);
% 7: Draw Rectangle
% 8: Draw Triangle
% 9: Draw-Zigzag(H);
% 10: Draw-Zigzag(V)
% 11:Draw-N(V)
% 12:Draw-O(V)
% 13:Draw 1
% 14:Draw 2
% 15:Draw 3
% 16:Draw 4
% 17:Draw 5
% 18:Draw 6
% 19:Draw 7
% 20:Draw 8
% 21:Draw 9
% 22:Draw 10

%% 20181109  
% 1: Push&Pull; 2: Sweep; 3: Clap; 4:Slide; 5: Draw-Zigzag(Vertical); 6:Draw-N(Vertical);
convert = [1,2,3,4,10,11];

main_path = "F:\Widar3.0ReleaseData\CSI\20181109\";
sub_folders = ["user1","user2","user3"];
for sub_folder = sub_folders
    action = [];
    csi_result = {};
    csi_result_idx = 1;

    path = fullfile(main_path, sub_folder);
    csi_files = dir(fullfile(path, '*.dat'));
    fprintf("%d files in %s\n",length(csi_files),path);
    for i1 = 1:length(csi_files)
        csi_file = fullfile(csi_files(i1).folder,csi_files(i1).name);
        if mod(i1, 1000) == 0
            fprintf('Processing the %d-th file: %s\n', i1, csi_file);
        end
        % 解包名字
        pattern = '(\w+)-(\d+)-(\d+)-(\d+)-(\d+)-r(\d+)\.dat';
        tokens = regexp(csi_files(i1).name, pattern, 'tokens');
        if isempty(tokens)
            error('Filename does not match expected format: %s', filename);
        end
        tokens = tokens{1};
        parsed_data.id = str2double(tokens{1}(end)); % user1 -> 1
        parsed_data.a  = str2double(tokens{2});
        parsed_data.b  = str2double(tokens{3});
        parsed_data.c  = str2double(tokens{4});
        parsed_data.d  = str2double(tokens{5});
        parsed_data.Rx = str2double(tokens{6});

        %% amp
        csi = read_bf_file(csi_file);
        csi = remove_empty_csi(csi);
        if length(csi) < 50
            disp(['跳过文件: ', csi_file, '，length(csi) = ', num2str(length(csi))]);
            continue; 
        end
        csi = scale_csi_5300(csi,"5300");
        amp = get_amplitude(csi);
        amp = amp_hampel_v2(amp,20);
        amp = amp_DWT(amp);
        amp = reshape(amp, size(amp,1),[]);
        csi_result{csi_result_idx} = amp;
        csi_result_idx = csi_result_idx+1; 
        action = [action convert(parsed_data.a)];

    end
    save_filename = sprintf("all_csi_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'csi_result', '-v7.3');  % 保存为MAT文件
    fprintf("保存成功: %s\n", save_path);
    save_filename = sprintf("action_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'action', '-v7.3'); 
    fprintf("保存成功: %s\n", save_path);
    save_id = save_id+1;
end

%%  20181112 
% 1: Draw-1; 2: Draw-2; 3: Draw-3; 4:Draw-4; 5: Draw-5; 
% 6: Draw-6; 7:Draw-7; 8: Draw-8; 9: Draw-9; 0:Draw-0;
convert = [13,14,15,16,17,18,19,20,21,22];
main_path = "F:\Widar3.0ReleaseData\CSI\20181112\";
sub_folders = ["user1","user2"];

for sub_folder = sub_folders
        action = [];
    csi_result = {};
    csi_result_idx = 1;

    path = fullfile(main_path, sub_folder);
    csi_files = dir(fullfile(path, '*.dat'));
    fprintf("%d files in %s\n",length(csi_files),path);
    for i1 = 1:length(csi_files)
        csi_file = fullfile(csi_files(i1).folder,csi_files(i1).name);
        if mod(i1, 1000) == 0
            fprintf('Processing the %d-th file: %s\n', i1, csi_file);
        end
        % 解包名字
        pattern = '(\w+)-(\d+)-(\d+)-(\d+)-(\d+)-r(\d+)\.dat';
        tokens = regexp(csi_files(i1).name, pattern, 'tokens');
        if isempty(tokens)
            error('Filename does not match expected format: %s', filename);
        end
        tokens = tokens{1};
        parsed_data.id = str2double(tokens{1}(end)); % user1 -> 1
        parsed_data.a  = str2double(tokens{2});
        parsed_data.b  = str2double(tokens{3});
        parsed_data.c  = str2double(tokens{4});
        parsed_data.d  = str2double(tokens{5});
        parsed_data.Rx = str2double(tokens{6});

        %% todo 
        action = [action convert(parsed_data.a)];

        %% amp
        csi = read_bf_file(csi_file);
        csi = remove_empty_csi(csi);
        csi = scale_csi_5300(csi,"5300");
        amp = amp_DWT(amp_hampel_v2(get_amplitude(csi),20));
        amp = reshape(amp, size(amp,1),[]);
        csi_result{csi_result_idx} = amp;
        csi_result_idx = csi_result_idx+1;

    end
            save_filename = sprintf("all_csi_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'csi_result', '-v7.3');  % 保存为MAT文件
    fprintf("保存成功: %s\n", save_path);
    save_filename = sprintf("action_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'action', '-v7.3'); 
    fprintf("保存成功: %s\n", save_path);
    save_id = save_id+1;
end

%% 20181115
% 1: Push&Pull; 2: Sweep; 3: Clap; 4:Draw-O(Vertical); 5: Draw Zigzag(Vertical); 
% 6: Draw-N(Vertical);
convert = [1,2,3,12,10,11];
main_path = "F:\Widar3.0ReleaseData\CSI\20181115\";
sub_folders = ["user1"];

for sub_folder = sub_folders
        action = [];
    csi_result = {};
    csi_result_idx = 1;

    path = fullfile(main_path, sub_folder);
    csi_files = dir(fullfile(path, '*.dat'));
    fprintf("%d files in %s\n",length(csi_files),path);
    for i1 = 1:length(csi_files)
        csi_file = fullfile(csi_files(i1).folder,csi_files(i1).name);
        if mod(i1, 1000) == 0
            fprintf('Processing the %d-th file: %s\n', i1, csi_file);
        end
        % 解包名字
        pattern = '(\w+)-(\d+)-(\d+)-(\d+)-(\d+)-r(\d+)\.dat';
        tokens = regexp(csi_files(i1).name, pattern, 'tokens');
        if isempty(tokens)
            error('Filename does not match expected format: %s', filename);
        end
        tokens = tokens{1};
        parsed_data.id = str2double(tokens{1}(end)); % user1 -> 1
        parsed_data.a  = str2double(tokens{2});
        parsed_data.b  = str2double(tokens{3});
        parsed_data.c  = str2double(tokens{4});
        parsed_data.d  = str2double(tokens{5});
        parsed_data.Rx = str2double(tokens{6});

        %% todo 
        action = [action convert(parsed_data.a)];

        %% amp
        csi = read_bf_file(csi_file);
        csi = remove_empty_csi(csi);
        csi = scale_csi_5300(csi,"5300");
        amp = amp_DWT(amp_hampel_v2(get_amplitude(csi),20));
        amp = reshape(amp, size(amp,1),[]);
        csi_result{csi_result_idx} = amp;
        csi_result_idx = csi_result_idx+1;

    end
            save_filename = sprintf("all_csi_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'csi_result', '-v7.3');  % 保存为MAT文件
    fprintf("保存成功: %s\n", save_path);
    save_filename = sprintf("action_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'action', '-v7.3'); 
    fprintf("保存成功: %s\n", save_path);
    save_id = save_id+1;
end

%% 20181116
% 1: Draw-1; 2: Draw-2; 3: Draw-3; 4:Draw-4; 5: Draw-5; 
% 6: Draw-6; 7:Draw-7; 8: Draw-8; 9: Draw-9; 0:Draw-0;
convert = [13,14,15,16,17,18,19,20,21,22];
main_path = "F:\Widar3.0ReleaseData\CSI\20181116\";
sub_folders = ["user1"];

for sub_folder = sub_folders
        action = [];
    csi_result = {};
    csi_result_idx = 1;

    path = fullfile(main_path, sub_folder);
    csi_files = dir(fullfile(path, '*.dat'));
    fprintf("%d files in %s\n",length(csi_files),path);
    for i1 = 1:length(csi_files)
        csi_file = fullfile(csi_files(i1).folder,csi_files(i1).name);
        if mod(i1, 1000) == 0
            fprintf('Processing the %d-th file: %s\n', i1, csi_file);
        end
        % 解包名字
        pattern = '(\w+)-(\d+)-(\d+)-(\d+)-(\d+)-r(\d+)\.dat';
        tokens = regexp(csi_files(i1).name, pattern, 'tokens');
        if isempty(tokens)
            error('Filename does not match expected format: %s', filename);
        end
        tokens = tokens{1};
        parsed_data.id = str2double(tokens{1}(end)); % user1 -> 1
        parsed_data.a  = str2double(tokens{2});
        parsed_data.b  = str2double(tokens{3});
        parsed_data.c  = str2double(tokens{4});
        parsed_data.d  = str2double(tokens{5});
        parsed_data.Rx = str2double(tokens{6});

        %% todo 
        action = [action convert(parsed_data.a)];

        %% amp
        csi = read_bf_file(csi_file);
        csi = remove_empty_csi(csi);
        csi = scale_csi_5300(csi,"5300");
        amp = amp_DWT(amp_hampel_v2(get_amplitude(csi),20));
        amp = reshape(amp, size(amp,1),[]);
        csi_result{csi_result_idx} = amp;
        csi_result_idx = csi_result_idx+1;

    end
            save_filename = sprintf("all_csi_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'csi_result', '-v7.3');  % 保存为MAT文件
    fprintf("保存成功: %s\n", save_path);
    save_filename = sprintf("action_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'action', '-v7.3'); 
    fprintf("保存成功: %s\n", save_path);
    save_id = save_id+1;
end

%% 20181117
% 1: Push&Pull; 2: Sweep; 3: Clap; 4:Draw-O(Vertical); 5: Draw Zigzag(Vertical); 
% 6: Draw-N(Vertical);
convert = [1,2,3,12,10,11];
main_path = "F:\Widar3.0ReleaseData\CSI\20181117\";
sub_folders = ["user4"];

for sub_folder = sub_folders
        action = [];
    csi_result = {};
    csi_result_idx = 1;

    path = fullfile(main_path, sub_folder);
    csi_files = dir(fullfile(path, '*.dat'));
    fprintf("%d files in %s\n",length(csi_files),path);
    for i1 = 1:length(csi_files)
        csi_file = fullfile(csi_files(i1).folder,csi_files(i1).name);
        if mod(i1, 1000) == 0
            fprintf('Processing the %d-th file: %s\n', i1, csi_file);
        end
        % 解包名字
        pattern = '(\w+)-(\d+)-(\d+)-(\d+)-(\d+)-r(\d+)\.dat';
        tokens = regexp(csi_files(i1).name, pattern, 'tokens');
        if isempty(tokens)
            error('Filename does not match expected format: %s', filename);
        end
        tokens = tokens{1};
        parsed_data.id = str2double(tokens{1}(end)); % user1 -> 1
        parsed_data.a  = str2double(tokens{2});
        parsed_data.b  = str2double(tokens{3});
        parsed_data.c  = str2double(tokens{4});
        parsed_data.d  = str2double(tokens{5});
        parsed_data.Rx = str2double(tokens{6});

        %% todo 
        action = [action convert(parsed_data.a)];

        %% amp
        csi = read_bf_file(csi_file);
        csi = remove_empty_csi(csi);
        csi = scale_csi_5300(csi,"5300");
        amp = amp_DWT(amp_hampel_v2(get_amplitude(csi),20));
        amp = reshape(amp, size(amp,1),[]);
        csi_result{csi_result_idx} = amp;
        csi_result_idx = csi_result_idx+1;

    end
            save_filename = sprintf("all_csi_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'csi_result', '-v7.3');  % 保存为MAT文件
    fprintf("保存成功: %s\n", save_path);
    save_filename = sprintf("action_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'action', '-v7.3'); 
    fprintf("保存成功: %s\n", save_path);
    save_id = save_id+1;
end
%% 20181118
% 1: Push&Pull; 2: Sweep; 3: Clap; 4:Draw-O(Vertical); 5: Draw Zigzag(Vertical); 
% 6: Draw-N(Vertical);
convert = [1,2,3,12,10,11];
main_path = "F:\Widar3.0ReleaseData\CSI\20181118\";
sub_folders = ["user2","user3"];

for sub_folder = sub_folders
        action = [];
    csi_result = {};
    csi_result_idx = 1;

    path = fullfile(main_path, sub_folder);
    csi_files = dir(fullfile(path, '*.dat'));
    fprintf("%d files in %s\n",length(csi_files),path);
    for i1 = 1:length(csi_files)
        csi_file = fullfile(csi_files(i1).folder,csi_files(i1).name);
        if mod(i1, 1000) == 0
            fprintf('Processing the %d-th file: %s\n', i1, csi_file);
        end
        % 解包名字
        pattern = '(\w+)-(\d+)-(\d+)-(\d+)-(\d+)-r(\d+)\.dat';
        tokens = regexp(csi_files(i1).name, pattern, 'tokens');
        if isempty(tokens)
            error('Filename does not match expected format: %s', filename);
        end
        tokens = tokens{1};
        parsed_data.id = str2double(tokens{1}(end)); % user1 -> 1
        parsed_data.a  = str2double(tokens{2});
        parsed_data.b  = str2double(tokens{3});
        parsed_data.c  = str2double(tokens{4});
        parsed_data.d  = str2double(tokens{5});
        parsed_data.Rx = str2double(tokens{6});

        %% todo 
        action = [action convert(parsed_data.a)];
        
        %% amp
        csi = read_bf_file(csi_file);
        csi = remove_empty_csi(csi);
        if length(csi) < 50
            disp(['跳过文件: ', csi_file, '，length(csi) = ', num2str(length(csi))]);
            continue; 
        end
        csi = scale_csi_5300(csi,"5300");
        amp = amp_DWT(amp_hampel_v2(get_amplitude(csi),20));
        amp = reshape(amp, size(amp,1),[]);
        csi_result{csi_result_idx} = amp;
        csi_result_idx = csi_result_idx+1;

    end
            save_filename = sprintf("all_csi_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'csi_result', '-v7.3');  % 保存为MAT文件
    fprintf("保存成功: %s\n", save_path);
    save_filename = sprintf("action_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'action', '-v7.3'); 
    fprintf("保存成功: %s\n", save_path);
    save_id = save_id+1;
end

%% 20181121
% 1: Slide; 2: Draw-O(Horizontal); 3:Draw-Zigzag(Horizontal);
% 4: Draw N(Horizontal); 5: Draw Triangle(Horizontal); 6: Draw Rectangle(Horizontal);
convert = [4,6,9,5,8,7];
main_path = "F:\Widar3.0ReleaseData\CSI\20181121\";
sub_folders = ["user1","user2","user3"];

for sub_folder = sub_folders
        action = [];
    csi_result = {};
    csi_result_idx = 1;

    path = fullfile(main_path, sub_folder);
    csi_files = dir(fullfile(path, '*.dat'));
    fprintf("%d files in %s\n",length(csi_files),path);
    for i1 = 1:length(csi_files)
        csi_file = fullfile(csi_files(i1).folder,csi_files(i1).name);
        if mod(i1, 1000) == 0
            fprintf('Processing the %d-th file: %s\n', i1, csi_file);
        end
        % 解包名字
        pattern = '(\w+)-(\d+)-(\d+)-(\d+)-(\d+)-r(\d+)\.dat';
        tokens = regexp(csi_files(i1).name, pattern, 'tokens');
        if isempty(tokens)
            error('Filename does not match expected format: %s', filename);
        end
        tokens = tokens{1};
        parsed_data.id = str2double(tokens{1}(end)); % user1 -> 1
        parsed_data.a  = str2double(tokens{2});
        parsed_data.b  = str2double(tokens{3});
        parsed_data.c  = str2double(tokens{4});
        parsed_data.d  = str2double(tokens{5});
        parsed_data.Rx = str2double(tokens{6});

        %% todo 
        action = [action convert(parsed_data.a)];
        
        %% amp
        csi = read_bf_file(csi_file);
        csi = remove_empty_csi(csi);
                if length(csi) < 50
            disp(['跳过文件: ', csi_file, '，length(csi) = ', num2str(length(csi))]);
            continue; 
        end
        csi = scale_csi_5300(csi,"5300");
        amp = amp_DWT(amp_hampel_v2(get_amplitude(csi),20));
        amp = reshape(amp, size(amp,1),[]);
        csi_result{csi_result_idx} = amp;
        csi_result_idx = csi_result_idx+1;

    end
            save_filename = sprintf("all_csi_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'csi_result', '-v7.3');  % 保存为MAT文件
    fprintf("保存成功: %s\n", save_path);
    save_filename = sprintf("action_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'action', '-v7.3'); 
    fprintf("保存成功: %s\n", save_path);
    save_id = save_id+1;
end

%% 20181127
% 1: Slide; 2: Draw-O(Horizontal); 3:Draw-Zigzag(Horizontal); 
% 4: DrawN(Horizontal); 5: DrawTriangle(Horizontal); 6: DrawRectangle(Horizontal);
convert = [4,6,9,5,8,7];
main_path = "F:\Widar3.0ReleaseData\CSI\20181127\";
sub_folders = ["user2","user5"];

for sub_folder = sub_folders
        action = [];
    csi_result = {};
    csi_result_idx = 1;

    path = fullfile(main_path, sub_folder);
    csi_files = dir(fullfile(path, '*.dat'));
    fprintf("%d files in %s\n",length(csi_files),path);
    for i1 = 1:length(csi_files)
        csi_file = fullfile(csi_files(i1).folder,csi_files(i1).name);
        if mod(i1, 1000) == 0
            fprintf('Processing the %d-th file: %s\n', i1, csi_file);
        end
        % 解包名字
        pattern = '(\w+)-(\d+)-(\d+)-(\d+)-(\d+)-r(\d+)\.dat';
        tokens = regexp(csi_files(i1).name, pattern, 'tokens');
        if isempty(tokens)
            error('Filename does not match expected format: %s', filename);
        end
        tokens = tokens{1};
        parsed_data.id = str2double(tokens{1}(end)); % user1 -> 1
        parsed_data.a  = str2double(tokens{2});
        parsed_data.b  = str2double(tokens{3});
        parsed_data.c  = str2double(tokens{4});
        parsed_data.d  = str2double(tokens{5});
        parsed_data.Rx = str2double(tokens{6});

        %% todo 
        action = [action convert(parsed_data.a)];
        
        %% amp
        csi = read_bf_file(csi_file);
        csi = remove_empty_csi(csi);
                if length(csi) < 50
            disp(['跳过文件: ', csi_file, '，length(csi) = ', num2str(length(csi))]);
            continue; 
        end
        csi = scale_csi_5300(csi,"5300");
        amp = amp_DWT(amp_hampel_v2(get_amplitude(csi),20));
        amp = reshape(amp, size(amp,1),[]);
        csi_result{csi_result_idx} = amp;
        csi_result_idx = csi_result_idx+1;

    end
            save_filename = sprintf("all_csi_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'csi_result', '-v7.3');  % 保存为MAT文件
    fprintf("保存成功: %s\n", save_path);
    save_filename = sprintf("action_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'action', '-v7.3'); 
    fprintf("保存成功: %s\n", save_path);
    save_id = save_id+1;
end

%% 20181128
% 1: Push&Pull; 2: Sweep; 3: Clap; 
% 4:Draw-O(Horizontal); 5: DrawZigzag(Horizontal); 6: DrawN(Horizontal);
convert = [1,2,3,6,9,5];
main_path = "F:\Widar3.0ReleaseData\CSI\20181128\";
sub_folders = ["user6"];

for sub_folder = sub_folders
        action = [];
    csi_result = {};
    csi_result_idx = 1;

    path = fullfile(main_path, sub_folder);
    csi_files = dir(fullfile(path, '*.dat'));
    fprintf("%d files in %s\n",length(csi_files),path);
    for i1 = 1:length(csi_files)
        csi_file = fullfile(csi_files(i1).folder,csi_files(i1).name);
        if mod(i1, 1000) == 0
            fprintf('Processing the %d-th file: %s\n', i1, csi_file);
        end
        % 解包名字
        pattern = '(\w+)-(\d+)-(\d+)-(\d+)-(\d+)-r(\d+)\.dat';
        tokens = regexp(csi_files(i1).name, pattern, 'tokens');
        if isempty(tokens)
            error('Filename does not match expected format: %s', filename);
        end
        tokens = tokens{1};
        parsed_data.id = str2double(tokens{1}(end)); % user1 -> 1
        parsed_data.a  = str2double(tokens{2});
        parsed_data.b  = str2double(tokens{3});
        parsed_data.c  = str2double(tokens{4});
        parsed_data.d  = str2double(tokens{5});
        parsed_data.Rx = str2double(tokens{6});

        %% todo 
        action = [action convert(parsed_data.a)];
        
        %% amp
        csi = read_bf_file(csi_file);
        csi = remove_empty_csi(csi);
                if length(csi) < 50
            disp(['跳过文件: ', csi_file, '，length(csi) = ', num2str(length(csi))]);
            continue; 
        end
        csi = scale_csi_5300(csi,"5300");
        amp = amp_DWT(amp_hampel_v2(get_amplitude(csi),20));
        amp = reshape(amp, size(amp,1),[]);
        csi_result{csi_result_idx} = amp;
        csi_result_idx = csi_result_idx+1;

    end
            save_filename = sprintf("all_csi_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'csi_result', '-v7.3');  % 保存为MAT文件
    fprintf("保存成功: %s\n", save_path);
    save_filename = sprintf("action_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'action', '-v7.3'); 
    fprintf("保存成功: %s\n", save_path);
    save_id = save_id+1;
end
%% 20181130_user5_10_11
% 1: Push&Pull; 2: Sweep; 3: Clap; 4:Slide; 5: Draw-O(Horizontal);
% 6:Draw-Zigzag(Horizontal); 7: Draw N(Horizontal); 8: Draw Triangle(Horizontal); 
% 9: Draw Rectangle(Horizontal);
convert = [1,2,3,4,6,9,5,8,7];
main_path = "F:\Widar3.0ReleaseData\CSI\20181130_user5_10_11\";
sub_folders = ["user5","user10","user11"];

for sub_folder = sub_folders
        action = [];
    csi_result = {};
    csi_result_idx = 1;

    path = fullfile(main_path, sub_folder);
    csi_files = dir(fullfile(path, '*.dat'));
    fprintf("%d files in %s\n",length(csi_files),path);
    for i1 = 1:length(csi_files)
        csi_file = fullfile(csi_files(i1).folder,csi_files(i1).name);
        if mod(i1, 1000) == 0
            fprintf('Processing the %d-th file: %s\n', i1, csi_file);
        end
        % 解包名字
        pattern = '(\w+)-(\d+)-(\d+)-(\d+)-(\d+)-r(\d+)\.dat';
        tokens = regexp(csi_files(i1).name, pattern, 'tokens');
        if isempty(tokens)
            error('Filename does not match expected format: %s', filename);
        end
        tokens = tokens{1};
        parsed_data.id = str2double(tokens{1}(end)); % user1 -> 1
        parsed_data.a  = str2double(tokens{2});
        parsed_data.b  = str2double(tokens{3});
        parsed_data.c  = str2double(tokens{4});
        parsed_data.d  = str2double(tokens{5});
        parsed_data.Rx = str2double(tokens{6});

        %% todo 
        action = [action convert(parsed_data.a)];
        
        %% amp
        csi = read_bf_file(csi_file);
        csi = remove_empty_csi(csi);
                if length(csi) < 50
            disp(['跳过文件: ', csi_file, '，length(csi) = ', num2str(length(csi))]);
            continue; 
        end
        csi = scale_csi_5300(csi,"5300");
        amp = amp_DWT(amp_hampel_v2(get_amplitude(csi),20));
        amp = reshape(amp, size(amp,1),[]);
        csi_result{csi_result_idx} = amp;
        csi_result_idx = csi_result_idx+1;

    end
            save_filename = sprintf("all_csi_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'csi_result', '-v7.3');  % 保存为MAT文件
    fprintf("保存成功: %s\n", save_path);
    save_filename = sprintf("action_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'action', '-v7.3'); 
    fprintf("保存成功: %s\n", save_path);
    save_id = save_id+1;
end

%% 20181130_user12_13_14
% 1: Push&Pull; 2: Sweep; 3: Clap; 4:Slide; 5: Draw-O(Horizontal);
% 6:Draw-Zigzag(Horizontal); 7: Draw N(Horizontal); 8: Draw Triangle(Horizontal); 
% 9: Draw Rectangle(Horizontal);
convert = [1,2,3,4,6,9,5,8,7];
main_path = "F:\Widar3.0ReleaseData\CSI\20181130_user12_13_14\";
sub_folders = ["user12","user13","user14"];

for sub_folder = sub_folders
        action = [];
    csi_result = {};
    csi_result_idx = 1;

    path = fullfile(main_path, sub_folder);
    csi_files = dir(fullfile(path, '*.dat'));
    fprintf("%d files in %s\n",length(csi_files),path);
    for i1 = 1:length(csi_files)
        csi_file = fullfile(csi_files(i1).folder,csi_files(i1).name);
        if mod(i1, 1000) == 0
            fprintf('Processing the %d-th file: %s\n', i1, csi_file);
        end
        % 解包名字
        pattern = '(\w+)-(\d+)-(\d+)-(\d+)-(\d+)-r(\d+)\.dat';
        tokens = regexp(csi_files(i1).name, pattern, 'tokens');
        if isempty(tokens)
            error('Filename does not match expected format: %s', filename);
        end
        tokens = tokens{1};
        parsed_data.id = str2double(tokens{1}(end)); % user1 -> 1
        parsed_data.a  = str2double(tokens{2});
        parsed_data.b  = str2double(tokens{3});
        parsed_data.c  = str2double(tokens{4});
        parsed_data.d  = str2double(tokens{5});
        parsed_data.Rx = str2double(tokens{6});

        %% todo 
        action = [action convert(parsed_data.a)];
        
        %% amp
        csi = read_bf_file(csi_file);
        csi = remove_empty_csi(csi);
                if length(csi) < 50
            disp(['跳过文件: ', csi_file, '，length(csi) = ', num2str(length(csi))]);
            continue; 
        end
        csi = scale_csi_5300(csi,"5300");
        amp = amp_DWT(amp_hampel_v2(get_amplitude(csi),20));
        amp = reshape(amp, size(amp,1),[]);
        csi_result{csi_result_idx} = amp;
        csi_result_idx = csi_result_idx+1;

    end
            save_filename = sprintf("all_csi_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'csi_result', '-v7.3');  % 保存为MAT文件
    fprintf("保存成功: %s\n", save_path);
    save_filename = sprintf("action_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'action', '-v7.3'); 
    fprintf("保存成功: %s\n", save_path);
    save_id = save_id+1;
end

%% 20181130_user15_16_17
% 1: Push&Pull; 2: Sweep; 3: Clap; 4:Slide; 5: Draw-O(Horizontal);
% 6:Draw-Zigzag(Horizontal); 7: Draw N(Horizontal); 8: Draw Triangle(Horizontal); 
% 9: Draw Rectangle(Horizontal);
convert = [1,2,3,4,6,9,5,8,7];
main_path = "F:\Widar3.0ReleaseData\CSI\20181130_user15_16_17\";
sub_folders = ["user15","user16","user17"];

for sub_folder = sub_folders
        action = [];
    csi_result = {};
    csi_result_idx = 1;

    path = fullfile(main_path, sub_folder);
    csi_files = dir(fullfile(path, '*.dat'));
    fprintf("%d files in %s\n",length(csi_files),path);
    for i1 = 1:length(csi_files)
        csi_file = fullfile(csi_files(i1).folder,csi_files(i1).name);
        if mod(i1, 1000) == 0
            fprintf('Processing the %d-th file: %s\n', i1, csi_file);
        end
        % 解包名字
        pattern = '(\w+)-(\d+)-(\d+)-(\d+)-(\d+)-r(\d+)\.dat';
        tokens = regexp(csi_files(i1).name, pattern, 'tokens');
        if isempty(tokens)
            error('Filename does not match expected format: %s', filename);
        end
        tokens = tokens{1};
        parsed_data.id = str2double(tokens{1}(end)); % user1 -> 1
        parsed_data.a  = str2double(tokens{2});
        parsed_data.b  = str2double(tokens{3});
        parsed_data.c  = str2double(tokens{4});
        parsed_data.d  = str2double(tokens{5});
        parsed_data.Rx = str2double(tokens{6});

        %% todo 
        action = [action convert(parsed_data.a)];
        
        %% amp
        csi = read_bf_file(csi_file);
        csi = remove_empty_csi(csi);
                if length(csi) < 50
            disp(['跳过文件: ', csi_file, '，length(csi) = ', num2str(length(csi))]);
            continue; 
        end
        csi = scale_csi_5300(csi,"5300");
        amp = amp_DWT(amp_hampel_v2(get_amplitude(csi),20));
        amp = reshape(amp, size(amp,1),[]);
        csi_result{csi_result_idx} = amp;
        csi_result_idx = csi_result_idx+1;

    end
            save_filename = sprintf("all_csi_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'csi_result', '-v7.3');  % 保存为MAT文件
    fprintf("保存成功: %s\n", save_path);
    save_filename = sprintf("action_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'action', '-v7.3'); 
    fprintf("保存成功: %s\n", save_path);
    save_id = save_id+1;
end

%% 20181204
% 1: Push&Pull; 2: Sweep; 3: Clap; 4:Slide; 5: Draw-O(Horizontal);
% 6:Draw-Zigzag(Horizontal); 7: Draw N(Horizontal); 8: Draw Triangle(Horizontal); 
% 9: Draw Rectangle(Horizontal);
convert = [1,2,3,4,6,9,5,8,7];
main_path = "F:\Widar3.0ReleaseData\CSI\20181204\";
sub_folders = ["user1"];

for sub_folder = sub_folders
        action = [];
    csi_result = {};
    csi_result_idx = 1;

    path = fullfile(main_path, sub_folder);
    csi_files = dir(fullfile(path, '*.dat'));
    fprintf("%d files in %s\n",length(csi_files),path);
    for i1 = 1:length(csi_files)
        csi_file = fullfile(csi_files(i1).folder,csi_files(i1).name);
        if mod(i1, 1000) == 0
            fprintf('Processing the %d-th file: %s\n', i1, csi_file);
        end
        % 解包名字
        pattern = '(\w+)-(\d+)-(\d+)-(\d+)-(\d+)-r(\d+)\.dat';
        tokens = regexp(csi_files(i1).name, pattern, 'tokens');
        if isempty(tokens)
            error('Filename does not match expected format: %s', filename);
        end
        tokens = tokens{1};
        parsed_data.id = str2double(tokens{1}(end)); % user1 -> 1
        parsed_data.a  = str2double(tokens{2});
        parsed_data.b  = str2double(tokens{3});
        parsed_data.c  = str2double(tokens{4});
        parsed_data.d  = str2double(tokens{5});
        parsed_data.Rx = str2double(tokens{6});

        %% todo 
        action = [action convert(parsed_data.a)];
        
        %% amp
        csi = read_bf_file(csi_file);
        csi = remove_empty_csi(csi);
                if length(csi) < 50
            disp(['跳过文件: ', csi_file, '，length(csi) = ', num2str(length(csi))]);
            continue; 
        end
        csi = scale_csi_5300(csi,"5300");
        amp = amp_DWT(amp_hampel_v2(get_amplitude(csi),20));
        amp = reshape(amp, size(amp,1),[]);
        csi_result{csi_result_idx} = amp;
        csi_result_idx = csi_result_idx+1;

    end
            save_filename = sprintf("all_csi_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'csi_result', '-v7.3');  % 保存为MAT文件
    fprintf("保存成功: %s\n", save_path);
    save_filename = sprintf("action_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'action', '-v7.3'); 
    fprintf("保存成功: %s\n", save_path);
    save_id = save_id+1;
end

%% 20181205_user2
% [User2]1: Draw-O(Horizontal); 2:Draw-Zigzag(Horizontal); 3: DrawN(Horizontal); 
% 4: DrawTriangle(Horizontal); 5: DrawRectangle(Horizontal);
convert = [6,9,5,8,7];
main_path = "F:\Widar3.0ReleaseData\CSI\20181205\";
sub_folders = ["user2"];

for sub_folder = sub_folders
        action = [];
    csi_result = {};
    csi_result_idx = 1;

    path = fullfile(main_path, sub_folder);
    csi_files = dir(fullfile(path, '*.dat'));
    fprintf("%d files in %s\n",length(csi_files),path);
    for i1 = 1:length(csi_files)
        csi_file = fullfile(csi_files(i1).folder,csi_files(i1).name);
        if mod(i1, 1000) == 0
            fprintf('Processing the %d-th file: %s\n', i1, csi_file);
        end
        % 解包名字
        pattern = '(\w+)-(\d+)-(\d+)-(\d+)-(\d+)-r(\d+)\.dat';
        tokens = regexp(csi_files(i1).name, pattern, 'tokens');
        if isempty(tokens)
            error('Filename does not match expected format: %s', filename);
        end
        tokens = tokens{1};
        parsed_data.id = str2double(tokens{1}(end)); % user1 -> 1
        parsed_data.a  = str2double(tokens{2});
        parsed_data.b  = str2double(tokens{3});
        parsed_data.c  = str2double(tokens{4});
        parsed_data.d  = str2double(tokens{5});
        parsed_data.Rx = str2double(tokens{6});

        %% todo 
        action = [action convert(parsed_data.a)];
        
        %% amp
        csi = read_bf_file(csi_file);
        csi = remove_empty_csi(csi);
                if length(csi) < 50
            disp(['跳过文件: ', csi_file, '，length(csi) = ', num2str(length(csi))]);
            continue; 
        end
        csi = scale_csi_5300(csi,"5300");
        amp = amp_DWT(amp_hampel_v2(get_amplitude(csi),20));
        amp = reshape(amp, size(amp,1),[]);
        csi_result{csi_result_idx} = amp;
        csi_result_idx = csi_result_idx+1;

    end
            save_filename = sprintf("all_csi_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'csi_result', '-v7.3');  % 保存为MAT文件
    fprintf("保存成功: %s\n", save_path);
    save_filename = sprintf("action_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'action', '-v7.3'); 
    fprintf("保存成功: %s\n", save_path);
    save_id = save_id+1;
end
%% 20181205_user3
% [User3]1: Slide; 2: DrawO(Horizontal); 3: DrawZigzag(Horizontal);
% 4: DrawN(Horizontal); 5: DrawTriangle(Horizontal); 6: Draw Rectangle(Horizontal);
convert = [4,6,9,5,8,7];
main_path = "F:\Widar3.0ReleaseData\CSI\20181205\";
sub_folders = ["user3"];

for sub_folder = sub_folders
        action = [];
    csi_result = {};
    csi_result_idx = 1;

    path = fullfile(main_path, sub_folder);
    csi_files = dir(fullfile(path, '*.dat'));
    fprintf("%d files in %s\n",length(csi_files),path);
    for i1 = 1:length(csi_files)
        csi_file = fullfile(csi_files(i1).folder,csi_files(i1).name);
        if mod(i1, 1000) == 0
            fprintf('Processing the %d-th file: %s\n', i1, csi_file);
        end
        % 解包名字
        pattern = '(\w+)-(\d+)-(\d+)-(\d+)-(\d+)-r(\d+)\.dat';
        tokens = regexp(csi_files(i1).name, pattern, 'tokens');
        if isempty(tokens)
            error('Filename does not match expected format: %s', filename);
        end
        tokens = tokens{1};
        parsed_data.id = str2double(tokens{1}(end)); % user1 -> 1
        parsed_data.a  = str2double(tokens{2});
        parsed_data.b  = str2double(tokens{3});
        parsed_data.c  = str2double(tokens{4});
        parsed_data.d  = str2double(tokens{5});
        parsed_data.Rx = str2double(tokens{6});

        %% todo 
        action = [action convert(parsed_data.a)];
        
        %% amp
        csi = read_bf_file(csi_file);
        csi = remove_empty_csi(csi);
                if length(csi) < 50
            disp(['跳过文件: ', csi_file, '，length(csi) = ', num2str(length(csi))]);
            continue; 
        end
        csi = scale_csi_5300(csi,"5300");
        amp = amp_DWT(amp_hampel_v2(get_amplitude(csi),20));
        amp = reshape(amp, size(amp,1),[]);
        csi_result{csi_result_idx} = amp;
        csi_result_idx = csi_result_idx+1;

    end
            save_filename = sprintf("all_csi_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'csi_result', '-v7.3');  % 保存为MAT文件
    fprintf("保存成功: %s\n", save_path);
    save_filename = sprintf("action_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'action', '-v7.3'); 
    fprintf("保存成功: %s\n", save_path);
    save_id = save_id+1;
end
%% 20181208
% [User2]:1: Push&Pull; 2: Sweep; 3:Clap; 4: Slide;
convert = [1,2,3,4];
main_path = "F:\Widar3.0ReleaseData\CSI\20181208\";
sub_folders = ["user2"];

for sub_folder = sub_folders
        action = [];
    csi_result = {};
    csi_result_idx = 1;

    path = fullfile(main_path, sub_folder);
    csi_files = dir(fullfile(path, '*.dat'));
    fprintf("%d files in %s\n",length(csi_files),path);
    for i1 = 1:length(csi_files)
        csi_file = fullfile(csi_files(i1).folder,csi_files(i1).name);
        if mod(i1, 1000) == 0
            fprintf('Processing the %d-th file: %s\n', i1, csi_file);
        end
        % 解包名字
        pattern = '(\w+)-(\d+)-(\d+)-(\d+)-(\d+)-r(\d+)\.dat';
        tokens = regexp(csi_files(i1).name, pattern, 'tokens');
        if isempty(tokens)
            error('Filename does not match expected format: %s', filename);
        end
        tokens = tokens{1};
        parsed_data.id = str2double(tokens{1}(end)); % user1 -> 1
        parsed_data.a  = str2double(tokens{2});
        parsed_data.b  = str2double(tokens{3});
        parsed_data.c  = str2double(tokens{4});
        parsed_data.d  = str2double(tokens{5});
        parsed_data.Rx = str2double(tokens{6});

        %% todo 
        action = [action convert(parsed_data.a)];
        
        %% amp
        csi = read_bf_file(csi_file);
        csi = remove_empty_csi(csi);
                if length(csi) < 50
            disp(['跳过文件: ', csi_file, '，length(csi) = ', num2str(length(csi))]);
            continue; 
        end
        csi = scale_csi_5300(csi,"5300");
        amp = amp_DWT(amp_hampel_v2(get_amplitude(csi),20));
        amp = reshape(amp, size(amp,1),[]);
        csi_result{csi_result_idx} = amp;
        csi_result_idx = csi_result_idx+1;

    end
            save_filename = sprintf("all_csi_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'csi_result', '-v7.3');  % 保存为MAT文件
    fprintf("保存成功: %s\n", save_path);
    save_filename = sprintf("action_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'action', '-v7.3'); 
    fprintf("保存成功: %s\n", save_path);
    save_id = save_id+1;
end
%% 20181208
% [User3]:1: Push&Pull;2: Sweep; 3:Clap;
convert = [1,2,3];
main_path = "F:\Widar3.0ReleaseData\CSI\20181208\";
sub_folders = ["user3"];

for sub_folder = sub_folders
        action = [];
    csi_result = {};
    csi_result_idx = 1;

    path = fullfile(main_path, sub_folder);
    csi_files = dir(fullfile(path, '*.dat'));
    fprintf("%d files in %s\n",length(csi_files),path);
    for i1 = 1:length(csi_files)
        csi_file = fullfile(csi_files(i1).folder,csi_files(i1).name);
        if mod(i1, 1000) == 0
            fprintf('Processing the %d-th file: %s\n', i1, csi_file);
        end
        % 解包名字
        pattern = '(\w+)-(\d+)-(\d+)-(\d+)-(\d+)-r(\d+)\.dat';
        tokens = regexp(csi_files(i1).name, pattern, 'tokens');
        if isempty(tokens)
            error('Filename does not match expected format: %s', filename);
        end
        tokens = tokens{1};
        parsed_data.id = str2double(tokens{1}(end)); % user1 -> 1
        parsed_data.a  = str2double(tokens{2});
        parsed_data.b  = str2double(tokens{3});
        parsed_data.c  = str2double(tokens{4});
        parsed_data.d  = str2double(tokens{5});
        parsed_data.Rx = str2double(tokens{6});

        %% todo 
        action = [action convert(parsed_data.a)];
        
        %% amp
        csi = read_bf_file(csi_file);
        csi = remove_empty_csi(csi);
                if length(csi) < 50
            disp(['跳过文件: ', csi_file, '，length(csi) = ', num2str(length(csi))]);
            continue; 
        end
        csi = scale_csi_5300(csi,"5300");
        amp = amp_DWT(amp_hampel_v2(get_amplitude(csi),20));
        amp = reshape(amp, size(amp,1),[]);
        csi_result{csi_result_idx} = amp;
        csi_result_idx = csi_result_idx+1;

    end
            save_filename = sprintf("all_csi_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'csi_result', '-v7.3');  % 保存为MAT文件
    fprintf("保存成功: %s\n", save_path);
    save_filename = sprintf("action_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'action', '-v7.3'); 
    fprintf("保存成功: %s\n", save_path);
    save_id = save_id+1;
end

%% 20181209
% [User2]:1: Push&Pull;
convert = [1];
main_path = "F:\Widar3.0ReleaseData\CSI\20181209\";
sub_folders = ["user2"];

for sub_folder = sub_folders
        action = [];
    csi_result = {};
    csi_result_idx = 1;

    path = fullfile(main_path, sub_folder);
    csi_files = dir(fullfile(path, '*.dat'));
    fprintf("%d files in %s\n",length(csi_files),path);
    for i1 = 1:length(csi_files)
        csi_file = fullfile(csi_files(i1).folder,csi_files(i1).name);
        if mod(i1, 1000) == 0
            fprintf('Processing the %d-th file: %s\n', i1, csi_file);
        end
        % 解包名字
        pattern = '(\w+)-(\d+)-(\d+)-(\d+)-(\d+)-r(\d+)\.dat';
        tokens = regexp(csi_files(i1).name, pattern, 'tokens');
        if isempty(tokens)
            error('Filename does not match expected format: %s', filename);
        end
        tokens = tokens{1};
        parsed_data.id = str2double(tokens{1}(end)); % user1 -> 1
        parsed_data.a  = str2double(tokens{2});
        parsed_data.b  = str2double(tokens{3});
        parsed_data.c  = str2double(tokens{4});
        parsed_data.d  = str2double(tokens{5});
        parsed_data.Rx = str2double(tokens{6});

        %% todo 
        action = [action convert(parsed_data.a)];
        
        %% amp
        csi = read_bf_file(csi_file);
        csi = remove_empty_csi(csi);
                if length(csi) < 50
            disp(['跳过文件: ', csi_file, '，length(csi) = ', num2str(length(csi))]);
            continue; 
        end
        csi = scale_csi_5300(csi,"5300");
        amp = amp_DWT(amp_hampel_v2(get_amplitude(csi),20));
        amp = reshape(amp, size(amp,1),[]);
        csi_result{csi_result_idx} = amp;
        csi_result_idx = csi_result_idx+1;

    end
        save_filename = sprintf("all_csi_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'csi_result', '-v7.3');  % 保存为MAT文件
    fprintf("保存成功: %s\n", save_path);
    save_filename = sprintf("action_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'action', '-v7.3'); 
    fprintf("保存成功: %s\n", save_path);
    save_id = save_id+1;
end

%% 20181209
% [User6]:1: Push&Pull; 2: Sweep; 3:Clap; 
% 4: Slide; 5: Draw-O(Horizontal); 6: Draw-Zigzag(Horizontal);
convert = [1,2,3,4,6,9];
main_path = "F:\Widar3.0ReleaseData\CSI\20181209\";
sub_folders = ["user6"];

for sub_folder = sub_folders
        action = [];
    csi_result = {};
    csi_result_idx = 1;

    path = fullfile(main_path, sub_folder);
    csi_files = dir(fullfile(path, '*.dat'));
    fprintf("%d files in %s\n",length(csi_files),path);
    for i1 = 1:length(csi_files)
        csi_file = fullfile(csi_files(i1).folder,csi_files(i1).name);
        if mod(i1, 1000) == 0
            fprintf('Processing the %d-th file: %s\n', i1, csi_file);
        end
        % 解包名字
        pattern = '(\w+)-(\d+)-(\d+)-(\d+)-(\d+)-r(\d+)\.dat';
        tokens = regexp(csi_files(i1).name, pattern, 'tokens');
        if isempty(tokens)
            error('Filename does not match expected format: %s', filename);
        end
        tokens = tokens{1};
        parsed_data.id = str2double(tokens{1}(end)); % user1 -> 1
        parsed_data.a  = str2double(tokens{2});
        parsed_data.b  = str2double(tokens{3});
        parsed_data.c  = str2double(tokens{4});
        parsed_data.d  = str2double(tokens{5});
        parsed_data.Rx = str2double(tokens{6});

        %% todo 
        action = [action convert(parsed_data.a)];
        
        %% amp
        csi = read_bf_file(csi_file);
        csi = remove_empty_csi(csi);
                if length(csi) < 50
            disp(['跳过文件: ', csi_file, '，length(csi) = ', num2str(length(csi))]);
            continue; 
        end
        csi = scale_csi_5300(csi,"5300");
        amp = amp_DWT(amp_hampel_v2(get_amplitude(csi),20));
        amp = reshape(amp, size(amp,1),[]);
        csi_result{csi_result_idx} = amp;
        csi_result_idx = csi_result_idx+1;

    end
            save_filename = sprintf("all_csi_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'csi_result', '-v7.3');  % 保存为MAT文件
    fprintf("保存成功: %s\n", save_path);
    save_filename = sprintf("action_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'action', '-v7.3'); 
    fprintf("保存成功: %s\n", save_path);
    save_id = save_id+1;
end

%% 20181211
% 1: Push&Pull; 2: Sweep; 3: Clap; 4:Slide; 
% 5: Draw-O(Horizontal); 6:Draw-Zigzag(Horizontal);
convert = [1,2,3,4,6,9];
main_path = "F:\Widar3.0ReleaseData\CSI\20181211\";
sub_folders = ["user3","user7","user8","user9"];

for sub_folder = sub_folders
        action = [];
    csi_result = {};
    csi_result_idx = 1;

    path = fullfile(main_path, sub_folder);
    csi_files = dir(fullfile(path, '*.dat'));
    fprintf("%d files in %s\n",length(csi_files),path);
    for i1 = 1:length(csi_files)
        csi_file = fullfile(csi_files(i1).folder,csi_files(i1).name);
        if mod(i1, 1000) == 0
            fprintf('Processing the %d-th file: %s\n', i1, csi_file);
        end
        % 解包名字
        pattern = '(\w+)-(\d+)-(\d+)-(\d+)-(\d+)-r(\d+)\.dat';
        tokens = regexp(csi_files(i1).name, pattern, 'tokens');
        if isempty(tokens)
            error('Filename does not match expected format: %s', filename);
        end
        tokens = tokens{1};
        parsed_data.id = str2double(tokens{1}(end)); % user1 -> 1
        parsed_data.a  = str2double(tokens{2});
        parsed_data.b  = str2double(tokens{3});
        parsed_data.c  = str2double(tokens{4});
        parsed_data.d  = str2double(tokens{5});
        parsed_data.Rx = str2double(tokens{6});

        %% todo 
        action = [action convert(parsed_data.a)];
        
        %% amp
        csi = read_bf_file(csi_file);
        csi = remove_empty_csi(csi);
                if length(csi) < 50
            disp(['跳过文件: ', csi_file, '，length(csi) = ', num2str(length(csi))]);
            continue; 
        end
        csi = scale_csi_5300(csi,"5300");
        amp = amp_DWT(amp_hampel_v2(get_amplitude(csi),20));
        amp = reshape(amp, size(amp,1),[]);
        csi_result{csi_result_idx} = amp;
        csi_result_idx = csi_result_idx+1;

    end
            save_filename = sprintf("all_csi_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'csi_result', '-v7.3');  % 保存为MAT文件
    fprintf("保存成功: %s\n", save_path);
    save_filename = sprintf("action_%d.mat", save_id);
    save_path = fullfile(save_dir, save_filename);
    save(save_path, 'action', '-v7.3'); 
    fprintf("保存成功: %s\n", save_path);
    save_id = save_id+1;
end

