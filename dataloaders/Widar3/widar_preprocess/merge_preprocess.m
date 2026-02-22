function merge_preprocess(save_dir, save_id, total_files)
% Merge per-file outputs produced by worker-mode into final all_csi and action MAT
% Usage: merge_preprocess('/path/to/processed', 1, 18000)

if nargin < 3
    error('Usage: merge_preprocess(save_dir, save_id, total_files)')
end
csi_result = cell(1, total_files);
action = [];
for i = 1:total_files
    cfile = fullfile(save_dir, sprintf('csi_%d_%06d.mat', save_id, i));
    afile = fullfile(save_dir, sprintf('action_%d_%06d.mat', save_id, i));
    if ~exist(cfile, 'file')
        warning('Missing file: %s', cfile);
        continue;
    end
    if ~exist(afile, 'file')
        warning('Missing file: %s', afile);
    end
    S = load(cfile, 'csi_single');
    if isfield(S, 'csi_single')
        csi_result{i} = S.csi_single;
    else
        warning('File %s does not contain csi_single', cfile);
    end
    A = load(afile, 'action_single');
    if isfield(A, 'action_single')
        action(end+1) = A.action_single; %#ok<AGROW>
    else
        action(end+1) = NaN; %#ok<AGROW>
    end
end
% Save merged results
out_csi = fullfile(save_dir, sprintf('all_csi_%d.mat', save_id));
out_action = fullfile(save_dir, sprintf('action_%d.mat', save_id));
save(out_csi, 'csi_result', '-v7.3');
save(out_action, 'action');
fprintf('Merged saved to %s and %s\n', out_csi, out_action);
end
