% Native-ODE calibration check of selected inferred parameter vectors.
audit_dir = fileparts(mfilename('fullpath'));
project_dir = fileparts(fileparts(audit_dir));
native_dir = fullfile(project_dir, 'Data', 'MCT2024-MCT_v1.0.0', ...
    'UTokyo-FML-MCT2024-147ccd2');
addpath(audit_dir);
addpath(native_dir);
init_file = fullfile(native_dir, 'yfinal_2000.mat');
target_dir = fullfile(audit_dir, 'reverse_calibration_audit');
if ~exist(target_dir, 'dir'), mkdir(target_dir); end
tbl = readtable(fullfile(audit_dir, 'reverse_grid25_inferred_params.csv'));
% The first five are largest low-only gain errors; the last five sample the
% grid for checking low+mid emulator calibration with original equations.
low_indices = [3, 21, 2, 15, 22];
lowmid_indices = [0, 6, 12, 18, 24];
for i = 1:height(tbl)
    index = tbl.index(i);
    mode = char(string(tbl.mode(i)));
    if strcmp(mode, 'low')
        if ~ismember(index, low_indices), continue; end
        loads = [0.02];
    else
        if ~ismember(index, lowmid_indices), continue; end
        loads = [0.02, 0.06];
    end
    for k = 1:numel(loads)
        id = sprintf('audit_%02d_%s_l%02d', index, mode, round(100*loads(k)));
        target = fullfile(target_dir, [id '.mat']);
        if exist(target, 'file'), continue; end
        fprintf('Running %s\n', id);
        a = mct2024_reverse_variant(loads(k), 20, init_file, ...
            tbl.mct_gain(i), tbl.pCa0_scale(i));
        save(target, 'a', '-v7.3');
    end
end
