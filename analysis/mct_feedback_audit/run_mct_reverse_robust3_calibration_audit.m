% Native calibration check on four prespecified held-out test identities.
audit_dir = fileparts(mfilename('fullpath'));
project_dir = fileparts(fileparts(audit_dir));
native_dir = fullfile(project_dir, 'Data', 'MCT2024-MCT_v1.0.0', ...
    'UTokyo-FML-MCT2024-147ccd2');
addpath(audit_dir);
addpath(native_dir);
init_file = fullfile(native_dir, 'yfinal_2000.mat');
base_dir = fullfile(audit_dir, 'reverse_robust3');
out_dir = fullfile(base_dir, 'calibration_audit');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
tbl = readtable(fullfile(base_dir, 'inferred_parameters.csv'));
chosen = {'test_000', 'test_005', 'test_010', 'test_015'};
for i = 1:height(tbl)
    id = char(string(tbl.id(i)));
    mode = char(string(tbl.mode(i)));
    if ~ismember(id, chosen), continue; end
    if strcmp(mode, 'noisy_lowmid'), continue; end
    if strcmp(mode, 'clean_low'), loads = [0.02]; else, loads = [0.02, 0.06]; end
    for j = 1:numel(loads)
        target = fullfile(out_dir, sprintf('%s_%s_l%02d.mat', id, mode, round(loads(j)*100)));
        if exist(target, 'file'), continue; end
        fprintf('Calibrate %s %s CL %.2f\n', id, mode, loads(j));
        a = mct2024_reverse_3param(loads(j), 40, init_file, ...
            tbl.mct_gain(i), tbl.pCa0_scale(i), tbl.serca_scale(i));
        save(target, 'a', '-v7.3');
    end
end
