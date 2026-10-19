function run_mct_regime100_load_check(id)
% Exploratory long low/middle-load check of selected high-load period-2 candidates.
audit_dir = fileparts(mfilename('fullpath'));
project_dir = fileparts(fileparts(audit_dir));
native_dir = fullfile(project_dir, 'Data', 'MCT2024-MCT_v1.0.0', ...
    'UTokyo-FML-MCT2024-147ccd2');
addpath(audit_dir);
addpath(native_dir);
init_file = fullfile(native_dir, 'yfinal_2000.mat');
base_dir = fullfile(audit_dir, 'reverse_robust3');
out_dir = fullfile(base_dir, 'regime100', 'load_check');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
tbl = readtable(fullfile(base_dir, 'design.csv'));
i = find(strcmp(string(tbl.id), string(id)), 1);
if isempty(i), error('Unknown id: %s', string(id)); end
for load = [0.02, 0.06]
    target = fullfile(out_dir, sprintf('%s_l%02d_100.mat', ...
        char(string(id)), round(load * 100)));
    if exist(target, 'file'), continue; end
    fprintf('Load check %s CL %g\n', string(id), load);
    a = mct2024_reverse_3param(load, 100, init_file, ...
        tbl.mct_gain(i), tbl.pCa0_scale(i), tbl.serca_scale(i));
    save(target, 'a', '-v7.3');
end
end
