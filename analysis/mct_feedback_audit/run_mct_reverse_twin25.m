% Run the native high-load ODE at target-blind inferred parameters.
audit_dir = fileparts(mfilename('fullpath'));
project_dir = fileparts(fileparts(audit_dir));
native_dir = fullfile(project_dir, 'Data', 'MCT2024-MCT_v1.0.0', ...
    'UTokyo-FML-MCT2024-147ccd2');
addpath(audit_dir);
addpath(native_dir);
init_file = fullfile(native_dir, 'yfinal_2000.mat');
target_dir = fullfile(audit_dir, 'reverse_twin25');
if ~exist(target_dir, 'dir'), mkdir(target_dir); end
tbl = readtable(fullfile(audit_dir, 'reverse_grid25_inferred_params.csv'));
for i = 1:height(tbl)
    mode = char(string(tbl.mode(i)));
    id = sprintf('twin_%02d_%s', tbl.index(i), mode);
    target = fullfile(target_dir, [id '.mat']);
    if exist(target, 'file'), continue; end
    fprintf('Running %s\n', id);
    a = mct2024_reverse_variant(0.10, 20, init_file, ...
        tbl.mct_gain(i), tbl.pCa0_scale(i));
    save(target, 'a', '-v7.3');
end
