% Independent second half of inferred test twins (test_008 ... test_015).
audit_dir = fileparts(mfilename('fullpath'));
project_dir = fileparts(fileparts(audit_dir));
native_dir = fullfile(project_dir, 'Data', 'MCT2024-MCT_v1.0.0', ...
    'UTokyo-FML-MCT2024-147ccd2');
addpath(audit_dir);
addpath(native_dir);
init_file = fullfile(native_dir, 'yfinal_2000.mat');
base_dir = fullfile(audit_dir, 'reverse_robust3');
out_dir = fullfile(base_dir, 'native_twins');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
tbl = readtable(fullfile(base_dir, 'inferred_parameters.csv'));
for i = 1:height(tbl)
    id = char(string(tbl.id(i)));
    index = sscanf(id, 'test_%d');
    if index < 8, continue; end
    mode = char(string(tbl.mode(i)));
    target = fullfile(out_dir, sprintf('%s_%s_%d.mat', id, mode, tbl.rep(i)));
    if exist(target, 'file'), continue; end
    fprintf('Running %s %s replicate %d\n', id, mode, tbl.rep(i));
    a = mct2024_reverse_3param(0.10, 40, init_file, ...
        tbl.mct_gain(i), tbl.pCa0_scale(i), tbl.serca_scale(i));
    save(target, 'a', '-v7.3');
end
