% Exploratory trajectory check of the 60-beat growing alternation example.
audit_dir = fileparts(mfilename('fullpath'));
project_dir = fileparts(fileparts(audit_dir));
native_dir = fullfile(project_dir, 'Data', 'MCT2024-MCT_v1.0.0', ...
    'UTokyo-FML-MCT2024-147ccd2');
addpath(audit_dir);
addpath(native_dir);
init_file = fullfile(native_dir, 'yfinal_2000.mat');
base_dir = fullfile(audit_dir, 'reverse_robust3');
tbl = readtable(fullfile(base_dir, 'design.csv'));
i = find(strcmp(string(tbl.id), 'train_036'), 1);
target = fullfile(base_dir, 'train036_high100.mat');
if ~exist(target, 'file')
    a = mct2024_reverse_3param(0.10, 100, init_file, ...
        tbl.mct_gain(i), tbl.pCa0_scale(i), tbl.serca_scale(i));
    save(target, 'a', '-v7.3');
end
