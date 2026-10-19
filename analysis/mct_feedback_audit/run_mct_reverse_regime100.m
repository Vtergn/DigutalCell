function run_mct_reverse_regime100(block)
% Uniform high-load 100-beat run for the frozen regime viability gate.
audit_dir = fileparts(mfilename('fullpath'));
project_dir = fileparts(fileparts(audit_dir));
native_dir = fullfile(project_dir, 'Data', 'MCT2024-MCT_v1.0.0', ...
    'UTokyo-FML-MCT2024-147ccd2');
addpath(audit_dir);
addpath(native_dir);
init_file = fullfile(native_dir, 'yfinal_2000.mat');
base_dir = fullfile(audit_dir, 'reverse_robust3');
output_dir = fullfile(base_dir, 'regime100', 'native');
if ~exist(output_dir, 'dir'), mkdir(output_dir); end
tbl = readtable(fullfile(base_dir, 'design.csv'));
switch char(block)
    case 'train_a', wanted = 1:16;
    case 'train_b', wanted = 17:32;
    case 'train_c', wanted = 33:48;
    case 'test', wanted = 49:64;
    otherwise, error('Unknown block: %s', string(block));
end
for i = wanted
    id = char(string(tbl.id(i)));
    target = fullfile(output_dir, sprintf('%s_l10_100.mat', id));
    if exist(target, 'file'), continue; end
    if strcmp(id, 'train_036')
        copyfile(fullfile(base_dir, 'train036_high100.mat'), target);
        continue;
    end
    fprintf('Running %s at CL 0.10 for 100 beats\n', id);
    a = mct2024_reverse_3param(0.10, 100, init_file, ...
        tbl.mct_gain(i), tbl.pCa0_scale(i), tbl.serca_scale(i));
    save(target, 'a', '-v7.3');
    fprintf('Saved %s\n', target);
end
end
