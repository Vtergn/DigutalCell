% Independent middle train block; main runner remains at earlier identities.
audit_dir = fileparts(mfilename('fullpath'));
project_dir = fileparts(fileparts(audit_dir));
native_dir = fullfile(project_dir, 'Data', 'MCT2024-MCT_v1.0.0', ...
    'UTokyo-FML-MCT2024-147ccd2');
addpath(audit_dir);
addpath(native_dir);
init_file = fullfile(native_dir, 'yfinal_2000.mat');
base_dir = fullfile(audit_dir, 'reverse_robust3');
output_dir = fullfile(base_dir, 'native_cohort');
if ~exist(output_dir, 'dir'), mkdir(output_dir); end
tbl = readtable(fullfile(base_dir, 'design.csv'));
loads = [0.02, 0.06, 0.10];
for i = 1:height(tbl)
    id = char(string(tbl.id(i)));
    if ~startsWith(id, 'train_'), continue; end
    index = sscanf(id, 'train_%d');
    if index < 24 || index > 31, continue; end
    for il = 1:numel(loads)
        target = fullfile(output_dir, sprintf('%s_l%02d.mat', id, round(100*loads(il))));
        if exist(target, 'file'), continue; end
        fprintf('Running %s at CL %.2f\n', id, loads(il));
        a = mct2024_reverse_3param(loads(il), 40, init_file, ...
            tbl.mct_gain(i), tbl.pCa0_scale(i), tbl.serca_scale(i));
        save(target, 'a', '-v7.3');
    end
end
