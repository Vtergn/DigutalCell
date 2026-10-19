% Convergence check for the nearest low-load pair from the 20-beat pilot.
audit_dir = fileparts(mfilename('fullpath'));
project_dir = fileparts(fileparts(audit_dir));
native_dir = fullfile(project_dir, 'Data', 'MCT2024-MCT_v1.0.0', ...
    'UTokyo-FML-MCT2024-147ccd2');
addpath(audit_dir);
addpath(native_dir);
init_file = fullfile(native_dir, 'yfinal_2000.mat');
target_dir = fullfile(audit_dir, 'reverse_cohort');
gains = [0.25, 1.25];
pca_scales = [1.1, 1.0];
loads = [0.02, 0.10];
for i = 1:2
    for il = 1:2
        id = sprintf('g%03d_p%03d_l%02d_40', round(gains(i)*100), ...
            round(pca_scales(i)*100), round(loads(il)*100));
        target = fullfile(target_dir, [id '.mat']);
        if exist(target, 'file'), continue; end
        fprintf('Running %s\n', id);
        a = mct2024_reverse_variant(loads(il), 40, init_file, ...
            gains(i), pca_scales(i));
        save(target, 'a', '-v7.3');
    end
end
