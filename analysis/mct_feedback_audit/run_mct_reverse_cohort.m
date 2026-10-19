% Controlled 3x3 parameter-grid pilot. These are simulated variants, not cells.
audit_dir = fileparts(mfilename('fullpath'));
project_dir = fileparts(fileparts(audit_dir));
native_dir = fullfile(project_dir, 'Data', 'MCT2024-MCT_v1.0.0', ...
    'UTokyo-FML-MCT2024-147ccd2');
addpath(audit_dir);
addpath(native_dir);
init_file = fullfile(native_dir, 'yfinal_2000.mat');
target_dir = fullfile(audit_dir, 'reverse_cohort');
if ~exist(target_dir, 'dir'), mkdir(target_dir); end
gains = [0.25, 0.75, 1.25];
pca_scales = [0.9, 1.0, 1.1];
loads = [0.02, 0.10];
for ig = 1:numel(gains)
    for ip = 1:numel(pca_scales)
        for il = 1:numel(loads)
            id = sprintf('g%03d_p%03d_l%02d', round(gains(ig)*100), ...
                round(pca_scales(ip)*100), round(loads(il)*100));
            target = fullfile(target_dir, [id '.mat']);
            if exist(target, 'file'), continue; end
            fprintf('Running %s\n', id);
            a = mct2024_reverse_variant(loads(il), 20, init_file, ...
                gains(ig), pca_scales(ip));
            save(target, 'a', '-v7.3');
        end
    end
end
