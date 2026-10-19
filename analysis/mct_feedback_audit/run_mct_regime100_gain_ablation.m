function run_mct_regime100_gain_ablation(id)
% Exploratory ablation with frozen labels; see mct_regime100_gain_ablation_protocol.md.
audit_dir = fileparts(mfilename('fullpath'));
project_dir = fileparts(fileparts(audit_dir));
native_dir = fullfile(project_dir, 'Data', 'MCT2024-MCT_v1.0.0', ...
    'UTokyo-FML-MCT2024-147ccd2');
addpath(audit_dir);
addpath(native_dir);
init_file = fullfile(native_dir, 'yfinal_2000.mat');
base_dir = fullfile(audit_dir, 'reverse_robust3');
out_dir = fullfile(base_dir, 'regime100', 'gain_ablation');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
tbl = readtable(fullfile(base_dir, 'design.csv'));
i = find(strcmp(string(tbl.id), string(id)), 1);
if isempty(i), error('Unknown id: %s', string(id)); end
for gain = [0, 0.7]
    target = fullfile(out_dir, sprintf('%s_gain%02d_l10_100.mat', ...
        char(string(id)), round(gain * 10)));
    if exist(target, 'file'), continue; end
    fprintf('Ablation %s gain %g\n', string(id), gain);
    a = mct2024_reverse_3param(0.10, 100, init_file, ...
        gain, tbl.pCa0_scale(i), tbl.serca_scale(i));
    save(target, 'a', '-v7.3');
end
end
