function run_mct_reverse_regime100_tight(id)
% Re-run a pre-labeled period-2 candidate with stricter ode15s settings.
audit_dir = fileparts(mfilename('fullpath'));
project_dir = fileparts(fileparts(audit_dir));
native_dir = fullfile(project_dir, 'Data', 'MCT2024-MCT_v1.0.0', ...
    'UTokyo-FML-MCT2024-147ccd2');
addpath(audit_dir);
addpath(native_dir);
init_file = fullfile(native_dir, 'yfinal_2000.mat');
base_dir = fullfile(audit_dir, 'reverse_robust3');
out_dir = fullfile(base_dir, 'regime100', 'tight');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
tbl = readtable(fullfile(base_dir, 'design.csv'));
i = find(strcmp(string(tbl.id), string(id)), 1);
if isempty(i), error('Unknown cohort id: %s', string(id)); end
target = fullfile(out_dir, sprintf('%s_l10_100_tight.mat', char(string(id))));
if exist(target, 'file'), return; end
a = mct2024_reverse_3param_tight(0.10, 100, init_file, ...
    tbl.mct_gain(i), tbl.pCa0_scale(i), tbl.serca_scale(i));
save(target, 'a', '-v7.3');
end
