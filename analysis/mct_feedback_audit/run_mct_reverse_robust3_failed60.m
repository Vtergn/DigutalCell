% Diagnostic only: rerun conditions failing the frozen 40-beat convergence gate.
audit_dir = fileparts(mfilename('fullpath'));
project_dir = fileparts(fileparts(audit_dir));
native_dir = fullfile(project_dir, 'Data', 'MCT2024-MCT_v1.0.0', ...
    'UTokyo-FML-MCT2024-147ccd2');
addpath(audit_dir);
addpath(native_dir);
init_file = fullfile(native_dir, 'yfinal_2000.mat');
base_dir = fullfile(audit_dir, 'reverse_robust3');
output_dir = fullfile(base_dir, 'failed_60beat');
if ~exist(output_dir, 'dir'), mkdir(output_dir); end
tbl = readtable(fullfile(base_dir, 'design.csv'));
cohort = jsondecode(fileread(fullfile(base_dir, 'cohort_score.json')));
fail = cohort.convergence_failures;
for i = 1:numel(fail)
    id = char(string(fail(i).id));
    load = char(string(fail(i).load));
    cl = sscanf(load, 'l%d')/100;
    target = fullfile(output_dir, sprintf('%s_%s_60.mat', id, load));
    if exist(target, 'file'), continue; end
    j = find(strcmp(string(tbl.id), id), 1);
    fprintf('Running failed condition %s %s to 60 beats\n', id, load);
    a = mct2024_reverse_3param(cl, 60, init_file, ...
        tbl.mct_gain(j), tbl.pCa0_scale(j), tbl.serca_scale(j));
    save(target, 'a', '-v7.3');
end
