function human_sac_reference_pilot(output_dir)
% Small, read-only-reference human ToR-ORd+SAC mechanistic pilot.
% Uses published SAC current parameters, not rabbit MCT gains. Acute imposed
% stretch drives SACs; Land contraction is not yet coupled in this run.
if nargin < 1 || isempty(output_dir)
    output_dir = fullfile(fileparts(mfilename('fullpath')), 'human_sac_reference_pilot');
end
if ~exist(output_dir, 'dir'), mkdir(output_dir); end
project_dir = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(project_dir, 'Data', 'torord_official', 'matlab'));
addpath(fullfile(project_dir, 'Data', 'human_SAC_reference'));

par = [0.2203, 10.3203, 12.8208, 1.4110, 0.8742];
isac_factor = [4.47, 0.001248];
bcl = 1000;
initial = getStartingState('Torord_endo');
state = initial(:);
options = odeset('RelTol', 1e-6, 'AbsTol', 1e-8, 'MaxStep', 1);
metric = nan(500,3);  % beat, APD-above-67, Ca peak uM
converged = false;
for beat = 1:500
    [t, y] = ode15s(@model_SAC, [0, bcl], state, options, ...
        1, 1, par, isac_factor, 0, 0, 0);
    state = y(end,:).';
    grid = 0:999;
    voltage = interp1(t, y(:,1), grid, 'pchip');
    metric(beat,:) = [beat, sum(voltage > -67), 1000*max(y(:,6))];
    if beat >= 100 && mod(beat,20) == 0
        recent = metric(beat-9:beat,:);
        apd_range = max(recent(:,2)) - min(recent(:,2));
        ca_relative_range = (max(recent(:,3)) - min(recent(:,3))) / mean(recent(:,3));
        if apd_range <= 1 && ca_relative_range <= 0.002
            converged = true;
            break;
        end
    end
end
metric = metric(1:beat,:);
writematrix(metric, fullfile(output_dir, 'prepace_metrics.csv'));
save(fullfile(output_dir, 'prepace_state.mat'), 'state', 'metric', ...
    'par', 'isac_factor', 'converged', '-v7.3');
fprintf('Prepace beats: %d; convergence criterion met: %d\n', beat, converged);

if ~converged
    warning('Prepace convergence criterion not met; challenge outputs exploratory only');
end
for lam = [1.00, 1.05, 1.10]
    t_all = []; y_all = []; current = state;
    for bounds = [0 600; 600 610; 610 bcl].'
        [t_seg, y_seg] = ode15s(@model_SAC, bounds.', current, options, ...
            1, lam, par, isac_factor, 600, 610, 0);
        current = y_seg(end,:).';
        if ~isempty(t_all)
            t_seg = t_seg(2:end); y_seg = y_seg(2:end,:);
        end
        t_all = [t_all; t_seg]; %#ok<AGROW>
        y_all = [y_all; y_seg]; %#ok<AGROW>
    end
    out = table(t_all, y_all(:,1), 1000*y_all(:,6), ...
        'VariableNames', {'time_ms','Vm_mV','Cai_uM'});
    writetable(out, fullfile(output_dir, sprintf('lambda_%03d_trace.csv', round(100*lam))));
end
metadata = struct('model', 'Buonocunto ToR-ORd+SAC, author source', ...
    'source', 'https://github.com/melaniabuonoc/Electrophysiological-effects-of-stretch-activated-ion-channels', ...
    'prepace_beats', beat, 'convergence_criterion_met', converged, ...
    'prepace_last10_apd_range_ms', max(metric(end-9:end,2))-min(metric(end-9:end,2)), ...
    'prepace_last10_ca_relative_range', (max(metric(end-9:end,3))-min(metric(end-9:end,3)))/mean(metric(end-9:end,3)), ...
    'lambda', [1.00,1.05,1.10], 'stretch_onset_ms', 600, 'stretch_offset_ms', 610, ...
    'par', par, 'isac_factor', isac_factor, ...
    'note', 'Published human-ventricular ionic SAC module; no Land feedback or human-cell calibration');
fid = fopen(fullfile(output_dir, 'metadata.json'), 'w');
fprintf(fid, '%s', jsonencode(metadata, PrettyPrint=true));
fclose(fid);
end
