function human_sac_prepacing500_audit(output_dir)
% Check that the acute human SAC findings survive the published 500-beat prepace.
if nargin < 1 || isempty(output_dir)
    output_dir = fullfile(fileparts(mfilename('fullpath')), 'human_sac_reference_pilot');
end
project_dir = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(project_dir, 'Data', 'human_SAC_reference'));
s = load(fullfile(output_dir, 'prepace_state.mat'), 'state', 'metric', 'par', 'isac_factor');
state = s.state;
metric = nan(500,3);
metric(1:size(s.metric,1),:) = s.metric;
options = odeset('RelTol', 1e-6, 'AbsTol', 1e-8, 'MaxStep', 1);
for beat = size(s.metric,1)+1:500
    [t, y] = ode15s(@model_SAC, [0, 1000], state, options, ...
        1, 1, s.par, s.isac_factor, 0, 0, 0);
    state = y(end,:).';
    voltage = interp1(t, y(:,1), 0:999, 'pchip');
    metric(beat,:) = [beat, sum(voltage > -67), 1000*max(y(:,6))];
end
writematrix(metric, fullfile(output_dir, 'prepace500_metrics.csv'));
save(fullfile(output_dir, 'prepace500_state.mat'), 'state', 'metric', '-v7.3');
fprintf('Beat 120/500 APD: %g/%g ms; Ca peak: %.6g/%.6g uM\n', ...
    metric(size(s.metric,1),2), metric(500,2), ...
    metric(size(s.metric,1),3), metric(500,3));
for onset = [150, 600]
    offset = onset + 10;
    for lam = [1.00, 1.10]
        t_all = []; y_all = []; current = state;
        for bounds = [0 onset; onset offset; offset 1000].'
            [t_seg, y_seg] = ode15s(@model_SAC, bounds.', current, options, ...
                1, lam, s.par, s.isac_factor, onset, offset, 0);
            current = y_seg(end,:).';
            if ~isempty(t_all)
                t_seg = t_seg(2:end); y_seg = y_seg(2:end,:);
            end
            t_all = [t_all; t_seg]; %#ok<AGROW>
            y_all = [y_all; y_seg]; %#ok<AGROW>
        end
        out = table(t_all, y_all(:,1), 1000*y_all(:,6), ...
            'VariableNames', {'time_ms','Vm_mV','Cai_uM'});
        writetable(out, fullfile(output_dir, sprintf( ...
            'prepace500_lambda_%03d_onset_%03d_trace.csv', round(100*lam), onset)));
    end
end
end
