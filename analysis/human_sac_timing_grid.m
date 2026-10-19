function human_sac_timing_grid(output_dir)
% Acute stretch timing grid using the author's human ToR-ORd+SAC equations.
if nargin < 1 || isempty(output_dir)
    output_dir = fullfile(fileparts(mfilename('fullpath')), 'human_sac_reference_pilot');
end
addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'Data', 'human_SAC_reference'));
s = load(fullfile(output_dir, 'prepace_state.mat'), 'state', 'par', 'isac_factor', 'converged');
if ~s.converged, error('SAC-specific prepace criterion not met'); end
options = odeset('RelTol', 1e-6, 'AbsTol', 1e-8, 'MaxStep', 1);
for onset = [150, 300, 600]
    offset = onset + 10;
    for lam = [1.00, 1.05, 1.10]
        target = fullfile(output_dir, sprintf('lambda_%03d_onset_%03d_trace.csv', ...
            round(100*lam), onset));
        if exist(target, 'file'), continue; end
        t_all = []; y_all = []; current = s.state;
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
        writetable(out, target);
    end
end
end
