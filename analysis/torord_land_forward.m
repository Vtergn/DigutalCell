function torord_land_forward(output_dir, nbeats, stretches, cell_parameters, land_beta)
% Run the original ToR-ORd equations and Land2016 contraction equations.
% This is one-way Ca-to-tension coupling, with constant isometric stretch.
% The same electrical parameters and Ca waveform are used at every stretch.
if nargin < 1 || isempty(output_dir)
    output_dir = fullfile(fileparts(mfilename('fullpath')), 'torord_land_forward_output');
end
if nargin < 2 || isempty(nbeats)
    nbeats = 3;
end
if nargin < 3 || isempty(stretches)
    stretches = [1.000, 1.025, 1.050, 1.075, 1.100];
end
if nargin < 4 || isempty(cell_parameters)
    cell_parameters = struct();
end
if nargin < 5 || isempty(land_beta)
    land_beta = [2.3, -2.4]; % Land2016 defaults: beta0, beta1
end
if any(stretches < 1) || any(stretches > 1.2) || any(diff(stretches) <= 0)
    error('stretches must increase strictly within [1, 1.2]');
end
if numel(land_beta) ~= 2
    error('land_beta must contain [beta0, beta1]');
end
if ~exist(output_dir, 'dir'), mkdir(output_dir); end
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'Data', 'torord_official', 'matlab'));
addpath(fullfile(root, 'Data', '7fb', '7fb-959277c48267', 'Land_model_matlab', 'release'));

params.bcl = 1000;
params.model = @model_Torord;
params.verbose = false;
params.cellType = 0;
names = fieldnames(cell_parameters);
for j = 1:numel(names)
    params.(names{j}) = cell_parameters.(names{j});
end
params.model = @model_Torord;
initial = getStartingState('Torord_endo');
[time, states] = modelRunner(initial, [], params, nbeats, nbeats-1);
t_raw = time{1};
x_raw = states{1};
t = (0:params.bcl)';
vm = interp1(t_raw, x_raw(:,1), t, 'pchip');
ca_mM = interp1(t_raw, x_raw(:,6), t, 'pchip');
ca_uM = 1000 * ca_mM;
if any(~isfinite(vm)) || any(~isfinite(ca_uM)) || any(ca_uM <= 0)
    error('Invalid electrical or calcium trajectory');
end

trace = table(t, vm, ca_mM, ca_uM, 'VariableNames', ...
    {'time_ms','Vm_mV','Cai_mM','Cai_uM'});
summary = table('Size', [numel(stretches), 11], ...
    'VariableTypes', repmat({'double'}, 1, 11), ...
    'VariableNames', {'lambda','Vrest_mV','Vpeak_mV','Cai_rest_uM', ...
    'Cai_peak_uM','Cai_peak_time_ms','Tend_total_kPa','Ta_end_kPa','Tpeak_kPa', ...
    'Ta_peak_kPa','Tpeak_time_ms'});

for j = 1:numel(stretches)
    lam = stretches(j);
    % Use periodic Cai from the converged electrical beat. Land's own state
    % is conditioned over four beats and only the final beat is exported.
    cai_at = @(s) interp1(t, ca_uM, min(params.bcl, mod(s,params.bcl)), 'pchip');
    land_rhs = @(s,y) Land2016(s,y,cai_at(s),'intact',lam,0,land_beta);
    [land_t,land_y] = ode15s(land_rhs, [0, 4*params.bcl], Land2016());
    sample_t = 3*params.bcl + t;
    sample_y = interp1(land_t, land_y, sample_t, 'pchip');
    tension = zeros(size(t)); active = tension; passive = tension;
    for k = 1:numel(t)
        [~,tension(k),active(k),passive(k)] = ...
            Land2016(sample_t(k),sample_y(k,:),ca_uM(k),'intact',lam,0,land_beta);
    end
    if any(~isfinite(tension)) || any(tension < -1e-6)
        error('Invalid Land tension at lambda %.3f', lam);
    end
    col = strrep(sprintf('%.3f',lam),'.','p');
    trace.(['T_' col '_kPa']) = tension;
    trace.(['Ta_' col '_kPa']) = active;
    trace.(['Tp_' col '_kPa']) = passive;
    summary.lambda(j) = lam;
    summary.Vrest_mV(j) = min(vm);
    summary.Vpeak_mV(j) = max(vm);
    summary.Cai_rest_uM(j) = min(ca_uM);
    [summary.Cai_peak_uM(j), idx_ca] = max(ca_uM);
    summary.Cai_peak_time_ms(j) = t(idx_ca);
    summary.Tend_total_kPa(j) = tension(end);
    summary.Ta_end_kPa(j) = active(end);
    [summary.Tpeak_kPa(j), idx_t] = max(tension);
    summary.Tpeak_time_ms(j) = t(idx_t);
    summary.Ta_peak_kPa(j) = max(active);
end
writetable(trace, fullfile(output_dir,'traces.csv'));
writetable(summary, fullfile(output_dir,'summary.csv'));
if any(diff(summary.Ta_peak_kPa) <= 0)
    warning('Active tension did not increase with stretch; inspect inputs');
end
fig = figure('Visible','off','Position',[100 100 1050 750]);
tiledlayout(3,1);
nexttile; plot(t,vm,'k'); ylabel('V_m (mV)'); xlim([0 600]);
nexttile; plot(t,ca_uM,'k'); ylabel('Ca_i (uM)'); xlim([0 600]);
nexttile; hold on;
for j = 1:numel(stretches)
    col = strrep(sprintf('%.3f',stretches(j)),'.','p');
    plot(t,trace.(['Ta_' col '_kPa']),'DisplayName',sprintf('lambda %.3f',stretches(j)));
end
ylabel('Active tension (kPa)'); xlabel('Time (ms)'); xlim([0 600]); legend('Location','best');
exportgraphics(fig,fullfile(output_dir,'ap_ca_active_tension.png'),'Resolution',150);
close(fig);
metadata = struct('model','ToR-ORd endocardial + Land2016 intact', ...
    'coupling','one-way Ca to tension','bcl_ms',params.bcl, ...
    'electrical_prepace_beats',nbeats,'land_conditioning_beats',4, ...
    'stretches',stretches,'cell_parameters',cell_parameters, ...
    'land_beta',land_beta,'Cai_conversion','mM x 1000 = uM');
fid = fopen(fullfile(output_dir,'metadata.json'),'w');
fprintf(fid,'%s',jsonencode(metadata,PrettyPrint=true));
fclose(fid);
disp(summary);
end
