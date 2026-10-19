function human_bidirectional_strict_audit(output_dir)
% Recompute the complete-feedback challenges with stricter ODE tolerances.
if nargin < 1 || isempty(output_dir)
    output_dir = fullfile(fileparts(mfilename('fullpath')), 'human_bidirectional_output');
end
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'Data','human_SAC_reference'));
addpath(fullfile(root,'Data','7fb','7fb-959277c48267','Land_model_matlab','release'));
s = load(fullfile(output_dir,'prepace_state.mat'),'state','par','isac_factor','beta');
options = odeset('RelTol',1e-8,'AbsTol',1e-10,'MaxStep',.5);
grid = (0:1000).';
rows = zeros(3,5);
for j=1:3
    lam = [1,1.05,1.1]; lam = lam(j);
    [time,states] = ode15s(@(t,y) human_bidirectional_rhs(t,y,lam, ...
        true,true,s.par,s.isac_factor,s.beta),[0 1000],s.state,options);
    sampled = interp1(time,states,grid,'pchip');
    raw = readtable(fullfile(output_dir, ...
        sprintf('trpn1_sac1_lambda%03d.csv',round(100*lam))));
    dvm = max(abs(sampled(:,1)-raw.Vm_mV));
    dca = max(abs(1000*sampled(:,6)-raw.Cai_uM));
    apd = sum(sampled(1:1000,1)>-67);
    rows(j,:) = [lam,apd,dvm,dca,1000*max(sampled(:,6))];
end
result = array2table(rows,'VariableNames', ...
    {'lambda','APD_above_minus67_ms','max_abs_delta_vs_default_Vm_mV', ...
    'max_abs_delta_vs_default_Ca_uM','Ca_peak_uM'});
writetable(result,fullfile(output_dir,'strict_solver_audit.csv'));
disp(result);
end
