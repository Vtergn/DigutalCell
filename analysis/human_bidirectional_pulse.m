function human_bidirectional_pulse(output_dir)
% A short imposed length pulse on the calibrated human closed-loop state.
if nargin < 1 || isempty(output_dir)
    output_dir = fullfile(fileparts(mfilename('fullpath')), 'human_bidirectional_output');
end
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'Data','human_SAC_reference'));
addpath(fullfile(root,'Data','7fb','7fb-959277c48267','Land_model_matlab','release'));
s = load(fullfile(output_dir,'prepace_state.mat'),'state','par','isac_factor','beta');
options = odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',1);
grid = (0:1000).';
rows = zeros(0,6);
for onset = [150,600]
    ramp_ms = 5; hold_ms = 10; peak_lambda = 1.1;
    bounds = [0,onset,onset+ramp_ms,onset+ramp_ms+hold_ms, ...
        onset+2*ramp_ms+hold_ms,1000];
    for trpn_on = [false,true]
        for sac_on = [false,true]
            pulse = @(t) human_stretch_profile(t,onset,peak_lambda, ...
                ramp_ms,hold_ms);
            t_all=[]; y_all=[]; current=s.state;
            for j=1:numel(bounds)-1
                [t_seg,y_seg] = ode15s(@(t,y) human_bidirectional_rhs( ...
                    t,y,pulse,trpn_on,sac_on,s.par,s.isac_factor,s.beta), ...
                    bounds(j:j+1),current,options);
                current=y_seg(end,:).';
                if j>1
                    t_seg=t_seg(2:end); y_seg=y_seg(2:end,:);
                end
                t_all=[t_all;t_seg]; %#ok<AGROW>
                y_all=[y_all;y_seg]; %#ok<AGROW>
            end
            sampled=interp1(t_all,y_all,grid,'pchip');
            vm=sampled(:,1); ca=1000*sampled(:,6);
            ta=zeros(numel(grid),1); tp=ta; lam=ta;
            for k=1:numel(grid)
                mechanical=pulse(grid(k)); lam(k)=mechanical(1);
                [~,~,ta(k),tp(k)] = Land2016(grid(k), ...
                    sampled(k,44:50),ca(k),'intact',lam(k), ...
                    mechanical(2),s.beta);
            end
            label=sprintf('pulse_onset%03d_trpn%d_sac%d',onset,trpn_on,sac_on);
            trace=table(grid,lam,vm,ca,ta,tp,ta+tp, ...
                'VariableNames',{'time_ms','lambda','Vm_mV','Cai_uM', ...
                'Ta_kPa','Tp_kPa','Ttotal_kPa'});
            writetable(trace,fullfile(output_dir,[label,'.csv']));
            rows(end+1,:)=[onset,trpn_on,sac_on, ...
                sum(vm(1:1000)>-67),max(ca),max(ta)]; %#ok<AGROW>
        end
    end
end
summary=array2table(rows,'VariableNames',{'onset_ms','trpn_feedback', ...
    'SAC_feedback','APD_above_minus67_ms','Ca_peak_uM','Ta_peak_kPa'});
writetable(summary,fullfile(output_dir,'pulse_summary.csv'));
disp(summary);
end
