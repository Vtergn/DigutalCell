function human_bidirectional_minimal(output_dir, prepace_beats)
% Minimal human ventricular electromechanical closed loop.
% Published ToR-ORd+SAC RHS (Buonocunto) + Land2016 RHS. The original SAC
% model's equilibrium troponin buffer is replaced by Land's dynamic TRPN
% binding flux following the coupling structure of Margara et al.
% Constant imposed lambda is an isometric length challenge, not force input.
if nargin < 1 || isempty(output_dir)
    output_dir = fullfile(fileparts(mfilename('fullpath')), 'human_bidirectional_output');
end
if nargin < 2 || isempty(prepace_beats), prepace_beats = 500; end
if ~exist(output_dir, 'dir'), mkdir(output_dir); end
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'Data','human_SAC_reference'));
addpath(fullfile(root,'Data','7fb','7fb-959277c48267','Land_model_matlab','release'));
source = load(fullfile(fileparts(mfilename('fullpath')), ...
    'human_sac_reference_pilot','prepace500_state.mat'),'state');
par = [0.2203,10.3203,12.8208,1.4110,0.8742];
isac_factor = [4.47,0.001248];
beta = [2.3,-2.4];
initial = [source.state(:); Land2016().'];
options = odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',2);
state = initial;
metric = nan(prepace_beats,4);
for beat = 1:prepace_beats
    [time,states] = ode15s(@(t,y) human_bidirectional_rhs(t,y,1,true,true,par, ...
        isac_factor,beta),[0 1000],state,options);
    state = states(end,:).';
    grid = (0:999).';
    sampled = interp1(time,states,grid,'pchip');
    metric(beat,:) = [beat,sum(sampled(:,1)>-67), ...
        1000*max(sampled(:,6)),min(sampled(:,6))];
    if mod(beat,100)==0
        fprintf('Coupled prepace %d/%d: APD %g ms, Ca peak %.5g uM\n', ...
            beat,prepace_beats,metric(beat,2),metric(beat,3));
    end
end
writematrix(metric,fullfile(output_dir,'prepace_metrics.csv'));
save(fullfile(output_dir,'prepace_state.mat'),'state','metric','par', ...
    'isac_factor','beta','-v7.3');
if prepace_beats >= 10
    tail = metric(end-9:end,:);
    if range(tail(:,2))>1 || range(tail(:,3))/mean(tail(:,3))>.002
        warning('Coupled prepace may not be converged: inspect metrics');
    end
end

label = strings(0); lam_col=[]; apd=[]; ca_peak=[]; tension_peak=[];
for trpn_on = [false,true]
    for sac_on = [false,true]
        for lam = [1.00,1.05,1.10]
            [time,states] = ode15s(@(t,y) human_bidirectional_rhs(t,y,lam, ...
                trpn_on,sac_on,par,isac_factor,beta),[0 1000],state,options);
            grid = (0:1000).';
            sampled = interp1(time,states,grid,'pchip');
            vm = sampled(:,1); ca = 1000*sampled(:,6);
            ta = zeros(numel(grid),1); tp = ta;
            for k=1:numel(grid)
                [~,~,ta(k),tp(k)] = Land2016(grid(k), ...
                    sampled(k,44:50),ca(k),'intact',lam,0,beta);
            end
            key = sprintf('trpn%d_sac%d_lambda%03d',trpn_on,sac_on,round(100*lam));
            trace = table(grid,vm,ca,ta,tp,ta+tp, ...
                'VariableNames',{'time_ms','Vm_mV','Cai_uM','Ta_kPa', ...
                'Tp_kPa','Ttotal_kPa'});
            writetable(trace,fullfile(output_dir,[key '.csv']));
            label(end+1,1) = string(key); %#ok<AGROW>
            lam_col(end+1,1) = lam; %#ok<AGROW>
            apd(end+1,1) = sum(vm(1:1000)>-67); %#ok<AGROW>
            ca_peak(end+1,1) = max(ca); %#ok<AGROW>
            tension_peak(end+1,1) = max(ta); %#ok<AGROW>
        end
    end
end
summary = table(label,lam_col,apd,ca_peak,tension_peak, ...
    'VariableNames',{'condition','lambda','APD_above_minus67_ms', ...
    'Cai_peak_uM','Ta_peak_kPa'});
writetable(summary,fullfile(output_dir,'summary.csv'));
disp(summary);
end
