function human_bidirectional_pulse_audit(output_dir)
% Phase-locked 1 ms rise + 10 ms hold + 1 ms release stretch audit.
if nargin < 1 || isempty(output_dir)
    output_dir = fullfile(fileparts(mfilename('fullpath')), 'human_bidirectional_output');
end
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'Data','human_SAC_reference'));
addpath(fullfile(root,'Data','7fb','7fb-959277c48267','Land_model_matlab','release'));
s=load(fullfile(output_dir,'prepace_state.mat'),'state','par','isac_factor','beta');
opt=odeset('RelTol',1e-7,'AbsTol',1e-9,'MaxStep',.5);
rows={}; onsets=[150,300,600]; amplitudes=[1.05,1.10];
for onset=onsets
    for lambda=amplitudes
        for trpn=[false,true]
            for sac=[false,true]
                % Match every challenge to the same pathway configuration at lambda=1.
                control=run_protocol(s.state,1,onset,trpn,sac,s,opt);
                y=run_protocol(s.state,lambda,onset,trpn,sac,s,opt);
                dv=y(:,1)-control(:,1); dc=1000*(y(:,6)-control(:,6));
                apd_delta=threshold_apd(y(:,1),-67)-threshold_apd(control(:,1),-67);
                ca_peak_delta=1000*(max(y(:,6))-max(control(:,6)));
                upcross=find(y(1:end-1,1)<0 & y(2:end,1)>=0);
                upcross=upcross(upcross>onset+13 & upcross<990);
                ectopic=false;
                for ix=upcross(:).'
                    if min(y(max(1,ix-30):ix,1)) < -60, ectopic=true; end
                end
                rows(end+1,:)={onset,lambda,trpn,sac,apd_delta, ...
                    min(dv),max(dv),min(dc),max(dc),ca_peak_delta,max(1000*y(:,6)), ...
                    max(y(:,46)),min(y(:,6)),ectopic}; %#ok<AGROW>
                name=sprintf('pulse_t%03d_lam%03d_trpn%d_sac%d.csv', ...
                    onset,round(100*lambda),trpn,sac);
                tt=(0:1000).';
                writetable(table(tt,y(:,1),1000*y(:,6),y(:,46), ...
                    'VariableNames',{'time_ms','Vm_mV','Cai_uM','TRPN_fraction'}), ...
                    fullfile(output_dir,name));
            end
        end
    end
end
summary=cell2table(rows,'VariableNames',{'onset_ms','lambda','TnC_feedback', ...
    'SAC_module','delta_APD_ms','min_delta_Vm_mV','max_delta_Vm_mV', ...
    'min_delta_Ca_uM','max_delta_Ca_uM','delta_Ca_peak_uM','Ca_peak_uM', ...
    'TRPN_peak_fraction','min_Cai_mM','extra_upstroke'});
writetable(summary,fullfile(output_dir,'pulse_audit.csv'));
disp(summary);
end

function states=run_protocol(initial,lambda,onset,trpn,sac,s,opt)
ramp=1; hold_ms=10; finish=onset+2*ramp+hold_ms;
breaks=[0,onset,onset+ramp,onset+ramp+hold_ms,finish,1000];
state=initial; all_t=[]; all_y=[];
for k=1:numel(breaks)-1
    a=breaks(k); b=breaks(k+1);
    fun=@(t) length_schedule(t,lambda,onset,ramp,hold_ms);
    [t,y]=ode15s(@(t,y) human_bidirectional_rhs(t,y,fun,trpn,sac, ...
        s.par,s.isac_factor,s.beta),[a,b],state,opt);
    state=y(end,:).';
    if ~isempty(all_t), t=t(2:end); y=y(2:end,:); end
    all_t=[all_t;t]; all_y=[all_y;y]; %#ok<AGROW>
end
states=interp1(all_t,all_y,(0:1000).','pchip');
end

function z=length_schedule(t,lambda,onset,ramp,hold_ms)
delta=lambda-1; start=onset; plateau_end=onset+ramp+hold_ms;
if t<start || t>=plateau_end+ramp
    z=[1,0];
elseif t<start+ramp
    z=[1+delta*(t-start)/ramp,delta/ramp];
elseif t<plateau_end
    z=[lambda,0];
else
    z=[lambda-delta*(t-plateau_end)/ramp,-delta/ramp];
end
end

function duration=threshold_apd(vm,threshold)
up=find(vm(1:end-1)<=threshold & vm(2:end)>threshold,1,'first');
if isempty(up), duration=NaN; return; end
down=find(vm(up+1:end-1)>=threshold & vm(up+2:end)<threshold,1,'first');
if isempty(down), duration=NaN; return; end
down=down+up;
tup=up+(threshold-vm(up))/(vm(up+1)-vm(up));
tdown=down+(threshold-vm(down))/(vm(down+1)-vm(down));
duration=tdown-tup;
end
