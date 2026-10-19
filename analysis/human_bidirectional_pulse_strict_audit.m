function human_bidirectional_pulse_strict_audit(output_dir)
% Tight-tolerance repeats of the most consequential brief-pulse cases.
if nargin<1 || isempty(output_dir)
    output_dir=fullfile(fileparts(mfilename('fullpath')),'human_bidirectional_output');
end
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'Data','human_SAC_reference'));
addpath(fullfile(root,'Data','7fb','7fb-959277c48267','Land_model_matlab','release'));
s=load(fullfile(output_dir,'prepace_state.mat'),'state','par','isac_factor','beta');
opt=odeset('RelTol',1e-9,'AbsTol',1e-11,'MaxStep',.1);
rows=zeros(4,7); idx=0;
for onset=[150,600]
    for trpn=[false,true]
        idx=idx+1;
        ctl=run_one(s.state,1,onset,trpn,true,s,opt);
        y=run_one(s.state,1.1,onset,trpn,true,s,opt);
        dv=y(:,1)-ctl(:,1); dc=1000*(y(:,6)-ctl(:,6));
        apd=threshold_apd(y(:,1),-67)-threshold_apd(ctl(:,1),-67);
        rows(idx,:)=[onset,trpn,apd,min(dv),max(dv),min(dc),max(dc)];
    end
end
result=array2table(rows,'VariableNames',{'onset_ms','TnC_feedback', ...
    'delta_APD_ms','min_delta_Vm_mV','max_delta_Vm_mV', ...
    'min_delta_Ca_uM','max_delta_Ca_uM'});
writetable(result,fullfile(output_dir,'pulse_strict_audit.csv'));
disp(result);
end

function states=run_one(initial,lambda,onset,trpn,sac,s,opt)
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
delta=lambda-1; plateau_end=onset+ramp+hold_ms;
if t<onset || t>=plateau_end+ramp
    z=[1,0];
elseif t<onset+ramp
    z=[1+delta*(t-onset)/ramp,delta/ramp];
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
