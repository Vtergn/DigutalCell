function human_feedback_native_twin_check(output_dir)
% Blind 12-cell closure check: estimated parameters -> original 50-state ODE.
if nargin<1 || isempty(output_dir)
    output_dir=fullfile(fileparts(mfilename('fullpath')),'human_bidirectional_output');
end
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'Data','human_SAC_reference'));
addpath(fullfile(root,'Data','7fb','7fb-959277c48267','Land_model_matlab','release'));
s=load(fullfile(output_dir,'prepace_state.mat'),'state','par','isac_factor','beta');
input=readtable(fullfile(output_dir,'feedback_native_twin_input.csv'));
grid=(0:1000)'; opt=odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',1);
control=simulate(s.state,1,225,s.beta,struct('g_ionic',1),s,opt,grid);
control_apd=threshold_apd(control(:,1),-67);
control_ca=1000*control(:,6);
output=zeros(height(input),6);
for i=1:height(input)
    beta=[s.beta(1),input.beta1_est(i)];
    feedback=struct('g_ionic',input.g_est(i));
    y=simulate(s.state,1.075,225,beta,feedback,s,opt,grid);
    vm=y(:,1); ca=1000*y(:,6); onset=225;
    win=(grid>=onset & grid<=onset+150);
    dv=vm(win)-control(win,1); dc=ca(win)-control_ca(win);
    t20=onset+21;
    [~,~,ta20]=Land2016(grid(t20),y(t20,44:50),ca(t20), ...
        'intact',1,0,beta);
    output(i,:)=[threshold_apd(vm,-67)-control_apd, max(dv),min(dc), ...
        input.tref_est(i)*ta20,vm(t20)-control(t20,1), ...
        ca(t20)-control_ca(t20)];
end
writematrix([input.cell_index,output], ...
    fullfile(output_dir,'feedback_native_twin_predictions.csv'));
disp('Native ODE target predictions saved.');
end

function states=simulate(initial,lambda,onset,beta,feedback,s,opt,grid)
ramp=1; hold_ms=10; finish=onset+2*ramp+hold_ms;
breaks=[0,onset,onset+ramp,onset+ramp+hold_ms,finish,1000];
state=initial; all_t=[]; all_y=[];
for k=1:numel(breaks)-1
    a=breaks(k); b=breaks(k+1);
    fun=@(t) length_schedule(t,lambda,onset,ramp,hold_ms);
    [t,y]=ode15s(@(t,y) human_bidirectional_rhs(t,y,fun,true,true, ...
        s.par,s.isac_factor,beta,feedback),[a,b],state,opt);
    state=y(end,:)';
    if ~isempty(all_t), t=t(2:end); y=y(2:end,:); end
    all_t=[all_t;t]; all_y=[all_y;y]; %#ok<AGROW>
end
states=interp1(all_t,all_y,grid,'pchip');
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
