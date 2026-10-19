function human_feedback_hetero_state_drift_audit(output_dir)
% Effect of 100 extra pre-paced beats on the *challenge response* itself.
if nargin<1 || isempty(output_dir)
    output_dir=fullfile(fileparts(mfilename('fullpath')),'human_bidirectional_output');
end
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'Data','human_SAC_reference'));
addpath(fullfile(root,'Data','7fb','7fb-959277c48267','Land_model_matlab','release'));
s=load(fullfile(output_dir,'prepace_state.mat'),'par','isac_factor','beta');
c=load(fullfile(output_dir,'feedback_hetero_prepacing_pilot.mat'),'theta');
l=load(fullfile(output_dir,'feedback_hetero_longpace_audit.mat'),'ix','states');
f=load(fullfile(output_dir,'feedback_hetero_challenges.mat'),'features');
grid=(0:1000)'; opt=odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',1);
rows=zeros(numel(l.ix),9);
for j=1:numel(l.ix)
    i=l.ix(j); th=c.theta(i,:); beta=[s.beta(1),th(5)];
    fb=struct('g_ionic',th(4),'ionic_scale',th(1:3));
    base=simulate(l.states(:,j),1,225,beta,fb,s,opt,grid);
    y=simulate(l.states(:,j),1.075,225,beta,fb,s,opt,grid);
    win=grid>=225 & grid<=375; t20=246;
    v=y(:,1); ca=1000*y(:,6);
    v0=base(:,1); ca0=1000*base(:,6);
    [~,~,ta20]=Land2016(grid(t20),y(t20,44:50),ca(t20), ...
        'intact',1,0,beta);
    new=[threshold_apd(v,-67)-threshold_apd(v0,-67), ...
        max(v(win)-v0(win)),min(ca(win)-ca0(win)),th(6)*ta20];
    old=squeeze(f.features(i,3,[1,3,4,7]))';
    rows(j,:)=[i,old,new];
    fprintf('cell %d challenge delta APD %.4f -> %.4f, Ca %.5f -> %.5f\n', ...
        i,old(1),new(1),old(3),new(3));
end
writematrix(rows,fullfile(output_dir,'feedback_hetero_state_drift_audit.csv'));
end

function states=simulate(initial,lambda,onset,beta,fb,s,opt,grid)
ramp=1; hold_ms=10; finish=onset+2*ramp+hold_ms;
breaks=[0,onset,onset+ramp,onset+ramp+hold_ms,finish,1000];
state=initial; all_t=[]; all_y=[];
for k=1:numel(breaks)-1
    a=breaks(k); b=breaks(k+1);
    fun=@(t) length_schedule(t,lambda,onset,ramp,hold_ms);
    [t,y]=ode15s(@(t,y) human_bidirectional_rhs(t,y,fun,true,true, ...
        s.par,s.isac_factor,beta,fb),[a,b],state,opt);
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
