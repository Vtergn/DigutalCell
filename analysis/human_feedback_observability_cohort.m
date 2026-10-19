function human_feedback_observability_cohort(ncells, output_dir)
% Controlled proof of principle: paired human closed-loop pulse challenges.
% Cell-specific effective ionic stretch gain, Land beta1 and Tref scale.
% All three share the same paced state at lambda=1; the first two are
% structurally invisible at rest in this implementation.
if nargin<1 || isempty(ncells), ncells=48; end
if nargin<2 || isempty(output_dir)
    output_dir=fullfile(fileparts(mfilename('fullpath')),'human_bidirectional_output');
end
if ~exist(output_dir,'dir'), mkdir(output_dir); end
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'Data','human_SAC_reference'));
addpath(fullfile(root,'Data','7fb','7fb-959277c48267','Land_model_matlab','release'));
s=load(fullfile(output_dir,'prepace_state.mat'),'state','par','isac_factor','beta');
rng(20261001);
% Stratified independent draws reduce accidental correlations in this pilot.
u=zeros(ncells,3);
for j=1:3
    u(:,j)=((randperm(ncells)'-1)+rand(ncells,1))/ncells;
end
g_ionic=0.25+1.50*u(:,1);
beta1=-4.5+4.0*u(:,2);
tref_scale=0.65+0.70*u(:,3);
params=[g_ionic,beta1,tref_scale];
% Early pulse, late pulse, and unseen intermediate-phase/lower-amplitude test.
onsets=[150,600,225]; lambdas=[1.10,1.10,1.075];
grid=(0:1000)'; ncond=3;
waveforms=zeros(numel(grid),3,ncond,ncells,'single');
features=zeros(ncells,ncond,12); baseline=zeros(ncells,3);
opt=odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',1);
control=simulate(s.state,1,150,s.beta,struct('g_ionic',1),s,opt,grid);
control_apd=threshold_apd(control(:,1),-67);
control_ca=1000*control(:,6);
base_ta=zeros(numel(grid),1);
for k=1:numel(grid)
    [~,~,base_ta(k)]=Land2016(grid(k),control(k,44:50), ...
        control_ca(k),'intact',1,0,s.beta);
end
for i=1:ncells
    beta=[s.beta(1),beta1(i)]; feedback=struct('g_ionic',g_ionic(i));
    baseline(i,:)=[control_apd,max(control_ca),tref_scale(i)*max(base_ta)];
    for c=1:ncond
        onset=onsets(c); lambda=lambdas(c);
        y=simulate(s.state,lambda,onset,beta,feedback,s,opt,grid);
        vm=y(:,1); ca=1000*y(:,6);
        ta=zeros(numel(grid),1);
        for k=1:numel(grid)
            mech=length_schedule(grid(k),lambda,onset,1,10);
            [~,~,ta(k)]=Land2016(grid(k),y(k,44:50),ca(k), ...
                'intact',mech(1),mech(2),beta);
        end
        ta=tref_scale(i)*ta;
        waveforms(:,:,c,i)=single([vm,ca,ta]);
        win=(grid>=onset & grid<=min(1000,onset+150));
        dv=vm(win)-control(win,1); dc=ca(win)-control_ca(win);
        t20=min(1001,onset+21); t50=min(1001,onset+51);
        features(i,c,:)=[threshold_apd(vm,-67)-control_apd, ...
            min(dv),max(dv),min(dc),max(dc), ...
            max(ta),ta(t20),ta(t50), ...
            vm(t20)-control(t20,1),vm(t50)-control(t50,1), ...
            ca(t20)-control_ca(t20),ca(t50)-control_ca(t50)];
    end
    if mod(i,8)==0, fprintf('Cohort %d/%d cells complete\n',i,ncells); end
end
save(fullfile(output_dir,'feedback_observability_cohort.mat'), ...
    'params','baseline','features','waveforms','onsets','lambdas', ...
    'grid','control_apd','control_ca','base_ta','-v7.3');
writematrix([params,baseline,reshape(features(:,1,:),ncells,12), ...
    reshape(features(:,2,:),ncells,12),reshape(features(:,3,:),ncells,12)], ...
    fullfile(output_dir,'feedback_observability_cohort.csv'));
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
