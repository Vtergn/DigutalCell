function human_feedback_hetero_challenges(output_dir)
% Paired mechanical challenges after per-cell electrical pre-pacing.
if nargin<1 || isempty(output_dir)
    output_dir=fullfile(fileparts(mfilename('fullpath')),'human_bidirectional_output');
end
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'Data','human_SAC_reference'));
addpath(fullfile(root,'Data','7fb','7fb-959277c48267','Land_model_matlab','release'));
s=load(fullfile(output_dir,'prepace_state.mat'),'par','isac_factor','beta');
c=load(fullfile(output_dir,'feedback_hetero_prepacing_pilot.mat'), ...
    'theta','states','last','converged');
if ~all(c.converged), error('Some virtual cells did not pass pre-pacing QC'); end
theta=c.theta; n=size(theta,1);
onsets=[150,600,225]; lambdas=[1.10,1.10,1.075];
grid=(0:1000)';
baseline=zeros(n,3); features=zeros(n,3,12);
waveforms=zeros(numel(grid),3,3,n,'single');
opt=odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',1);
for i=1:n
    initial=c.states(:,i); beta=[s.beta(1),theta(i,5)];
    fb=struct('g_ionic',theta(i,4),'ionic_scale',theta(i,1:3));
    control=simulate(initial,1,150,beta,fb,s,opt,grid);
    v0=control(:,1); ca0=1000*control(:,6);
    apd0=threshold_apd(v0,-67);
    ta0=tension(control,grid,1,150,beta,theta(i,6));
    baseline(i,:)=[apd0,max(ca0),max(ta0)];
    for j=1:3
        onset=onsets(j); lambda=lambdas(j);
        y=simulate(initial,lambda,onset,beta,fb,s,opt,grid);
        v=y(:,1); ca=1000*y(:,6);
        ta=tension(y,grid,lambda,onset,beta,theta(i,6));
        waveforms(:,:,j,i)=single([v,ca,ta]);
        win=grid>=onset & grid<=min(1000,onset+150);
        dv=v(win)-v0(win); dc=ca(win)-ca0(win);
        t20=min(1001,onset+21); t50=min(1001,onset+51);
        features(i,j,:)=[threshold_apd(v,-67)-apd0, ...
            min(dv),max(dv),min(dc),max(dc), ...
            max(ta),ta(t20),ta(t50), ...
            v(t20)-v0(t20),v(t50)-v0(t50), ...
            ca(t20)-ca0(t20),ca(t50)-ca0(t50)];
    end
    if mod(i,8)==0, fprintf('Hetero challenges %d/%d complete\n',i,n); end
end
save(fullfile(output_dir,'feedback_hetero_challenges.mat'), ...
    'theta','baseline','features','waveforms','onsets','lambdas','grid','-v7.3');
writematrix([theta,baseline,reshape(features(:,1,:),n,12), ...
    reshape(features(:,2,:),n,12),reshape(features(:,3,:),n,12)], ...
    fullfile(output_dir,'feedback_hetero_challenges.csv'));
end

function ta=tension(y,grid,lambda,onset,beta,scale)
ta=zeros(numel(grid),1); ca=1000*y(:,6);
for k=1:numel(grid)
    z=length_schedule(grid(k),lambda,onset,1,10);
    [~,~,ta(k)]=Land2016(grid(k),y(k,44:50),ca(k), ...
        'intact',z(1),z(2),beta);
end
ta=scale*ta;
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
