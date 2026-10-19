function human_feedback_13param_discrepancy(output_dir)
% Controlled structural mismatch: stretch gain changes with stretch size.
% NOT an independently published electromechanical model.
if nargin<1 || isempty(output_dir)
    output_dir=fullfile(fileparts(mfilename('fullpath')),'human_bidirectional_output');
end
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'Data','human_SAC_reference'));
addpath(fullfile(root,'Data','7fb','7fb-959277c48267','Land_model_matlab','release'));
base=load(fullfile(output_dir,'prepace_state.mat'),'par','isac_factor');
s=load(fullfile(output_dir,'feedback_13param_cohort.mat'), ...
    'theta','states','success','onsets','lambdas','holds','grid');
theta=s.theta; n=size(theta,1); ncond=numel(s.onsets); grid=s.grid;
baseline=nan(n,3); features=nan(n,ncond,12); success=false(n,1);
opt=odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',1);
for i=1:n
    if ~s.success(i), continue; end
    p=theta(i,:); fb=struct('ionic_scale',p(1:6), ...
        'g_ionic',p(7),'gain_exponent',.5, ...
        'land_params',struct('ca50',p(8),'TRPN_n',p(9), ...
        'nperm',p(10),'beta_0',p(11),'beta_1',p(12),'Tref',p(13)));
    try
        control=simulate(s.states(:,i),1,150,10,fb,base,opt,grid);
        v0=control(:,1); ca0=1000*control(:,6);
        ta0=tension(control,grid,1,150,10,fb.land_params);
        baseline(i,:)=[threshold_apd(v0,-67),max(ca0),max(ta0)];
        for j=1:ncond
            onset=s.onsets(j); lam=s.lambdas(j); hold_ms=s.holds(j);
            y=simulate(s.states(:,i),lam,onset,hold_ms,fb,base,opt,grid);
            v=y(:,1); ca=1000*y(:,6);
            ta=tension(y,grid,lam,onset,hold_ms,fb.land_params);
            win=grid>=onset & grid<=min(1000,onset+150);
            dv=v(win)-v0(win); dc=ca(win)-ca0(win);
            t20=min(1001,onset+21); t50=min(1001,onset+51);
            features(i,j,:)=[threshold_apd(v,-67)-baseline(i,1), ...
                min(dv),max(dv),min(dc),max(dc), ...
                max(ta),ta(t20),ta(t50), ...
                v(t20)-v0(t20),v(t50)-v0(t50), ...
                ca(t20)-ca0(t20),ca(t50)-ca0(t50)];
        end
        success(i)=all(isfinite(features(i,:,:)),'all');
    catch ME
        warning('discrepancy cell %d failed: %s',i,ME.message);
    end
    if mod(i,8)==0, fprintf('discrepancy %d/%d cells\n',i,n); end
end
save(fullfile(output_dir,'feedback_13param_discrepancy.mat'), ...
    'theta','baseline','features','success','-v7.3');
writematrix([theta,baseline,reshape(features,n,ncond*12),double(success)], ...
    fullfile(output_dir,'feedback_13param_discrepancy.csv'));
end

function ta=tension(y,grid,lambda,onset,hold_ms,land_params)
ta=zeros(numel(grid),1); ca=1000*y(:,6);
for k=1:numel(grid)
    z=length_schedule(grid(k),lambda,onset,1,hold_ms);
    [~,~,ta(k)]=Land2016_variable(grid(k),y(k,44:50),ca(k), ...
        'intact',z(1),z(2),land_params);
end
end

function states=simulate(initial,lambda,onset,hold_ms,fb,base,opt,grid)
ramp=1; finish=onset+2*ramp+hold_ms;
breaks=[0,onset,onset+ramp,onset+ramp+hold_ms,finish,1000];
state=initial; all_t=[]; all_y=[];
for k=1:numel(breaks)-1
    a=breaks(k); b=breaks(k+1);
    fun=@(t) length_schedule(t,lambda,onset,ramp,hold_ms);
    [t,y]=ode15s(@(t,y) human_bidirectional_rhs(t,y,fun,true,true, ...
        base.par,base.isac_factor,[2.3,-2.4],fb),[a,b],state,opt);
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
