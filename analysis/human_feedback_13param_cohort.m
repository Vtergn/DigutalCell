function human_feedback_13param_cohort(ncells,output_dir,seed,file_stem)
% Larger closed-loop synthetic cohort for staged protocol-design testing.
% All 13 parameters affect a single cell across every mechanical condition.
if nargin<1 || isempty(ncells), ncells=64; end
if nargin<2 || isempty(output_dir)
    output_dir=fullfile(fileparts(mfilename('fullpath')),'human_bidirectional_output');
end
if nargin<3 || isempty(seed), seed=20261001; end
if nargin<4 || isempty(file_stem), file_stem='feedback_13param_cohort'; end
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'Data','human_SAC_reference'));
addpath(fullfile(root,'Data','7fb','7fb-959277c48267','Land_model_matlab','release'));
base=load(fullfile(output_dir,'prepace_state.mat'),'state','par','isac_factor');
rng(seed);
names={'CaL','SERCA','NCX','IKr','IK1','INaL','g_ionic', ...
    'ca50','TRPN_n','nperm','beta0','beta1','Tref'};
lo=[.9,.85,.85,.85,.9,.8,.4,.75,1.8,4.5,1.8,-4.1,100];
hi=[1.1,1.15,1.15,1.15,1.1,1.2,1.6,.86,2.2,5.5,2.8,-.9,140];
u=zeros(ncells,numel(names));
for j=1:numel(names)
    u(:,j)=((randperm(ncells)'-1)+rand(ncells,1))/ncells;
end
theta=lo+(hi-lo).*u;
% First five protocols are design candidates. Last three are challenge tests.
onsets=[150,600,300,150,600,225,225,750];
lambdas=[1.10,1.10,1.10,1.05,1.05,1.075,1.125,1.08];
holds=[10,10,10,10,10,10,20,10];
grid=(0:1000)'; ncond=numel(onsets);
baseline=nan(ncells,3); features=nan(ncells,ncond,12);
waveforms=nan(numel(grid),3,ncond,ncells,'single');
states=nan(50,ncells); beats_used=nan(ncells,1);
converged=false(ncells,1); success=false(ncells,1);
optpace=odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',2);
optpulse=odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',1);
for i=1:ncells
    fb=feedback_struct(theta(i,:));
    state=base.state; prior=[];
    try
        for beat=1:160
            [t,y]=ode15s(@(t,x) human_bidirectional_rhs(t,x,1,true,true, ...
                base.par,base.isac_factor,[2.3,-2.4],fb), ...
                [0,1000],state,optpace);
            state=y(end,:)';
            if mod(beat,10)==0
                sampled=interp1(t,y,grid,'pchip');
                now=[threshold_apd(sampled(:,1),-67),1000*max(sampled(:,6))];
                if ~isempty(prior) && beat>=40 && ...
                        abs(now(1)-prior(1))<.25 && ...
                        abs(now(2)-prior(2))/now(2)<.002
                    converged(i)=true; break;
                end
                prior=now;
            end
        end
        beats_used(i)=beat;
        if ~converged(i), error('prepace criterion not met'); end
        states(:,i)=state;
        control=simulate(state,1,150,10,fb,base,optpulse,grid);
        v0=control(:,1); ca0=1000*control(:,6);
        ta0=tension(control,grid,1,150,10,fb.land_params);
        apd0=threshold_apd(v0,-67);
        baseline(i,:)=[apd0,max(ca0),max(ta0)];
        for j=1:ncond
            onset=onsets(j); lam=lambdas(j); hold_ms=holds(j);
            y=simulate(state,lam,onset,hold_ms,fb,base,optpulse,grid);
            v=y(:,1); ca=1000*y(:,6);
            ta=tension(y,grid,lam,onset,hold_ms,fb.land_params);
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
        success(i)=all(isfinite(features(i,:,:)),'all');
    catch ME
        warning('cell %d failed: %s',i,ME.message);
    end
    fprintf('13param cell %d/%d: beats=%g success=%d\n', ...
        i,ncells,beats_used(i),success(i));
    if mod(i,8)==0 || i==ncells
        save(fullfile(output_dir,[file_stem '_checkpoint.mat']), ...
            'theta','names','lo','hi','baseline','features','waveforms', ...
            'states','beats_used','converged','success','onsets', ...
            'lambdas','holds','grid','-v7.3');
    end
end
save(fullfile(output_dir,[file_stem '.mat']), ...
    'theta','names','lo','hi','baseline','features','waveforms', ...
    'states','beats_used','converged','success','onsets', ...
    'lambdas','holds','grid','-v7.3');
writematrix([theta,baseline,reshape(features,ncells,ncond*12), ...
    double(success)],fullfile(output_dir,[file_stem '.csv']));
end

function fb=feedback_struct(p)
fb=struct('ionic_scale',p(1:6),'g_ionic',p(7), ...
    'land_params',struct('ca50',p(8),'TRPN_n',p(9),'nperm',p(10), ...
    'beta_0',p(11),'beta_1',p(12),'Tref',p(13)));
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
