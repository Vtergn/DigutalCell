function human_feedback_13param_two_forms(output_dir)
% Two controlled model-form changes on 12 sealed cells, not independent models.
% Scenario 2: a saturating length-to-ionic input law, anchored at lambda 1.10.
% Scenario 3: 75% of the original dynamic TnC calcium-buffering capacity.
if nargin<1 || isempty(output_dir)
    output_dir=fullfile(fileparts(mfilename('fullpath')),'human_bidirectional_output');
end
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'Data','human_SAC_reference'));
addpath(fullfile(root,'Data','7fb','7fb-959277c48267','Land_model_matlab','release'));
base=load(fullfile(output_dir,'prepace_state.mat'),'state','par','isac_factor');
s=load(fullfile(output_dir,'feedback_13param_cohort.mat'),'theta','states','success');
% MATLAB's RNG permutation differs from NumPy. Read the exact Python test
% indices from the existing native input file instead.
native=readmatrix(fullfile(output_dir,'feedback_13param_native_input.csv'));
cell_ids=native(1:12,1);
assert(all(native(1:12,2)==0) && all(native(13:24,2)==1));
grid=(0:1000)'; opt=odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',1);
pace=odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',2);
onsets=[300,225,225,750]; lambdas=[1.10,1.075,1.125,1.08];
holds=[10,10,20,10];
rows=nan(24,1+13+3+4*12+1);
calibration_waveforms=nan(24,3,1001,'single');
for form=1:2
    for k=1:12
        cell_id=cell_ids(k); p=s.theta(cell_id,:);
        ix=(form-1)*12+k;
        fb=struct('ionic_scale',p(1:6),'g_ionic',p(7), ...
            'land_params',struct('ca50',p(8),'TRPN_n',p(9), ...
            'nperm',p(10),'beta_0',p(11),'beta_1',p(12),'Tref',p(13)));
        if form==1
            fb.sac_saturation=.05;
            state=s.states(:,cell_id);
        else
            fb.trpn_capacity_scale=.75;
            state=base.state; prior=[]; ok=false;
            for beat=1:160
                [t,y]=ode15s(@(t,x) human_bidirectional_rhs(t,x,1,true,true, ...
                    base.par,base.isac_factor,[2.3,-2.4],fb), ...
                    [0,1000],state,pace);
                state=y(end,:)';
                if mod(beat,10)==0
                    z=interp1(t,y,grid,'pchip');
                    now=[threshold_apd(z(:,1),-67),1000*max(z(:,6))];
                    if ~isempty(prior) && beat>=40 && ...
                            abs(now(1)-prior(1))<.25 && ...
                            abs(now(2)-prior(2))/now(2)<.002
                        ok=true; break;
                    end
                    prior=now;
                end
            end
            if ~ok, error('TnC alternate pre-pacing failed at cell %d',cell_id); end
        end
        control=simulate(state,1,150,10,fb,base,opt,grid);
        v0=control(:,1); ca0=1000*control(:,6);
        ta0=tension(control,grid,1,150,10,fb.land_params);
        apd0=threshold_apd(v0,-67);
        feats=nan(4,12);
        for j=1:4
            onset=onsets(j); lam=lambdas(j); hold_ms=holds(j);
            y=simulate(state,lam,onset,hold_ms,fb,base,opt,grid);
            v=y(:,1); ca=1000*y(:,6);
            ta=tension(y,grid,lam,onset,hold_ms,fb.land_params);
            if j==1
                calibration_waveforms(ix,:,:) = single([v,ca,ta].');
            end
            win=grid>=onset & grid<=min(1000,onset+150);
            dv=v(win)-v0(win); dc=ca(win)-ca0(win);
            t20=min(1001,onset+21); t50=min(1001,onset+51);
            feats(j,:)=[threshold_apd(v,-67)-apd0, ...
                min(dv),max(dv),min(dc),max(dc), ...
                max(ta),ta(t20),ta(t50), ...
                v(t20)-v0(t20),v(t50)-v0(t50), ...
                ca(t20)-ca0(t20),ca(t50)-ca0(t50)];
        end
        rows(ix,:)=[cell_id, p, apd0,max(ca0),max(ta0), ...
            reshape(feats.',1,[]), double(all(isfinite(feats),'all'))];
        fprintf('two-form %d/2 cell %d/12 ok\n',form,k);
    end
end
writematrix(rows,fullfile(output_dir,'feedback_13param_two_forms.csv'));
save(fullfile(output_dir,'feedback_13param_two_forms_waveforms.mat'), ...
    'calibration_waveforms','cell_ids','-v7.3');
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
