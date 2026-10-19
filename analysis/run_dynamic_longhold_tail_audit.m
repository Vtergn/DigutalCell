function run_dynamic_longhold_tail_audit(mode)
% Locked diagnostic only: true confirmation theta is used for oracle paths.
% The 4 cells were fixed by >5 mV correct-form long-hold error in the
% already scored baseline+C1/C2 inverse; nothing here trains a predictor.
here=fileparts(mfilename('fullpath'));root=fileparts(here);
if nargin<1,mode='main';end
out=fullfile(here,'human_bidirectional_output');
addpath(fullfile(root,'Data','human_SAC_reference'));
addpath(fullfile(root,'Data','7fb','7fb-959277c48267', ...
    'Land_model_matlab','release'));
base=load(fullfile(out,'prepace_state.mat'),'par','isac_factor');
inv=load(fullfile(out,'dynamic_inverse_calibration.mat'), ...
    'states','tab','map','unique_theta');
cohort=load(fullfile(out,'feedback_13param_confirm_cohort.mat'), ...
    'theta','states');
truth=load(fullfile(out,'dynamic_discrepancy_cohort.mat'),'waveforms');
cells=[8,18,43,57];
path_s=[0,.25,.5,.75,1];
group_names={'ionic_channels','Ca_handling','mechanical','feedback'};
groups={[1,4,5,6],[2,3],[8:11,13],[7,12]};
% Keep sensitivity analysis distinct from the sealed benchmark: oracle
% substitutions below cannot be used to score a deployable twin.
grid=(0:1000)';pulse=[300,1.10,100];window=301:602;
opt=odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',1);
paceopt=odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',2);
if strcmp(mode,'followup')
    followup=struct([]);q=0;
    for cell=cells
        row=find(inv.tab(:,1)==cell & inv.tab(:,2)==2 & ...
            inv.tab(:,3)==1);assert(numel(row)==1);u=inv.map(row);
        est=inv.unique_theta(u,:);actual=cohort.theta(cell,:);
        if cell==8,sets={1,4,5,6};
        elseif cell==18,sets={7,12,[1,4,5,6,7,12]};
        elseif cell==43,sets={1,4,5,6,[1,2,3,4,5,6]};
        else,sets={7,12};end
        target=squeeze(truth.waveforms(:,:,4,2,cell));
        baseline=squeeze(truth.waveforms(:,:,1,2,cell));
        for j=1:numel(sets)
            q=q+1;idx=sets{j};p=est;p(idx)=actual(idx);
            followup(q).cell=cell;followup(q).indices=idx;
            followup(q).ok=false;
            try
                fb=parameters(p);fb.sac_saturation=.05;
                [state,beats,metrics]=prepace( ...
                    cohort.states(:,cell),fb,base,paceopt,100);
                z=simulate(state,pulse,fb,base,opt,grid);
                z0=simulate(state,[],fb,base,opt,grid);
                y=trace_outputs(z,grid,pulse,fb.land_params);
                y0=trace_outputs(z0,grid,[],fb.land_params);
                aligned=baseline+y-y0;
                followup(q).mae=mean(abs(aligned(window,:)-target(window,:)),1);
                followup(q).apd=threshold_apd(y(:,1),-67);
                followup(q).ca_peak=max(y(:,2));
                followup(q).tension_peak=max(y(:,3));
                followup(q).beats=beats;
                followup(q).full_state_rel_rms=metrics;
                followup(q).ok=true;
            catch ME
                followup(q).error=ME.message;
            end
            fprintf('OAT cell=%d idx=%s Vmae=%.4g ok=%d\n', ...
                cell,mat2str(idx),safe_mae(followup(q)),followup(q).ok);
            save(fullfile(out,'dynamic_longhold_tail_followup.mat'), ...
                'followup','-v7.3');
        end
    end
    return;
end
results=struct([]);k=0;
for cell=cells
    row=find(inv.tab(:,1)==cell & inv.tab(:,2)==2 & ...
        inv.tab(:,3)==1);
    assert(numel(row)==1);u=inv.map(row);
    est=inv.unique_theta(u,:);actual=cohort.theta(cell,:);
    initial_true=cohort.states(:,cell);
    initial_est=inv.states(:,u);
    target=squeeze(truth.waveforms(:,:,4,2,cell));
    baseline=squeeze(truth.waveforms(:,:,1,2,cell));
    for kind=1:2
        if kind==1
            labels=arrayfun(@(s)sprintf('path_%.2f',s),path_s, ...
                'UniformOutput',false);
            vectors=est+path_s'.*(actual-est);
        else
            labels=group_names;
            vectors=repmat(est,numel(groups),1);
            for j=1:numel(groups)
                vectors(j,groups{j})=actual(groups{j});
            end
        end
        for j=1:numel(labels)
            k=k+1;p=vectors(j,:);fb=parameters(p);
            fb.sac_saturation=.05;
            results(k).cell=cell;
            results(k).kind=kind;
            results(k).label=labels{j};
            results(k).theta=p;
            results(k).ok=false;
            try
                % Reuse established endpoints; all intermediate and swap
                % parameter vectors receive independent native pre-pacing.
                if kind==1 && j==1
                    state=initial_est;beats=nan;metrics=nan(1,4);
                elseif kind==1 && j==numel(labels)
                    state=initial_true;beats=nan;metrics=nan(1,4);
                else
                    [state,beats,metrics]=prepace(initial_true,fb,base, ...
                        paceopt,100);
                end
                z=simulate(state,pulse,fb,base,opt,grid);
                z0=simulate(state,[],fb,base,opt,grid);
                y=trace_outputs(z,grid,pulse,fb.land_params);
                y0=trace_outputs(z0,grid,[],fb.land_params);
                aligned=baseline+y-y0;
                results(k).mae=mean(abs(aligned(window,:)-target(window,:)),1);
                results(k).apd=threshold_apd(y(:,1),-67);
                results(k).ca_peak=max(y(:,2));
                results(k).ca_min=min(y(:,2));
                results(k).tension_peak=max(y(:,3));
                results(k).beats=beats;
                results(k).full_state_rel_rms=metrics;
                results(k).ok=true;
            catch ME
                results(k).error=ME.message;
            end
            fprintf('TAIL cell=%d %s Vmae=%.4g ok=%d\n', ...
                cell,labels{j},safe_mae(results(k)),results(k).ok);
            save(fullfile(out,'dynamic_longhold_tail_audit.mat'), ...
                'results','cells','groups','group_names','path_s','-v7.3');
        end
    end
end
end

function value=safe_mae(r)
value=nan;if r.ok,value=r.mae(1);end
end

function [state,beats,metrics]=prepace(state,fb,base,opt,maxbeats)
metrics=nan(1,4);prior=state;
for beats=1:maxbeats
    [~,y]=ode15s(@(t,x) human_bidirectional_rhs(t,x,1, ...
        true,true,base.par,base.isac_factor,[2.3,-2.4],fb), ...
        [0,1000],state,opt);
    state=y(end,:)';
    if any(beats==[20,50,75,100])
        scale=max(max(abs(state),abs(prior)),1e-6);
        metrics(find([20,50,75,100]==beats,1))= ...
            sqrt(mean(((state-prior)./scale).^2));
    end
    prior=state;
end
end

function fb=parameters(p)
fb=struct('ionic_scale',p(1:6),'g_ionic',p(7), ...
    'land_params',struct('ca50',p(8),'TRPN_n',p(9), ...
    'nperm',p(10),'beta_0',p(11),'beta_1',p(12),'Tref',p(13)));
end

function state=simulate(initial,pulses,fb,base,opt,grid)
breaks=[0,1000];
for i=1:size(pulses,1)
    o=pulses(i,1);h=pulses(i,3);
    breaks=[breaks,o,o+1,o+1+h,o+2+h]; %#ok<AGROW>
end
breaks=unique(breaks);current=initial;all_t=[];all_y=[];
profile=@(t) length_schedule(t,pulses);
for j=1:numel(breaks)-1
    [t,y]=ode15s(@(t,y) human_bidirectional_rhs(t,y,profile, ...
        true,true,base.par,base.isac_factor,[2.3,-2.4],fb), ...
        breaks(j:j+1),current,opt);
    current=y(end,:)';
    if ~isempty(all_t),t=t(2:end);y=y(2:end,:);end
    all_t=[all_t;t];all_y=[all_y;y]; %#ok<AGROW>
end
state=interp1(all_t,all_y,grid,'pchip');
end

function out=trace_outputs(state,grid,pulses,land)
v=state(:,1);ca=1000*state(:,6);ta=zeros(numel(grid),1);
for k=1:numel(grid)
    z=length_schedule(grid(k),pulses);
    [~,~,ta(k)]=Land2016_variable(grid(k), ...
        state(k,44:50),ca(k),'intact',z(1),z(2),land);
end
out=[v,ca,ta];
if ~all(isfinite(out),'all') || min(v)<-150 || max(v)>100 || ...
        min(ca)<0 || max(ca)>10 || min(ta)<-1 || max(ta)>500
    error('nonphysiologic or nonfinite trace');
end
end

function z=length_schedule(t,pulses)
z=[1,0];
for i=1:size(pulses,1)
    o=pulses(i,1);amp=pulses(i,2)-1;h=pulses(i,3);
    if t<o || t>=o+2+h,continue;end
    if t<o+1,z=[1+amp*(t-o),amp];
    elseif t<o+1+h,z=[1+amp,0];
    else,z=[1+amp*(o+2+h-t),-amp];end
    return;
end
end

function duration=threshold_apd(vm,threshold)
up=find(vm(1:end-1)<=threshold & vm(2:end)>threshold,1,'first');
if isempty(up),duration=nan;return;end
down=find(vm(up+1:end-1)>=threshold & vm(up+2:end)<threshold,1,'first');
if isempty(down),duration=nan;return;end
down=down+up;
tup=up+(threshold-vm(up))/(vm(up+1)-vm(up));
tdown=down+(threshold-vm(down))/(vm(down+1)-vm(down));
duration=tdown-tup;
end
