function run_dynamic_tail_state_audit()
% Extra 200 beats from the *already pre-paced* estimated or true state.
% Test whether delayed-state drift can explain long-hold tail errors.
here=fileparts(mfilename('fullpath'));root=fileparts(here);
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
cells=[8,18,43,57];checkpoints=[0,20,50,100,200];
rows=struct([]);k=0;grid=(0:1000)';window=301:602;
paceopt=odeset('RelTol',1e-7,'AbsTol',1e-9,'MaxStep',2);
pulseopt=odeset('RelTol',1e-7,'AbsTol',1e-9,'MaxStep',1);
for cell=cells
    row=find(inv.tab(:,1)==cell & inv.tab(:,2)==2 & ...
        inv.tab(:,3)==1);assert(numel(row)==1);u=inv.map(row);
    target=squeeze(truth.waveforms(:,:,4,2,cell));
    baseline=squeeze(truth.waveforms(:,:,1,2,cell));
    for kind=1:3
        if kind==3 && cell~=18,continue;end
        if kind==1
            p=inv.unique_theta(u,:);state=inv.states(:,u);
        elseif kind==2
            p=cohort.theta(cell,:);state=cohort.states(:,cell);
        else
            p=inv.unique_theta(u,:)+.75*(cohort.theta(cell,:)- ...
                inv.unique_theta(u,:));
            state=cohort.states(:,cell);
        end
        fb=parameters(p);fb.sac_saturation=.05;
        prev=state;
        for beat=0:200
            if beat>0
                [~,y]=ode15s(@(t,x)human_bidirectional_rhs(t,x,1, ...
                    true,true,base.par,base.isac_factor,[2.3,-2.4],fb), ...
                    [0,1000],state,paceopt);
                prev=state;state=y(end,:)';
            end
            if ~ismember(beat,checkpoints),continue;end
            k=k+1;rows(k).cell=cell;rows(k).kind=kind;
            rows(k).extra_beats=beat;rows(k).ok=false;
            if beat>0
                scale=max(max(abs(state),abs(prev)),1e-6);
                rows(k).full_state_rel_rms= ...
                    sqrt(mean(((state-prev)./scale).^2));
                rows(k).full_state_rel_max= ...
                    max(abs((state-prev)./scale));
            else
                rows(k).full_state_rel_rms=nan;
                rows(k).full_state_rel_max=nan;
            end
            try
                a=simulate(state,[300,1.10,100],fb,base,pulseopt,grid);
                b=simulate(state,[],fb,base,pulseopt,grid);
                ya=trace_outputs(a,grid,[300,1.10,100],fb.land_params);
                yb=trace_outputs(b,grid,[],fb.land_params);
                pred=baseline+ya-yb;
                rows(k).mae=mean(abs(pred(window,:)-target(window,:)),1);
                rows(k).apd=threshold_apd(ya(:,1),-67);
                rows(k).ok=true;
            catch ME
                rows(k).error=ME.message;
            end
            fprintf('STATE cell=%d kind=%d beats=%d Vmae=%.4g rms=%.3g\n', ...
                cell,kind,beat,safe_mae(rows(k)),rows(k).full_state_rel_rms);
            save(fullfile(out,'dynamic_tail_state_audit.mat'), ...
                'rows','cells','checkpoints','-v7.3');
        end
    end
end
end

function value=safe_mae(r)
value=nan;if r.ok,value=r.mae(1);end
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
