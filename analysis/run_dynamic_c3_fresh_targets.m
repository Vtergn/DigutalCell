function run_dynamic_c3_fresh_targets()
% Frozen development inverse -> new-seed native saturating-SAC long hold.
% The hidden fresh long-hold target is never read here.
here=fileparts(mfilename('fullpath'));root=fileparts(here);
out=fullfile(here,'human_bidirectional_output');
addpath(fullfile(root,'Data','human_SAC_reference'));
addpath(fullfile(root,'Data','7fb','7fb-959277c48267', ...
    'Land_model_matlab','release'));
base=load(fullfile(out,'prepace_state.mat'),'par','isac_factor');
source=load(fullfile(out,'feedback_13param_c3_fresh.mat'),'states');
tab=readmatrix(fullfile(out,'dynamic_c3_fresh_inverse.csv'));
assert(size(tab,2)==15);
n=size(tab,1);grid=(0:1000)';
pred=nan(1001,3,2,n,'single');ok=false(1,n);
beats_used=zeros(1,n);
pace=odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',2);
pulse=odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',1);
for row=1:n
    cell=tab(row,1);p=tab(row,3:15);
    fb=parameters(p);fb.sac_saturation=.05;
    try
        [state,beats]=prepace(source.states(:,cell),fb,base,pace,grid);
        beats_used(row)=beats;
        for cond=1:2
            if cond==1,protocol=[];
            else,protocol=[300,1.10,100];end
            x=simulate(state,protocol,fb,base,pulse,grid);
            pred(:,:,cond,row)=single(trace_outputs(x,grid,protocol, ...
                fb.land_params));
        end
        ok(row)=true;
    catch ME
        warning('C3 fresh inverse row=%d: %s',row,ME.message);
    end
    if mod(row,10)==0 || row==n
        save(fullfile(out,'dynamic_c3_fresh_targets_checkpoint.mat'), ...
            'pred','ok','beats_used','tab','-v7.3');
        fprintf('C3 fresh native %d/%d valid=%d\n',row,n,sum(ok));
    end
end
save(fullfile(out,'dynamic_c3_fresh_targets.mat'), ...
    'pred','ok','beats_used','tab','-v7.3');
end

function [state,beats]=prepace(state,fb,base,opt,grid)
prior=[];converged=false;
for beats=1:160
    [t,y]=ode15s(@(t,x)human_bidirectional_rhs(t,x,1, ...
        true,true,base.par,base.isac_factor,[2.3,-2.4],fb), ...
        [0,1000],state,opt);
    state=y(end,:)';
    if mod(beats,5)==0
        z=interp1(t,y,grid,'pchip');
        now=[threshold_apd(z(:,1),-67),1000*max(z(:,6))];
        if ~isempty(prior) && beats>=20 && ...
                abs(now(1)-prior(1))<.25 && ...
                abs(now(2)-prior(2))/max(now(2),1e-6)<.002
            converged=true;break;
        end
        prior=now;
    end
end
if ~converged,error('prepace not converged');end
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
