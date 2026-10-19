function run_dynamic_inverse_calibration()
% Re-prepace inferred physiological parameters and generate only observed
% baseline/C1/C2 candidate equations. Sealed targets are not simulated here.
here=fileparts(mfilename('fullpath'));root=fileparts(here);
out=fullfile(here,'human_bidirectional_output');
addpath(fullfile(root,'Data','human_SAC_reference'));
addpath(fullfile(root,'Data','7fb','7fb-959277c48267', ...
    'Land_model_matlab','release'));
base=load(fullfile(out,'prepace_state.mat'),'par','isac_factor');
source=load(fullfile(out,'feedback_13param_cohort.mat'),'states');
tab=readmatrix(fullfile(out,'dynamic_physiology_inverse.csv'));
proto=readmatrix(fullfile(out,'dynamic_flux_pilot_input.csv'));
assert(size(tab,2)==16 && size(proto,2)==7);
[unique_theta,~,map]=unique(tab(:,4:16),'rows','stable');
n=size(unique_theta,1);assert(n<=size(tab,1));
states=nan(50,n);cal=nan(1001,3,5,n,'single');
ok=false(1,n);beats=zeros(1,n);
grid=(0:1000)';
pace=odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',2);
pulse=odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',1);
for u=1:n
    row=find(map==u,1,'first');p=unique_theta(u,:);
    fb=parameters(p);
    original=proto(proto(:,1)==tab(row,1) & ...
        proto(:,2)==tab(row,2),:);
    assert(size(original,1)==1);
    state=source.states(:,original(3));prior=[];converged=false;
    try
        for beat=1:160
            [t,y]=ode15s(@(t,x) human_bidirectional_rhs(t,x,1, ...
                true,true,base.par,base.isac_factor,[2.3,-2.4],fb), ...
                [0,1000],state,pace);
            state=y(end,:)';
            if mod(beat,5)==0
                z=interp1(t,y,grid,'pchip');
                now=[threshold_apd(z(:,1),-67),1000*max(z(:,6))];
                if ~isempty(prior) && beat>=20 && ...
                        abs(now(1)-prior(1))<.25 && ...
                        abs(now(2)-prior(2))/max(now(2),1e-6)<.002
                    converged=true;break;
                end
                prior=now;
            end
        end
        beats(u)=beat;
        if ~converged,error('prepace not converged');end
        states(:,u)=state;
        pulses={[],[150,1.05,10],[600,1.10,10]};
        for j=1:3
            z=simulate(state,pulses{j},fb,base,pulse,grid);
            cal(:,:,j,u)=single(trace_outputs(z,grid,pulses{j},fb.land_params));
        end
        zero=fb;zero.sac_gate=0;
        for j=2:3
            z=simulate(state,pulses{j},zero,base,pulse,grid);
            cal(:,:,j+2,u)=single(trace_outputs(z,grid,pulses{j},fb.land_params));
        end
        ok(u)=true;
    catch ME
        warning('INVERSE prepace %d/%d failed: %s',u,n,ME.message);
    end
    if mod(u,10)==0 || u==n
        save(fullfile(out,'dynamic_inverse_calibration_checkpoint.mat'), ...
            'states','cal','ok','beats','tab','map','unique_theta','-v7.3');
        fprintf('INVERSE calibration %d/%d valid=%d\n',u,n,sum(ok));
    end
end
save(fullfile(out,'dynamic_inverse_calibration.mat'), ...
    'states','cal','ok','beats','tab','map','unique_theta','-v7.3');
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
if isempty(up),duration=NaN;return;end
down=find(vm(up+1:end-1)>=threshold & vm(up+2:end)<threshold,1,'first');
if isempty(down),duration=NaN;return;end
down=down+up;
tup=up+(threshold-vm(up))/(vm(up+1)-vm(up));
tdown=down+(threshold-vm(down))/(vm(down+1)-vm(down));
duration=tdown-tup;
end
