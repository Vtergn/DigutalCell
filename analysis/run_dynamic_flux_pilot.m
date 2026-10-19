function run_dynamic_flux_pilot()
% Native ODE forward propagation of C1/C2-only individualization. The
% residual acts only on the SAC-derived ionic state increment, preserving
% the source equation's coupled ionic bookkeeping.
here=fileparts(mfilename('fullpath'));
root=fileparts(here);
out=fullfile(here,'human_bidirectional_output');
addpath(fullfile(root,'Data','human_SAC_reference'));
addpath(fullfile(root,'Data','7fb','7fb-959277c48267', ...
    'Land_model_matlab','release'));
base=load(fullfile(out,'prepace_state.mat'),'par','isac_factor');
source=load(fullfile(out,'feedback_13param_cohort.mat'), ...
    'theta','states');
tab=readmatrix(fullfile(out,'dynamic_flux_pilot_input.csv'));
assert(size(tab,2)==7);
n=size(tab,1); grid=(0:1000)';
protocols={ [300,1.10,100], [300,1.15,10], ...
            [150,1.08,10;600,1.10,20] };
waveforms=nan(1001,3,3,3,n,'single');
ok=false(3,n);
opt=odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',1);
for i=1:n
    p=source.theta(tab(i,3),:);
    initial=source.states(:,tab(i,3));
    fb=struct('ionic_scale',p(1:6),'g_ionic',p(7), ...
        'land_params',struct('ca50',p(8),'TRPN_n',p(9), ...
        'nperm',p(10),'beta_0',p(11),'beta_1',p(12), ...
        'Tref',p(13)));
    for method=1:3
        q=fb;
        if method>=2, q.sac_gate=tab(i,4); end
        if method==3, q.sac_residual_slope=tab(i,5); end
        try
            for j=1:3
                state=simulate(initial,protocols{j},q,base,opt,grid);
                trace=trace_outputs(state,grid,protocols{j}, ...
                    q.land_params);
                validate_trace(trace);
                waveforms(:,:,j,method,i)=single(trace);
            end
            ok(method,i)=true;
        catch ME
            warning('FLUX cell %d method %d: %s',i,method,ME.message);
        end
    end
    if mod(i,10)==0 || i==n
        save(fullfile(out,'dynamic_flux_pilot_checkpoint.mat'), ...
            'waveforms','ok','tab','-v7.3');
        fprintf('FLUX %d/%d passed=%d/%d/%d\n',i,n, ...
            sum(ok(1,:)),sum(ok(2,:)),sum(ok(3,:)));
    end
end
save(fullfile(out,'dynamic_flux_pilot_predictions.mat'), ...
    'waveforms','ok','tab','-v7.3');
end

function state=simulate(initial,pulses,fb,base,opt,grid)
breaks=[0,1000];
for i=1:size(pulses,1)
    o=pulses(i,1);h=pulses(i,3);
    breaks=[breaks,o,o+1,o+1+h,o+2+h]; %#ok<AGROW>
end
breaks=unique(breaks);
current=initial;all_t=[];all_y=[];
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
validate_trace(out);
end

function z=length_schedule(t,pulses)
z=[1,0];
for i=1:size(pulses,1)
    o=pulses(i,1);amp=pulses(i,2)-1;h=pulses(i,3);
    if t<o || t>=o+2+h,continue;end
    if t<o+1
        z=[1+amp*(t-o),amp];
    elseif t<o+1+h
        z=[1+amp,0];
    else
        z=[1+amp*(o+2+h-t),-amp];
    end
    return;
end
end

function validate_trace(z)
if ~all(isfinite(z),'all') || min(z(:,1))<-150 || ...
        max(z(:,1))>100 || min(z(:,2))<0 || ...
        max(z(:,2))>10 || min(z(:,3))<-1 || max(z(:,3))>500
    error('nonphysiologic or nonfinite trace');
end
end
