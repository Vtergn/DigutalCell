function generate_predictive_probe_library()
% Frozen response-diversity library screen on development cells only.
here=fileparts(mfilename('fullpath'));root=fileparts(here);
out=fullfile(here,'human_bidirectional_output');
addpath(fullfile(root,'Data','human_SAC_reference'));
addpath(fullfile(root,'Data','7fb','7fb-959277c48267', ...
    'Land_model_matlab','release'));
base=load(fullfile(out,'prepace_state.mat'),'par','isac_factor');
source=load(fullfile(out,'feedback_13param_cohort.mat'), ...
    'theta','states','success');
assert(numel(source.success)==64 && all(source.success));

probes={ ...
    [600,1.03,10,1], ...
    [600,1.05,10,1], ...
    [600,1.08,10,1], ...
    [600,1.10,20,1], ...
    [600,1.10,35,1], ...
    [600,1.08,35,1], ...
    [600,1.05,50,1], ...
    [600,1.08,10,10], ...
    [600,1.10,10,10], ...
    [150,1.03,10,1], ...
    [150,1.05,20,1], ...
    [150,1.08,10,10], ...
    [350,1.05,20,1], ...
    [450,1.05,20,10], ...
    [750,1.08,10,1], ...
    [150,1.03,5,1;600,1.08,10,1], ...
    [600,1.05,5,1;700,1.05,5,1], ...
    [600,1.10,0,5] ...
    };
assert(numel(probes)==18);
grid=(0:1000)';n=64;np=numel(probes);
baseline=nan(numel(grid),3,n,'single');
waveforms=nan(numel(grid),3,np,n,'single');
baseline_ok=false(n,1);ok=false(np,n);
opt=odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',1);
for cell=1:n
    p=source.theta(cell,:);
    fb=struct('ionic_scale',p(1:6),'g_ionic',p(7), ...
        'sac_saturation',.05, ...
        'land_params',struct('ca50',p(8),'TRPN_n',p(9), ...
        'nperm',p(10),'beta_0',p(11),'beta_1',p(12), ...
        'Tref',p(13)));
    try
        x=simulate(source.states(:,cell),[],fb,base,opt,grid);
        baseline(:,:,cell)=single(trace_outputs(x,grid,[],fb.land_params));
        baseline_ok(cell)=true;
    catch ME
        warning('PROBE LIB baseline cell=%d: %s',cell,ME.message);
    end
    if baseline_ok(cell)
        for j=1:np
            try
                x=simulate(source.states(:,cell),probes{j}, ...
                    fb,base,opt,grid);
                waveforms(:,:,j,cell)=single(trace_outputs( ...
                    x,grid,probes{j},fb.land_params));
                ok(j,cell)=true;
            catch ME
                warning('PROBE LIB cell=%d probe=%d: %s', ...
                    cell,j,ME.message);
            end
        end
    end
    if mod(cell,4)==0 || cell==n
        save(fullfile(out,'predictive_probe_library_checkpoint.mat'), ...
            'baseline','baseline_ok','waveforms','ok','probes','grid', ...
            '-v7.3');
        fprintf('PROBE LIB %d/%d baseline=%d all_probes=%d\n', ...
            cell,n,sum(baseline_ok),sum(all(ok,1)));
    end
end
save(fullfile(out,'predictive_probe_library.mat'), ...
    'baseline','baseline_ok','waveforms','ok','probes','grid','-v7.3');
end

function state=simulate(initial,pulses,fb,base,opt,grid)
breaks=[0,1000];
for i=1:size(pulses,1)
    o=pulses(i,1);h=pulses(i,3);r=pulses(i,4);
    assert(o>=0 && h>=0 && r>0 && o+2*r+h<=1000);
    breaks=[breaks,o,o+r,o+r+h,o+2*r+h]; %#ok<AGROW>
end
breaks=unique(breaks);current=initial;all_t=[];all_y=[];
profile=@(t) length_schedule4(t,pulses);
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
    z=length_schedule4(grid(k),pulses);
    [~,~,ta(k)]=Land2016_variable(grid(k), ...
        state(k,44:50),ca(k),'intact',z(1),z(2),land);
end
out=[v,ca,ta];
if ~all(isfinite(state),'all') || ~all(isfinite(out),'all') || ...
        min(v)<-150 || max(v)>100 || min(ca)<0 || ...
        max(ca)>10 || min(ta)<-1 || max(ta)>500
    error('nonphysiologic or nonfinite trace');
end
end

function z=length_schedule4(t,pulses)
z=[1,0];
for i=1:size(pulses,1)
    o=pulses(i,1);amp=pulses(i,2)-1;h=pulses(i,3);r=pulses(i,4);
    if t<o || t>=o+2*r+h,continue;end
    if t<o+r
        z=[1+amp*(t-o)/r,amp/r];
    elseif t<o+r+h
        z=[1+amp,0];
    else
        z=[1+amp*(o+2*r+h-t)/r,-amp/r];
    end
    return;
end
end
