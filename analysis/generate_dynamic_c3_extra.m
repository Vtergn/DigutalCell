function generate_dynamic_c3_extra(kind,selected_index)
% Saturating-SAC C3 truth only. Development gets all six candidates; the
% fresh-seed confirmation gets only the already selected candidate.
if nargin<1,kind='development';end
if nargin<2,selected_index=nan;end
assert(any(strcmp(kind,{'development','fresh','event_margin', ...
    'event_margin_replicate'})));
here=fileparts(mfilename('fullpath'));root=fileparts(here);
out=fullfile(here,'human_bidirectional_output');
addpath(fullfile(root,'Data','human_SAC_reference'));
addpath(fullfile(root,'Data','7fb','7fb-959277c48267', ...
    'Land_model_matlab','release'));
base=load(fullfile(out,'prepace_state.mat'),'par','isac_factor');
if strcmp(kind,'development')
    source=load(fullfile(out,'feedback_13param_cohort.mat'), ...
        'theta','states','success');
    candidates=[150,1.10,20;150,1.10,35;150,1.10,50; ...
                600,1.10,20;600,1.10,35;600,1.10,50];
    protocols=num2cell(candidates,2);
elseif strcmp(kind,'fresh')
    assert(isscalar(selected_index) && selected_index>=1 && ...
        selected_index<=6);
    source=load(fullfile(out,'feedback_13param_c3_fresh.mat'), ...
        'theta','states','success');
    candidates=[150,1.10,20;150,1.10,35;150,1.10,50; ...
                600,1.10,20;600,1.10,35;600,1.10,50];
    protocols={[],[150,1.05,10],[600,1.10,10], ...
        candidates(selected_index,:),[300,1.10,100]};
elseif strcmp(kind,'event_margin')
    source=load(fullfile(out,'feedback_13param_event_margin_confirm.mat'), ...
        'theta','states','success');
    candidates=[];
    protocols={[],[150,1.05,10],[600,1.10,10],[300,1.10,100]};
else
    source=load(fullfile(out,'feedback_13param_event_margin_replicate.mat'), ...
        'theta','states','success');
    candidates=[];
    protocols={[],[150,1.05,10],[600,1.10,10],[300,1.10,100]};
end
n=size(source.theta,1);nc=numel(protocols);grid=(0:1000)';
waveforms=nan(1001,3,nc,n,'single');ok=false(nc,n);
opt=odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',1);
for cell=1:n
    if ~source.success(cell),continue;end
    p=source.theta(cell,:);
    fb=struct('ionic_scale',p(1:6),'g_ionic',p(7), ...
        'sac_saturation',.05, ...
        'land_params',struct('ca50',p(8),'TRPN_n',p(9), ...
        'nperm',p(10),'beta_0',p(11),'beta_1',p(12),'Tref',p(13)));
    for j=1:nc
        try
            x=simulate(source.states(:,cell),protocols{j}, ...
                fb,base,opt,grid);
            waveforms(:,:,j,cell)=single(trace_outputs( ...
                x,grid,protocols{j},fb.land_params));
            ok(j,cell)=true;
        catch ME
            warning('C3 %s cell=%d cond=%d: %s', ...
                kind,cell,j,ME.message);
        end
    end
    if mod(cell,10)==0 || cell==n
        save(fullfile(out,['dynamic_c3_',kind,'_checkpoint.mat']), ...
            'waveforms','ok','protocols','candidates','selected_index','-v7.3');
        fprintf('C3 %s %d/%d complete=%d\n',kind,cell,n,sum(all(ok,1)));
    end
end
save(fullfile(out,['dynamic_c3_',kind,'.mat']), ...
    'waveforms','ok','protocols','candidates','selected_index','-v7.3');
if strcmp(kind,'event_margin') || strcmp(kind,'event_margin_replicate')
    % A separate, target-free artifact for deployable risk scoring.
    cal=waveforms(:,:,1:3,:);
    cal_ok=ok(1:3,:);
    calfile='dynamic_event_margin_calibration.mat';
    if strcmp(kind,'event_margin_replicate')
        calfile='dynamic_event_margin_replication_calibration.mat';
    end
    save(fullfile(out,calfile), ...
        'cal','cal_ok','-v7.3');
end
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
