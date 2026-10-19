function generate_dynamic_discrepancy_cohort(ncells,cohort_kind)
% Locked acute-pulse benchmark across three feedback equation forms.
% This does not claim independent-model external validation.
if nargin<1, ncells=24; end
if nargin<2, cohort_kind='confirmation'; end
assert(any(strcmp(cohort_kind,{'confirmation','development'})));
here=fileparts(mfilename('fullpath'));
root=fileparts(here);
out=fullfile(here,'human_bidirectional_output');
addpath(fullfile(root,'Data','human_SAC_reference'));
addpath(fullfile(root,'Data','7fb','7fb-959277c48267', ...
    'Land_model_matlab','release'));
base=load(fullfile(out,'prepace_state.mat'),'par','isac_factor');
if strcmp(cohort_kind,'development')
    input_name='feedback_13param_cohort.mat';
    final_name='dynamic_discrepancy_development.mat';
else
    input_name='feedback_13param_confirm_cohort.mat';
    final_name='dynamic_discrepancy_cohort.mat';
end
source=load(fullfile(out,input_name), ...
    'theta','states','success');
assert(ncells<=size(source.theta,1));
assert(all(source.success(1:ncells)));

% Rows are [onset_ms, lambda_peak, hold_ms]. Target 5 is compound.
protocols={ [150,1.05,10], [600,1.10,10], ...
            [300,1.10,100], [300,1.15,10], ...
            [150,1.08,10;600,1.10,20] };
names={'baseline','C1_early_105','C2_late_110', ...
       'T1_long_hold','T2_large_amplitude','T3_compound'};
forms={'prior','saturating_SAC','no_direct_SAC'};
grid=(0:1000)';
waveforms=nan(numel(grid),3,6,3,ncells,'single');
lambda_tracks=nan(numel(grid),6,'single');
lambda_tracks(:,1)=1;
for j=1:5
    for k=1:numel(grid)
        z=length_schedule(grid(k),protocols{j});
        lambda_tracks(k,j+1)=z(1);
    end
end
ok=false(3,ncells);
opt=odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',1);
for i=1:ncells
    p=source.theta(i,:);
    fb=struct('ionic_scale',p(1:6),'g_ionic',p(7), ...
        'land_params',struct('ca50',p(8),'TRPN_n',p(9), ...
        'nperm',p(10),'beta_0',p(11),'beta_1',p(12), ...
        'Tref',p(13)));
    try
        control=simulate(source.states(:,i),[],fb,base,opt,grid);
        base_trace=trace_outputs(control,grid,[],fb.land_params);
        for form=1:3
            specific=fb;
            if form==2, specific.sac_saturation=.05; end
            if form==3, specific.g_ionic=0; end
            waveforms(:,:,1,form,i)=single(base_trace);
            for j=1:5
                state=simulate(source.states(:,i),protocols{j}, ...
                    specific,base,opt,grid);
                trace=trace_outputs(state,grid,protocols{j}, ...
                    specific.land_params);
                validate_trace(trace);
                waveforms(:,:,j+1,form,i)=single(trace);
            end
            ok(form,i)=true;
            fprintf('DYNAMIC cell %d/%d form %s OK\n',i,ncells,forms{form});
        end
    catch ME
        warning('DYNAMIC cell %d failed: %s',i,ME.message);
    end
    save(fullfile(out,['dynamic_discrepancy_',cohort_kind,'_checkpoint.mat']), ...
        'waveforms','lambda_tracks','ok','forms','names','protocols', ...
        'grid','-v7.3');
end
save(fullfile(out,final_name), ...
    'waveforms','lambda_tracks','ok','forms','names','protocols', ...
    'grid','-v7.3');
fprintf('DYNAMIC complete valid per form %d/%d, %d/%d, %d/%d\n', ...
    sum(ok(1,:)),ncells,sum(ok(2,:)),ncells,sum(ok(3,:)),ncells);
end

function state=simulate(initial,pulses,fb,base,opt,grid)
if isempty(pulses)
    breaks=[0,1000];
else
    breaks=[0,1000];
    for i=1:size(pulses,1)
        o=pulses(i,1);h=pulses(i,3);
        breaks=[breaks,o,o+1,o+1+h,o+2+h]; %#ok<AGROW>
    end
    breaks=unique(breaks);
end
current=initial;all_t=[];all_y=[];
profile=@(t) length_schedule(t,pulses);
for j=1:numel(breaks)-1
    [t,y]=ode15s(@(t,y) human_bidirectional_rhs(t,y,profile, ...
        true,true,base.par,base.isac_factor,[2.3,-2.4],fb), ...
        breaks(j:j+1),current,opt);
    current=y(end,:)';
    if ~isempty(all_t), t=t(2:end);y=y(2:end,:);end
    all_t=[all_t;t];all_y=[all_y;y]; %#ok<AGROW>
end
state=interp1(all_t,all_y,grid,'pchip');
end

function out=trace_outputs(state,grid,pulses,land)
v=state(:,1);ca=1000*state(:,6);
ta=zeros(numel(grid),1);
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
if isempty(pulses),return;end
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
