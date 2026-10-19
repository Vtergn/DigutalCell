function run_dynamic_model_candidates()
% Three candidate feedback equations at the same inferred/re-paced theta.
% No confirmation truth parameters or target outputs are loaded.
here=fileparts(mfilename('fullpath'));root=fileparts(here);
out=fullfile(here,'human_bidirectional_output');
addpath(fullfile(root,'Data','human_SAC_reference'));
addpath(fullfile(root,'Data','7fb','7fb-959277c48267', ...
    'Land_model_matlab','release'));
base=load(fullfile(out,'prepace_state.mat'),'par','isac_factor');
s=load(fullfile(out,'dynamic_inverse_calibration.mat'), ...
    'states','ok','cal','unique_theta');
prior=load(fullfile(out,'dynamic_inverse_targets.mat'), ...
    'prior_pred','prior_ok');
n=size(s.unique_theta,1);
grid=(0:1000)';
protocols={ [300,1.10,100], [300,1.15,10], ...
            [150,1.08,10;600,1.10,20] };
calibration={ [150,1.05,10], [600,1.10,10] };
candidate_cal=nan(1001,3,2,3,n,'single');
candidate_target=nan(1001,3,3,3,n,'single');
candidate_ok=false(3,n);
opt=odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',1);
for u=1:n
    if ~s.ok(u),continue;end
    candidate_cal(:,:,1,1,u)=s.cal(:,:,2,u);
    candidate_cal(:,:,2,1,u)=s.cal(:,:,3,u);
    candidate_cal(:,:,1,3,u)=s.cal(:,:,4,u);
    candidate_cal(:,:,2,3,u)=s.cal(:,:,5,u);
    candidate_target(:,:,:,1,u)=prior.prior_pred(:,:,:,u);
    candidate_ok(1,u)=prior.prior_ok(u);
    for form=[2,3]
        fb=parameters(s.unique_theta(u,:));
        if form==2,fb.sac_saturation=.05;end
        if form==3,fb.sac_gate=0;end
        try
            if form==2
                for j=1:2
                    state=simulate(s.states(:,u),calibration{j}, ...
                        fb,base,opt,grid);
                    candidate_cal(:,:,j,form,u)=single(trace_outputs( ...
                        state,grid,calibration{j},fb.land_params));
                end
            end
            for j=1:3
                state=simulate(s.states(:,u),protocols{j},fb,base,opt,grid);
                candidate_target(:,:,j,form,u)=single(trace_outputs( ...
                    state,grid,protocols{j},fb.land_params));
            end
            candidate_ok(form,u)=true;
        catch ME
            warning('CANDIDATE u=%d form=%d: %s',u,form,ME.message);
        end
    end
    if mod(u,20)==0 || u==n
        save(fullfile(out,'dynamic_model_candidates_checkpoint.mat'), ...
            'candidate_cal','candidate_target','candidate_ok','-v7.3');
        fprintf('CANDIDATES %d/%d valid=%d/%d/%d\n',u,n, ...
            sum(candidate_ok(1,:)),sum(candidate_ok(2,:)), ...
            sum(candidate_ok(3,:)));
    end
end
save(fullfile(out,'dynamic_model_candidates.mat'), ...
    'candidate_cal','candidate_target','candidate_ok','-v7.3');
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
