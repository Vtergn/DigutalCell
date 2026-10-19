function run_dynamic_inverse_residual_targets()
% Retest the *previously fixed* bounded SAC flux-residual rule after
% improved physiological state inference. No true test theta is loaded.
here=fileparts(mfilename('fullpath'));root=fileparts(here);
out=fullfile(here,'human_bidirectional_output');
addpath(fullfile(root,'Data','human_SAC_reference'));
addpath(fullfile(root,'Data','7fb','7fb-959277c48267', ...
    'Land_model_matlab','release'));
base=load(fullfile(out,'prepace_state.mat'),'par','isac_factor');
s=load(fullfile(out,'dynamic_inverse_calibration.mat'), ...
    'states','ok','map','unique_theta');
gate=readmatrix(fullfile(out,'dynamic_inverse_residual_input.csv'));
n=size(s.unique_theta,1);nr=size(gate,1);
assert(size(gate,2)==6 && nr==numel(s.map));
grid=(0:1000)';
protocols={ [300,1.10,100], [300,1.15,10], ...
            [150,1.08,10;600,1.10,20] };
residual_pred=nan(1001,3,3,nr,'single');
residual_ok=false(1,nr);
opt=odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',1);
for r=1:nr
    u=s.map(r);
    if ~s.ok(u) || ~isfinite(gate(r,5)),continue;end
    fb=parameters(s.unique_theta(u,:));fb.sac_gate=gate(r,5);
    fb.sac_residual_slope=gate(r,6);
    try
        for j=1:3
            state=simulate(s.states(:,u),protocols{j},fb,base,opt,grid);
            residual_pred(:,:,j,r)=single(trace_outputs( ...
                state,grid,protocols{j},fb.land_params));
        end
        residual_ok(r)=true;
    catch ME
        warning('INVERSE residual target %d: %s',r,ME.message);
    end
    if mod(r,20)==0 || r==nr
        save(fullfile(out,'dynamic_inverse_residual_checkpoint.mat'), ...
            'residual_pred','residual_ok','gate','-v7.3');
        fprintf('INVERSE residual target %d/%d valid=%d\n', ...
            r,nr,sum(residual_ok));
    end
end
save(fullfile(out,'dynamic_inverse_residual_targets.mat'), ...
    'residual_pred','residual_ok','gate','-v7.3');
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
