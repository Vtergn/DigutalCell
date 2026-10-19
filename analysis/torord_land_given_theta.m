function out = torord_land_given_theta(theta, stretches, nBeats)
% True-equation forward map for one explicit virtual-cell parameter vector.
% theta order matches torord_land_cohort.m and theta_names in its manifest.
if nargin < 2 || isempty(stretches), stretches = 1.125; end
if nargin < 3, nBeats = 12; end
theta = double(theta(:)');
if numel(theta) ~= 12 || any(~isfinite(theta))
    error('Expected twelve finite parameters');
end
if any(stretches < 0.95) || any(stretches > 1.2)
    error('Stretch outside supported pilot range [0.95,1.2]');
end
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'Data','torord_official','matlab'));
addpath(fileparts(mfilename('fullpath')));
time_ms = (0:1000)';
p = struct('bcl',1000,'model',@model_Torord,'verbose',false, ...
    'cellType',0,'ICaL_Multiplier',theta(1), ...
    'Jup_Multiplier',theta(2),'INaCa_Multiplier',theta(3), ...
    'IKr_Multiplier',theta(4));
x0 = getStartingState('Torord_endo');
beats_used = 0;
while true
    block = nBeats;
    if beats_used>0, block=8; end
    [bt,bx] = modelRunner(x0,[],p,block,block-2);
    beats_used = beats_used+block;
    previous = 1000*interp1(bt{1},bx{1}(:,6),time_ms,'pchip');
    cai_uM = 1000*interp1(bt{2},bx{2}(:,6),time_ms,'pchip');
    vm_mV = interp1(bt{2},bx{2}(:,1),time_ms,'pchip');
    ca_drift = max(abs(cai_uM-previous))/max(cai_uM);
    if ca_drift<=.01 || beats_used>=60, break; end
    x0=bx{2}(end,:);
end
if any(~isfinite(vm_mV)) || any(~isfinite(cai_uM)) ...
        || any(cai_uM<=0) || max(vm_mV)<0 || ca_drift>.01
    error('Electrical simulation failed quality check');
end
m = struct('Tref',theta(5),'ca50',theta(6),'TRPN_n',theta(7), ...
    'nperm',theta(8),'beta_0',theta(9),'beta_1',theta(10), ...
    'phi',theta(11),'perm50',theta(12));
Ta = zeros(numel(stretches),numel(time_ms));
Tp = Ta;
land_drift = zeros(numel(stretches),1);
for j=1:numel(stretches)
    lam=stretches(j);
    ca_at = @(s) interp1(time_ms,cai_uM,min(1000,mod(s,1000)),'pchip');
    rhs = @(s,y) Land2016_variable(s,y,ca_at(s),'intact',lam,0,m);
    for cycles=[5,7]
        [lt,ly]=ode15s(rhs,[0 1000*cycles],Land2016_variable());
        y=interp1(lt,ly,1000*(cycles-1)+time_ms,'pchip');
        yp=interp1(lt,ly,1000*(cycles-2)+time_ms,'pchip');
        current=zeros(size(time_ms)); old=current; passive=current;
        for k=1:numel(time_ms)
            [~,~,current(k),passive(k)]=Land2016_variable(1000*(cycles-1)+time_ms(k),y(k,:),cai_uM(k),'intact',lam,0,m);
            [~,~,old(k),~]=Land2016_variable(1000*(cycles-2)+time_ms(k),yp(k,:),cai_uM(k),'intact',lam,0,m);
        end
        land_drift(j)=max(abs(current-old))/max(current);
        if land_drift(j)<=.01, break; end
    end
    if any(~isfinite(current)) || max(current)<=0 || land_drift(j)>.01
        error('Land simulation failed quality check at lambda %.3f',lam);
    end
    Ta(j,:)=current;
    Tp(j,:)=passive;
end
out=struct('time_ms',time_ms,'Vm_mV',vm_mV,'Cai_uM',cai_uM, ...
    'lambda',stretches,'Ta_kPa',Ta,'Tp_kPa',Tp, ...
    'electrical_beats_used',beats_used,'calcium_cycle_drift',ca_drift, ...
    'land_cycle_drift',land_drift);
end
