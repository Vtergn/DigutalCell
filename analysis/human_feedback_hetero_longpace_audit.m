function human_feedback_hetero_longpace_audit(output_dir)
% Verify short convergence criterion against 100 additional paced beats.
if nargin<1 || isempty(output_dir)
    output_dir=fullfile(fileparts(mfilename('fullpath')),'human_bidirectional_output');
end
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'Data','human_SAC_reference'));
addpath(fullfile(root,'Data','7fb','7fb-959277c48267','Land_model_matlab','release'));
s=load(fullfile(output_dir,'prepace_state.mat'),'par','isac_factor','beta');
c=load(fullfile(output_dir,'feedback_hetero_prepacing_pilot.mat'), ...
    'theta','states','last');
[~,loCa]=min(c.last(:,3)); [~,hiCa]=max(c.last(:,3));
[~,loT]=min(c.last(:,4)); [~,hiT]=max(c.last(:,4));
ix=unique([1;loCa;hiCa;loT;hiT]);
opt=odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',2);
rows=zeros(numel(ix),8);
states=zeros(50,numel(ix));
for j=1:numel(ix)
    i=ix(j); state=c.states(:,i);
    fb=struct('g_ionic',c.theta(i,4),'ionic_scale',c.theta(i,1:3));
    beta=[s.beta(1),c.theta(i,5)];
    for beat=1:100
        [t,y]=ode15s(@(t,x) human_bidirectional_rhs(t,x,1,true,true, ...
            s.par,s.isac_factor,beta,fb),[0,1000],state,opt);
        state=y(end,:)';
    end
    grid=(0:1000)'; y=interp1(t,y,grid,'pchip');
    apd=threshold_apd(y(:,1),-67); capeak=1000*max(y(:,6));
    ta=zeros(numel(grid),1);
    for k=1:numel(grid)
        [~,~,ta(k)]=Land2016(grid(k),y(k,44:50), ...
            1000*y(k,6),'intact',1,0,beta);
    end
    tpeak=c.theta(i,6)*max(ta);
    states(:,j)=state;
    rows(j,:)=[i,c.last(i,2:4),apd,capeak,tpeak, ...
        (capeak-c.last(i,3))/c.last(i,3)];
    fprintf('long pace cell %d: APD %g -> %g, Ca %g -> %g, T %g -> %g\n', ...
        i,c.last(i,2),apd,c.last(i,3),capeak,c.last(i,4),tpeak);
end
writematrix(rows,fullfile(output_dir,'feedback_hetero_longpace_audit.csv'));
save(fullfile(output_dir,'feedback_hetero_longpace_audit.mat'),'ix','states','rows');
end

function duration=threshold_apd(vm,threshold)
up=find(vm(1:end-1)<=threshold & vm(2:end)>threshold,1,'first');
if isempty(up), duration=NaN; return; end
down=find(vm(up+1:end-1)>=threshold & vm(up+2:end)<threshold,1,'first');
if isempty(down), duration=NaN; return; end
down=down+up;
tup=up+(threshold-vm(up))/(vm(up+1)-vm(up));
tdown=down+(threshold-vm(down))/(vm(down+1)-vm(down));
duration=tdown-tup;
end
