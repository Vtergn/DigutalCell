function human_feedback_hetero_prepacing_pilot(ncells,output_dir)
% Feasibility/QC pilot for independent electrical baselines per virtual cell.
% Source ionic equations: model_SAC_hetero (three documented multipliers).
if nargin<1 || isempty(ncells), ncells=6; end
if nargin<2 || isempty(output_dir)
    output_dir=fullfile(fileparts(mfilename('fullpath')),'human_bidirectional_output');
end
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'Data','human_SAC_reference'));
addpath(fullfile(root,'Data','7fb','7fb-959277c48267','Land_model_matlab','release'));
s=load(fullfile(output_dir,'prepace_state.mat'),'state','par','isac_factor','beta');
rng(20261001);
u=zeros(ncells,6);
for j=1:6, u(:,j)=((randperm(ncells)'-1)+rand(ncells,1))/ncells; end
lo=[.85,.8,.8,.4,-4.1,.8];
hi=[1.15,1.2,1.2,1.6,-.9,1.2];
theta=lo+(hi-lo).*u;
% Columns: CaL, SERCA, NCX, effective stretch-ionic gain, beta1, Tref scale.
states=zeros(50,ncells); last=zeros(ncells,5); converged=false(ncells,1);
opt=odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',2);
for i=1:ncells
    state=s.state;
    fb=struct('g_ionic',theta(i,4),'ionic_scale',theta(i,1:3));
    beta=[s.beta(1),theta(i,5)];
    prior=[];
    for beat=1:160
        [t,y]=ode15s(@(t,x) human_bidirectional_rhs(t,x,1,true,true, ...
            s.par,s.isac_factor,beta,fb),[0,1000],state,opt);
        state=y(end,:)';
        if mod(beat,10)==0
            grid=(0:1000)'; sampled=interp1(t,y,grid,'pchip');
            apd=threshold_apd(sampled(:,1),-67);
            capeak=1000*max(sampled(:,6));
            if ~isempty(prior)
                apd_change=abs(apd-prior(1));
                ca_change=abs(capeak-prior(2))/capeak;
                if beat>=40 && apd_change<.25 && ca_change<.002
                    converged(i)=true; break;
                end
            end
            prior=[apd,capeak];
        end
    end
    ta=zeros(numel(grid),1);
    for k=1:numel(grid)
        [~,~,ta(k)]=Land2016(grid(k),sampled(k,44:50), ...
            1000*sampled(k,6),'intact',1,0,beta);
    end
    states(:,i)=state;
    last(i,:)=[beat,apd,capeak,theta(i,6)*max(ta),ca_change];
    fprintf('cell %d/%d: beats=%d APD=%.2f Ca=%.4f T=%.2f drift=%.4g converged=%d\n', ...
        i,ncells,beat,apd,capeak,last(i,4),ca_change,converged(i));
end
writematrix([(1:ncells)',theta,last,double(converged)], ...
    fullfile(output_dir,'feedback_hetero_prepacing_pilot.csv'));
save(fullfile(output_dir,'feedback_hetero_prepacing_pilot.mat'), ...
    'theta','states','last','converged','-v7.3');
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
