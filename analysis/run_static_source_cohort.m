function run_static_source_cohort()
% Source ToR-ORd/SAC-Land static-length cohort for a matched cross-model test.
% Cell parameters and prestimulus states are frozen from the earlier 64 cells.
here=fileparts(mfilename('fullpath'));
root=fileparts(here);
out=fullfile(here,'human_bidirectional_output');
addpath(fullfile(root,'Data','human_SAC_reference'));
addpath(fullfile(root,'Data','7fb','7fb-959277c48267', ...
    'Land_model_matlab','release'));
base=load(fullfile(out,'prepace_state.mat'),'par','isac_factor');
cohort=load(fullfile(out,'feedback_13param_cohort.mat'),'theta','states');
theta=cohort.theta; states=cohort.states;
lengths=[1,1.05,1.075,1.10];
grid=(0:999)';
n=size(theta,1);
features=nan(n,4,14); ok=false(n,4);
opt=odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',2);
for i=1:n
    p=theta(i,:);
    fb=struct('ionic_scale',p(1:6),'g_ionic',p(7), ...
        'land_params',struct('ca50',p(8),'TRPN_n',p(9), ...
        'nperm',p(10),'beta_0',p(11),'beta_1',p(12), ...
        'Tref',p(13)));
    for j=1:4
        lam=lengths(j);
        try
            [t,y]=ode15s(@(t,x) human_bidirectional_rhs(t,x,lam, ...
                true,true,base.par,base.isac_factor,[2.3,-2.4],fb), ...
                [0,1000],states(:,i),opt);
            sample=interp1(t,y,grid,'pchip');
            v=sample(:,1); c=1000*sample(:,6);
            s=zeros(numel(grid),1);
            for k=1:numel(grid)
                [~,~,s(k)]=Land2016_variable(grid(k), ...
                    sample(k,44:50),c(k),'intact',lam,0,fb.land_params);
            end
            apd=ap_duration(v,-67);
            features(i,j,:)=[apd,max(c),max(s),min(v),max(v), ...
                c(201),c(401),c(601),s(201),s(401),s(601), ...
                trapz(grid,c)/1000,trapz(grid,s)/1000,s(end)];
            ok(i,j)=all(isfinite(features(i,j,:)),'all');
            fprintf('SOURCE %d/%d LOAD %d OK=%d\n',i,n,j,ok(i,j));
        catch ME
            fprintf('SOURCE %d/%d LOAD %d FAILED %s\n',i,n,j,ME.message);
        end
    end
    save(fullfile(out,'feedback_static_cohort_checkpoint.mat'), ...
        'theta','lengths','features','ok','-v7');
end
writematrix(reshape(features,n,[]), ...
    fullfile(out,'feedback_static_cohort_features.csv'));
fprintf('SOURCE STATIC COMPLETE %d/%d full cells\n',sum(all(ok,2)),n);
end

function duration=ap_duration(vm,threshold)
up=find(vm(1:end-1)<=threshold & vm(2:end)>threshold,1,'first');
if isempty(up), duration=NaN; return; end
down=find(vm(up+1:end-1)>=threshold & vm(up+2:end)<threshold,1,'first');
if isempty(down), duration=NaN; return; end
down=down+up;
tup=up+(threshold-vm(up))/(vm(up+1)-vm(up));
tdown=down+(threshold-vm(down))/(vm(down+1)-vm(down));
duration=tdown-tup;
end
