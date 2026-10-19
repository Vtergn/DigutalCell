function human_feedback_hetero_native_check(output_dir)
% Six-parameter inverse -> re-pacing -> original coupled ODE target test.
if nargin<1 || isempty(output_dir)
    output_dir=fullfile(fileparts(mfilename('fullpath')),'human_bidirectional_output');
end
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'Data','human_SAC_reference'));
addpath(fullfile(root,'Data','7fb','7fb-959277c48267','Land_model_matlab','release'));
s=load(fullfile(output_dir,'prepace_state.mat'),'state','par','isac_factor','beta');
tab=readtable(fullfile(output_dir,'feedback_hetero_native_input.csv'));
opt=odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',2);
optpulse=odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',1);
grid=(0:1000)'; n=height(tab); prediction=zeros(n,6);
beats=zeros(n,1); converged=false(n,1);
for i=1:n
    th=[tab.est_CaL(i),tab.est_SERCA(i),tab.est_NCX(i), ...
        tab.est_g_ionic(i),tab.est_beta1(i),tab.est_Tref_scale(i)];
    fb=struct('g_ionic',th(4),'ionic_scale',th(1:3));
    beta=[s.beta(1),th(5)]; state=s.state; prior=[];
    for beat=1:160
        [t,y]=ode15s(@(t,x) human_bidirectional_rhs(t,x,1,true,true, ...
            s.par,s.isac_factor,beta,fb),[0,1000],state,opt);
        state=y(end,:)';
        if mod(beat,10)==0
            z=interp1(t,y,grid,'pchip');
            now=[threshold_apd(z(:,1),-67),1000*max(z(:,6))];
            if ~isempty(prior) && beat>=40 && ...
                    abs(now(1)-prior(1))<.25 && ...
                    abs(now(2)-prior(2))/now(2)<.002
                converged(i)=true; break;
            end
            prior=now;
        end
    end
    beats(i)=beat;
    control=simulate(state,1,225,beta,fb,s,optpulse,grid);
    y=simulate(state,1.075,225,beta,fb,s,optpulse,grid);
    v0=control(:,1); c0=1000*control(:,6);
    v=y(:,1); ca=1000*y(:,6); onset=225;
    win=grid>=onset & grid<=onset+150;
    dv=v(win)-v0(win); dc=ca(win)-c0(win);
    t20=onset+21;
    [~,~,ta20]=Land2016(grid(t20),y(t20,44:50),ca(t20), ...
        'intact',1,0,beta);
    prediction(i,:)=[threshold_apd(v,-67)-threshold_apd(v0,-67), ...
        max(dv),min(dc),th(6)*ta20, ...
        v(t20)-v0(t20),ca(t20)-c0(t20)];
    fprintf('native twin %d/%d: beats %d, converged %d\n', ...
        i,n,beat,converged(i));
end
writematrix([tab.cell_index,beats,double(converged),prediction], ...
    fullfile(output_dir,'feedback_hetero_native_predictions.csv'));
end

function states=simulate(initial,lambda,onset,beta,fb,s,opt,grid)
ramp=1; hold_ms=10; finish=onset+2*ramp+hold_ms;
breaks=[0,onset,onset+ramp,onset+ramp+hold_ms,finish,1000];
state=initial; all_t=[]; all_y=[];
for k=1:numel(breaks)-1
    a=breaks(k); b=breaks(k+1);
    fun=@(t) length_schedule(t,lambda,onset,ramp,hold_ms);
    [t,y]=ode15s(@(t,y) human_bidirectional_rhs(t,y,fun,true,true, ...
        s.par,s.isac_factor,beta,fb),[a,b],state,opt);
    state=y(end,:)';
    if ~isempty(all_t), t=t(2:end); y=y(2:end,:); end
    all_t=[all_t;t]; all_y=[all_y;y]; %#ok<AGROW>
end
states=interp1(all_t,all_y,grid,'pchip');
end

function z=length_schedule(t,lambda,onset,ramp,hold_ms)
delta=lambda-1; plateau_end=onset+ramp+hold_ms;
if t<onset || t>=plateau_end+ramp
    z=[1,0];
elseif t<onset+ramp
    z=[1+delta*(t-onset)/ramp,delta/ramp];
elseif t<plateau_end
    z=[lambda,0];
else
    z=[lambda-delta*(t-plateau_end)/ramp,-delta/ramp];
end
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
