function human_feedback_13param_native_check(output_dir,input_name,output_name)
% Inverse 13 parameters, independently re-pace, then forward three targets.
if nargin<1 || isempty(output_dir)
    output_dir=fullfile(fileparts(mfilename('fullpath')),'human_bidirectional_output');
end
if nargin<2 || isempty(input_name), input_name='feedback_13param_native_input.csv'; end
if nargin<3 || isempty(output_name), output_name='feedback_13param_native_predictions.csv'; end
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'Data','human_SAC_reference'));
addpath(fullfile(root,'Data','7fb','7fb-959277c48267','Land_model_matlab','release'));
base=load(fullfile(output_dir,'prepace_state.mat'),'state','par','isac_factor');
tab=readmatrix(fullfile(output_dir,input_name));
grid=(0:1000)'; n=size(tab,1); output=nan(n,22);
pace=odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',2);
pulse=odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',1);
onsets=[225,225,750]; lambdas=[1.075,1.125,1.08]; holds=[10,20,10];
for i=1:n
    output(i,1:2)=tab(i,1:2);
    p=tab(i,16:28);
    fb=struct('ionic_scale',p(1:6),'g_ionic',p(7), ...
        'land_params',struct('ca50',p(8),'TRPN_n',p(9), ...
        'nperm',p(10),'beta_0',p(11),'beta_1',p(12),'Tref',p(13)));
    state=base.state; prior=[]; ok=false;
    try
        for beat=1:160
            [t,y]=ode15s(@(t,x) human_bidirectional_rhs(t,x,1,true,true, ...
                base.par,base.isac_factor,[2.3,-2.4],fb), ...
                [0,1000],state,pace);
            state=y(end,:)';
            if mod(beat,10)==0
                z=interp1(t,y,grid,'pchip');
                now=[threshold_apd(z(:,1),-67),1000*max(z(:,6))];
                if ~isempty(prior) && beat>=40 && ...
                    abs(now(1)-prior(1))<.25 && ...
                    abs(now(2)-prior(2))/now(2)<.002
                    ok=true; break;
                end
                prior=now;
            end
        end
        output(i,3:4)=[beat,double(ok)];
        if ~ok, error('failed pre-pacing criterion'); end
        control=simulate(state,1,150,10,fb,base,pulse,grid);
        v0=control(:,1); ca0=1000*control(:,6);
        base_apd=threshold_apd(v0,-67);
        for j=1:3
            onset=onsets(j); lam=lambdas(j); hold_ms=holds(j);
            y=simulate(state,lam,onset,hold_ms,fb,base,pulse,grid);
            v=y(:,1); ca=1000*y(:,6);
            win=grid>=onset & grid<=min(1000,onset+150);
            dv=v(win)-v0(win); dc=ca(win)-ca0(win);
            t20=min(1001,onset+21);
            z=length_schedule(grid(t20),lam,onset,1,hold_ms);
            [~,~,ta20]=Land2016_variable(grid(t20),y(t20,44:50), ...
                ca(t20),'intact',z(1),z(2),fb.land_params);
            k=4+(j-1)*6;
            output(i,k+(1:6))=[threshold_apd(v,-67)-base_apd, ...
                max(dv),min(dc),ta20, ...
                v(t20)-v0(t20),ca(t20)-ca0(t20)];
        end
    catch ME
        warning('native 13-param cell %d failed: %s',i,ME.message);
    end
    fprintf('native 13param %d/%d: beats=%g ok=%d\n', ...
        i,n,output(i,3),ok);
end
writematrix(output,fullfile(output_dir,output_name));
end

function states=simulate(initial,lambda,onset,hold_ms,fb,base,opt,grid)
ramp=1; finish=onset+2*ramp+hold_ms;
breaks=[0,onset,onset+ramp,onset+ramp+hold_ms,finish,1000];
state=initial; all_t=[]; all_y=[];
for k=1:numel(breaks)-1
    a=breaks(k); b=breaks(k+1);
    fun=@(t) length_schedule(t,lambda,onset,ramp,hold_ms);
    [t,y]=ode15s(@(t,y) human_bidirectional_rhs(t,y,fun,true,true, ...
        base.par,base.isac_factor,[2.3,-2.4],fb),[a,b],state,opt);
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
