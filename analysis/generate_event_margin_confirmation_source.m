function generate_event_margin_confirmation_source()
% New 64-cell Latin-hypercube seed for the frozen event-margin test.
here=fileparts(mfilename('fullpath'));root=fileparts(here);
out=fullfile(here,'human_bidirectional_output');
addpath(fullfile(root,'Data','human_SAC_reference'));
base=load(fullfile(out,'prepace_state.mat'),'state','par','isac_factor');
rng(20261021);
n=64;
lo=[.9,.85,.85,.85,.9,.8,.4,.75,1.8,4.5,1.8,-4.1,100];
hi=[1.1,1.15,1.15,1.15,1.1,1.2,1.6,.86,2.2,5.5,2.8,-.9,140];
unit=zeros(n,13);
for j=1:13
    unit(:,j)=((randperm(n)'-1)+rand(n,1))/n;
end
theta=lo+(hi-lo).*unit;
states=nan(50,n);success=false(n,1);beats_used=nan(n,1);
opt=odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',2);
grid=(0:1000)';
for cell=1:n
    p=theta(cell,:);
    fb=struct('ionic_scale',p(1:6),'g_ionic',p(7), ...
        'land_params',struct('ca50',p(8),'TRPN_n',p(9), ...
        'nperm',p(10),'beta_0',p(11),'beta_1',p(12),'Tref',p(13)));
    state=base.state;prior=[];
    try
        for beat=1:160
            [t,y]=ode15s(@(t,x)human_bidirectional_rhs(t,x,1, ...
                true,true,base.par,base.isac_factor,[2.3,-2.4],fb), ...
                [0,1000],state,opt);
            state=y(end,:)';
            if mod(beat,10)==0
                z=interp1(t,y,grid,'pchip');
                now=[threshold_apd(z(:,1),-67),1000*max(z(:,6))];
                if ~isempty(prior) && beat>=40 && ...
                        abs(now(1)-prior(1))<.25 && ...
                        abs(now(2)-prior(2))/max(now(2),1e-6)<.002
                    success(cell)=true;break;
                end
                prior=now;
            end
        end
        beats_used(cell)=beat;
        if success(cell),states(:,cell)=state;end
    catch ME
        warning('Fresh cell %d prepace: %s',cell,ME.message);
    end
    if mod(cell,8)==0 || cell==n
        save(fullfile(out,'feedback_13param_event_margin_confirm_checkpoint.mat'), ...
            'theta','states','success','beats_used','lo','hi','-v7.3');
        fprintf('Event-margin source %d/%d valid=%d\n',cell,n,sum(success));
    end
end
save(fullfile(out,'feedback_13param_event_margin_confirm.mat'), ...
    'theta','states','success','beats_used','lo','hi','-v7.3');
end

function duration=threshold_apd(vm,threshold)
up=find(vm(1:end-1)<=threshold & vm(2:end)>threshold,1,'first');
if isempty(up),duration=nan;return;end
down=find(vm(up+1:end-1)>=threshold & vm(up+2:end)<threshold,1,'first');
if isempty(down),duration=nan;return;end
down=down+up;
tup=up+(threshold-vm(up))/(vm(up+1)-vm(up));
tdown=down+(threshold-vm(down))/(vm(down+1)-vm(down));
duration=tdown-tup;
end

