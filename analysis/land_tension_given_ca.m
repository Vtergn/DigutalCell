function out = land_tension_given_ca(cai_uM, thetaM, stretches)
% True Land ODE conditioned on an observed periodic Ca transient.
% This avoids rerunning ToR-ORd for mechanical-only parameter refinement.
if nargin<3 || isempty(stretches), stretches=[1 1.1]; end
if numel(cai_uM)~=1001 || numel(thetaM)~=8
    error('Expected 1001 Ca samples and eight mechanical parameters');
end
cai_uM=double(cai_uM(:));
thetaM=double(thetaM(:)');
if any(~isfinite(cai_uM)) || any(cai_uM<=0)
    error('Invalid intracellular Ca trace');
end
time_ms=(0:1000)';
m=struct('Tref',thetaM(1),'ca50',thetaM(2),'TRPN_n',thetaM(3), ...
    'nperm',thetaM(4),'beta_0',thetaM(5),'beta_1',thetaM(6), ...
    'phi',thetaM(7),'perm50',thetaM(8));
Ta=zeros(numel(stretches),1001);
drift=zeros(numel(stretches),1);
for j=1:numel(stretches)
    lam=stretches(j);
    ca_at=@(s) interp1(time_ms,cai_uM,min(1000,mod(s,1000)),'pchip');
    rhs=@(s,y) Land2016_variable(s,y,ca_at(s),'intact',lam,0,m);
    for cycles=[5 7]
        [lt,ly]=ode15s(rhs,[0 1000*cycles],Land2016_variable());
        y=interp1(lt,ly,1000*(cycles-1)+time_ms,'pchip');
        yp=interp1(lt,ly,1000*(cycles-2)+time_ms,'pchip');
        current=zeros(size(time_ms)); old=current;
        for k=1:numel(time_ms)
            [~,~,current(k),~]=Land2016_variable(1000*(cycles-1)+time_ms(k), ...
                y(k,:),cai_uM(k),'intact',lam,0,m);
            [~,~,old(k),~]=Land2016_variable(1000*(cycles-2)+time_ms(k), ...
                yp(k,:),cai_uM(k),'intact',lam,0,m);
        end
        drift(j)=max(abs(current-old))/max(current);
        if drift(j)<=.01, break; end
    end
    if any(~isfinite(current)) || max(current)<=0 || drift(j)>.01
        error('Land simulation failed quality check at lambda %.3f',lam);
    end
    Ta(j,:)=current;
end
out=struct('time_ms',time_ms,'lambda',stretches,'Ta_kPa',Ta, ...
    'land_cycle_drift',drift);
end
