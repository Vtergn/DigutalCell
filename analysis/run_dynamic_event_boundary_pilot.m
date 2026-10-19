function run_dynamic_event_boundary_pilot(mode)
% Exploratory fixed-state parameter scan, per frozen protocol.
if nargin<1,mode='acute';end
here=fileparts(mfilename('fullpath'));root=fileparts(here);
out=fullfile(here,'human_bidirectional_output');
addpath(here);
addpath(fullfile(root,'Data','human_SAC_reference'));
addpath(fullfile(root,'Data','7fb','7fb-959277c48267', ...
    'Land_model_matlab','release'));
base=load(fullfile(out,'prepace_state.mat'),'par','isac_factor');
inv=load(fullfile(out,'dynamic_inverse_calibration.mat'), ...
    'states','tab','map','unique_theta');
cohort=load(fullfile(out,'feedback_13param_confirm_cohort.mat'),'theta');
truth=load(fullfile(out,'dynamic_discrepancy_cohort.mat'),'waveforms');
cells=[8,43,57];indices=[6,4,7];
fractions=0:.125:1;grid=(0:1000)';pulse=[300,1.10,100];
opt=odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',1);
rows=struct([]);q=0;
if strcmp(mode,'repace') || strcmp(mode,'repace_refined')
    if strcmp(mode,'repace_refined')
        flanks={[.1625,.175],[.0875,.1],[.8125,.825]};
        pacefile='dynamic_event_boundary_repace_refined.mat';
    else
        flanks={[.125,.25],[0,.125],[.75,.875]};
        pacefile='dynamic_event_boundary_repace.mat';
    end
    checkpoints=[0,20,60];
    paceopt=odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',2);
    follow=struct([]);n=0;
    for j=1:numel(cells)
        cellid=cells(j);idx=indices(j);
        r=find(inv.tab(:,1)==cellid & inv.tab(:,2)==2 & ...
            inv.tab(:,3)==1);assert(numel(r)==1);
        u=inv.map(r);est=inv.unique_theta(u,:);
        actual=cohort.theta(cellid,:);
        target=squeeze(truth.waveforms(:,:,4,2,cellid));
        for s=flanks{j}
            theta=est;theta(idx)=est(idx)+s*(actual(idx)-est(idx));
            fb=feedback_struct(theta);fb.sac_saturation=.05;
            state=inv.states(:,u);last_rms=nan;
            for beat=0:max(checkpoints)
                if beat>0
                    prior=state;
                    [~,yy]=ode15s(@(t,y)human_bidirectional_rhs(t,y,1, ...
                        true,true,base.par,base.isac_factor,[2.3,-2.4],fb), ...
                        [0,1000],state,paceopt);
                    state=yy(end,:)';
                    scale=max(max(abs(state),abs(prior)),1e-6);
                    last_rms=sqrt(mean(((state-prior)./scale).^2));
                end
                if ~ismember(beat,checkpoints),continue;end
                n=n+1;follow(n).cell=cellid;follow(n).index=idx;
                follow(n).fraction=s;follow(n).parameter=theta(idx);
                follow(n).beats=beat;follow(n).last_state_rel_rms=last_rms;
                follow(n).ok=false;
                try
                    zz=simulate_hold(state,pulse,fb,base,opt,grid);
                    v=zz(:,1);win=301:602;
                    crossings=find(v(win(1:end-1))<=-50 & ...
                        v(win(2:end))>-50);
                    follow(n).crossing_ms=nan;
                    if ~isempty(crossings)
                        follow(n).crossing_ms=300+crossings(1);
                    end
                    follow(n).duration_above_minus50_ms=sum(v(win)>-50);
                    follow(n).v_mae_to_true_mV= ...
                        mean(abs(v(win)-target(win,1)));
                    follow(n).ok=true;
                catch ME
                    follow(n).error=ME.message;
                end
                fprintf('REPACE cell=%d s=%.3f beats=%d duration=%g rms=%g ok=%d\n', ...
                    cellid,s,beat,safe_field(follow(n), ...
                    'duration_above_minus50_ms'),last_rms,follow(n).ok);
                save(fullfile(out,pacefile), ...
                    'follow','flanks','checkpoints','-v7.3');
            end
        end
    end
    return;
end
if strcmp(mode,'strict')
    strictopt=odeset('RelTol',1e-7,'AbsTol',1e-9,'MaxStep',.5);
    flanks={[.1625,.175],[.0875,.1],[.8125,.825]};
    checks=struct([]);n=0;
    for j=1:numel(cells)
        cellid=cells(j);idx=indices(j);
        r=find(inv.tab(:,1)==cellid & inv.tab(:,2)==2 & ...
            inv.tab(:,3)==1);assert(numel(r)==1);
        u=inv.map(r);est=inv.unique_theta(u,:);
        actual=cohort.theta(cellid,:);initial=inv.states(:,u);
        for s=flanks{j}
            n=n+1;theta=est;
            theta(idx)=est(idx)+s*(actual(idx)-est(idx));
            fb=feedback_struct(theta);fb.sac_saturation=.05;
            checks(n).cell=cellid;checks(n).fraction=s;
            checks(n).parameter=theta(idx);checks(n).ok=false;
            try
                a=simulate_hold(initial,pulse,fb,base,opt,grid);
                b=simulate_hold(initial,pulse,fb,base,strictopt,grid);
                win=301:602;
                checks(n).duration_default=sum(a(win,1)>-50);
                checks(n).duration_strict=sum(b(win,1)>-50);
                checks(n).voltage_mae_default_vs_strict_mV= ...
                    mean(abs(a(win,1)-b(win,1)));
                checks(n).voltage_max_abs_difference_mV= ...
                    max(abs(a(win,1)-b(win,1)));
                checks(n).ok=true;
            catch ME
                checks(n).error=ME.message;
            end
            fprintf('STRICT cell=%d s=%.3f default=%g strict=%g Vdiff=%g ok=%d\n', ...
                cellid,s,safe_field(checks(n),'duration_default'), ...
                safe_field(checks(n),'duration_strict'), ...
                safe_field(checks(n),'voltage_mae_default_vs_strict_mV'), ...
                checks(n).ok);
        end
    end
    save(fullfile(out,'dynamic_event_boundary_strict.mat'), ...
        'checks','flanks','-v7.3');
    return;
end
if strcmp(mode,'waveform')
    flanks={[.1625,.175],[.0875,.1],[.8125,.825]};
    profiles=struct([]);n=0;
    for j=1:numel(cells)
        cellid=cells(j);idx=indices(j);
        r=find(inv.tab(:,1)==cellid & inv.tab(:,2)==2 & ...
            inv.tab(:,3)==1);assert(numel(r)==1);
        u=inv.map(r);est=inv.unique_theta(u,:);
        actual=cohort.theta(cellid,:);initial=inv.states(:,u);
        for s=flanks{j}
            n=n+1;theta=est;
            theta(idx)=est(idx)+s*(actual(idx)-est(idx));
            fb=feedback_struct(theta);fb.sac_saturation=.05;
            yy=simulate_hold(initial,pulse,fb,base,opt,grid);
            profiles(n).cell=cellid;profiles(n).fraction=s;
            profiles(n).parameter=theta(idx);
            profiles(n).voltage_mV=yy(:,1);
            profiles(n).calcium_uM=1000*yy(:,6);
        end
    end
    save(fullfile(out,'dynamic_event_boundary_waveforms.mat'), ...
        'profiles','pulse','grid','-v7.3');
    return;
end
for j=1:numel(cells)
    cellid=cells(j);idx=indices(j);
    r=find(inv.tab(:,1)==cellid & inv.tab(:,2)==2 & ...
        inv.tab(:,3)==1);assert(numel(r)==1);
    u=inv.map(r);est=inv.unique_theta(u,:);
    actual=cohort.theta(cellid,:);
    initial=inv.states(:,u);
    target=squeeze(truth.waveforms(:,:,4,2,cellid));
    these_fractions=fractions;
    if strcmp(mode,'refine')
        spans={[.125,.25],[0,.125],[.75,.875]};
        these_fractions=linspace(spans{j}(1),spans{j}(2),11);
    end
    for s=these_fractions
        q=q+1;theta=est;
        theta(idx)=est(idx)+s*(actual(idx)-est(idx));
        fb=feedback_struct(theta);
        fb.sac_saturation=.05;
        rows(q).cell=cellid;rows(q).index=idx;
        rows(q).fraction=s;rows(q).parameter=theta(idx);
        rows(q).ok=false;
        try
            yy=simulate_hold(initial,pulse,fb,base,opt,grid);
            v=yy(:,1);ca=1000*yy(:,6);
            win=301:602;
            crossings=find(v(win(1:end-1))<=-50 & ...
                v(win(2:end))>-50);
            if isempty(crossings)
                rows(q).crossing_ms=nan;
                marker=390;
            else
                rows(q).crossing_ms=300+crossings(1);
                marker=max(305,rows(q).crossing_ms-5);
            end
            rows(q).duration_above_minus50_ms=sum(v(win)>-50);
            rows(q).peak_v_mV=max(v(win));
            rows(q).min_v_mV=min(v(win));
            rows(q).peak_ca_uM=max(ca(win));
            rows(q).v_mae_to_true_mV= ...
                mean(abs(v(win)-target(win,1)));
            rows(q).marker_ms=marker;
            rows(q).marker_v_mV=v(marker+1);
            [rows(q).marker_dvdt,rows(q).marker_direct_stretch_dvdt, ...
                rows(q).marker_INaL,rows(q).marker_IKr, ...
                rows(q).marker_ICaL,rows(q).marker_INaCa]= ...
                current_diagnostic(marker,yy(marker+1,:)',fb,base);
            [rows(q).fixed390_dvdt,rows(q).fixed390_direct_stretch_dvdt, ...
                rows(q).fixed390_INaL,rows(q).fixed390_IKr, ...
                rows(q).fixed390_ICaL,rows(q).fixed390_INaCa]= ...
                current_diagnostic(390,yy(391,:)',fb,base);
            rows(q).ok=true;
        catch ME
            rows(q).error=ME.message;
        end
        fprintf('EVENT_BOUNDARY cell=%d idx=%d s=%.3f duration=%g Vmae=%g ok=%d\n', ...
            cellid,idx,s,safe_field(rows(q),'duration_above_minus50_ms'), ...
            safe_field(rows(q),'v_mae_to_true_mV'),rows(q).ok);
        filename='dynamic_event_boundary_pilot.mat';
        if strcmp(mode,'refine')
            filename='dynamic_event_boundary_refine.mat';
        end
        save(fullfile(out,filename), ...
            'rows','cells','indices','fractions','pulse','-v7.3');
    end
end
end

function value=safe_field(row,name)
value=nan;if row.ok,value=row.(name);end
end

function fb=feedback_struct(p)
fb=struct('ionic_scale',p(1:6),'g_ionic',p(7), ...
    'land_params',struct('ca50',p(8),'TRPN_n',p(9), ...
    'nperm',p(10),'beta_0',p(11),'beta_1',p(12),'Tref',p(13)));
end

function states=simulate_hold(initial,pulse,fb,base,opt,grid)
o=pulse(1);lambda=pulse(2);h=pulse(3);
breaks=[0,o,o+1,o+1+h,o+2+h,1000];
profile=@(t) length_schedule(t,pulse);
current=initial;all_t=[];all_y=[];
for k=1:numel(breaks)-1
    [t,y]=ode15s(@(t,y)human_bidirectional_rhs(t,y,profile, ...
        true,true,base.par,base.isac_factor,[2.3,-2.4],fb), ...
        breaks(k:k+1),current,opt);
    current=y(end,:)';
    if ~isempty(all_t),t=t(2:end);y=y(2:end,:);end
    all_t=[all_t;t];all_y=[all_y;y]; %#ok<AGROW>
end
states=interp1(all_t,all_y,grid,'pchip');
v=states(:,1);ca=1000*states(:,6);
if ~all(isfinite(states),'all') || min(v)<-150 || max(v)>100 || ...
        min(ca)<0 || max(ca)>10
    error('nonphysiologic or nonfinite trace');
end
end

function z=length_schedule(t,pulse)
o=pulse(1);amp=pulse(2)-1;h=pulse(3);
if t<o || t>=o+2+h,z=[1,0];
elseif t<o+1,z=[1+amp*(t-o),amp];
elseif t<o+1+h,z=[1+amp,0];
else,z=[1+amp*(o+2+h-t),-amp];end
end

function [dv,stretch,inal,ikr,ical,incx]=current_diagnostic(t,y,fb,base)
z=length_schedule(t,[300,1.10,100]);
d=z(1)-1;
lambda_eff=1+d*(1+.1/fb.sac_saturation)/(1+abs(d)/fb.sac_saturation);
off=model_SAC_hetero(t,y(1:43),0,1,base.par, ...
    base.isac_factor,-1,-1,0,fb.ionic_scale);
on=model_SAC_hetero(t,y(1:43),0,lambda_eff,base.par, ...
    base.isac_factor,-1,-1,0,fb.ionic_scale);
currents=off+fb.g_ionic*(on-off);
inal=currents(2);ikr=currents(5);ical=currents(4);
incx=currents(8)+currents(9);
rhs_on=human_bidirectional_rhs(t,y, ...
    @(tt)length_schedule(tt,[300,1.10,100]),true,true, ...
    base.par,base.isac_factor,[2.3,-2.4],fb);
rhs_off=human_bidirectional_rhs(t,y,1,true,true, ...
    base.par,base.isac_factor,[2.3,-2.4],fb);
dv=rhs_on(1);stretch=rhs_on(1)-rhs_off(1);
end
