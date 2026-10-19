function run_static_cohort(ncells)
% Independent ORd-MechChem truth. Published RHS/solver, published isometric
% initial sarcomere-length boundary; no acute-pulse modification is used.
if nargin<1, ncells=20; end
here=fileparts(mfilename('fullpath'));
addpath(here);
out=fullfile(here,'output');
if ~exist(out,'dir'), mkdir(out); end
rng(20261003);
names={'Ca_at','Ca_de','f_DA','f_AD','a','b','C_s','F_iso'};
scale=[.18,.18,.18,.18,.08,.08,.20,.15];
u=zeros(ncells,8);
for j=1:8
    u(:,j)=((randperm(ncells)'-1)+rand(ncells,1))/ncells;
end
theta=1+(2*u-1).*scale;
lengths=[2100,2205,2257.5,2310]; % nm: lambda 1,1.05,1.075,1.10
% Published solver's final saved point is just below 1000 ms.
grid=(0:999)'; dt=.002; times=0:dt:1000; nt=numel(times);
load(fullfile(here,'X0ORD_CL1000.mat'),'X0ORD');
features=nan(ncells,numel(lengths),14);
ok=false(ncells,numel(lengths));
for i=1:ncells
    for j=1:numel(lengths)
        Param; Variable;
        global params Vars
        assert(~isfield(params,'length_pulse'), ...
            'Acute pulse boundary must be disabled for static-preload truth');
        for q=1:8
            params.(names{q})=params.(names{q})*theta(i,q);
        end
        sl=lengths(j);
        params.CL=1000; params.computeS=1;
        params.SSL=sl; params.SL0=sl; params.Afterload=70;
        params.SL_init=zeros(nt+5,1); params.SL_init(1)=sl;
        params.HH=zeros(nt+5,1);
        Vars.S_titin=zeros(nt+5,1);
        Vars.Sx=zeros(nt+5,params.leng);
        Vars.SL=zeros(nt+5,1);
        init=zeros(1,2*params.leng+1);
        init(end)=sl/2-params.h_SE0;
        tic;
        try
            [T,Y1,Y]=Ode2Alt_EM('model_modif_bidir_BAR', ...
                'SVarDot_cai',times,X0ORD(:)',init,dt,1e-3,1000,0,0);
            elapsed=toc;
            vm=Y1(:,1); ca=1000*Y1(:,6);
            tension=Vars.Sx(1:numel(T),end)+Vars.S_titin(1:numel(T));
            if T(end)<999.9 || ~all(isfinite([vm;ca;tension])) || ...
                    max(abs(vm))>200 || min(tension)<0 || min(ca)<0
                error('nonphysiologic or incomplete trace');
            end
            trace=interp1(T,[vm,ca,tension],grid,'linear');
            if ~all(isfinite(trace(:))), error('nonfinite sampled trace'); end
            v=trace(:,1); c=trace(:,2); s=trace(:,3);
            apd=sum(v>-67);
            features(i,j,:)=[apd,max(c),max(s),min(v),max(v), ...
                c(201),c(401),c(601),s(201),s(401),s(601), ...
                trapz(grid,c)/1000,trapz(grid,s)/1000,s(end)];
            ok(i,j)=true;
            writematrix([grid,trace],fullfile(out, ...
                sprintf('cell%03d_load%02d.csv',i,j)));
            fprintf('CELL %d/%d LOAD %d/4 OK APD=%.1f Ca=%.4f T=%.3f time=%.1fs\n', ...
                i,ncells,j,apd,max(c),max(s),elapsed);
        catch ME
            fprintf('CELL %d/%d LOAD %d/4 FAILED %s\n', ...
                i,ncells,j,ME.message);
        end
        clear T Y1 Y vm ca tension trace
    end
    save(fullfile(out,'cohort_checkpoint.mat'),'theta','names', ...
        'lengths','features','ok','grid','-v7');
end
writematrix([theta,reshape(features,ncells,[])], ...
    fullfile(out,'cohort_features.csv'));
fprintf('COHORT COMPLETE full cells=%d/%d\n',sum(all(ok,2)),ncells);
end
