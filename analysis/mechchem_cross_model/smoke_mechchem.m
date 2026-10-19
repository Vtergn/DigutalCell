% Minimal diagnostic using the published ORd-MechChem RHS and solver.
% Does not alter model equations. Fresh mechanical state is not prepaced.
here=fileparts(mfilename('fullpath'));
addpath(here);
cd(here);
Param; Variable;
global params Vars
params.CL=1000; params.computeS=1;
params.SSL=2205; params.SL0=2205; params.Afterload=70;
dt=0.003;
times=0:dt:1000;
nt=numel(times);
params.SL_init=zeros(nt+5,1); params.SL_init(1)=2205;
params.HH=zeros(nt+5,1);
Vars.S_titin=zeros(nt+5,1);
Vars.Sx=zeros(nt+5,params.leng);
Vars.SL=zeros(nt+5,1);
init=zeros(1,2*params.leng+1);
init(end)=2205/2-params.h_SE0;
load('X0ORD_CL1000.mat','X0ORD');
tic;
[T,Y1,Y]=Ode2Alt_EM('model_modif_bidir_BAR','SVarDot_cai', ...
    times,X0ORD(:)',init,dt,1e-3,1000,0,0);
elapsed=toc;
Ca=1000*Y1(:,6);
Vm=Y1(:,1);
S=Vars.Sx(1:numel(T),end)+Vars.S_titin(1:numel(T));
fprintf('SMOKE n=%d seconds=%.2f Vrange=[%.3f,%.3f] Carange=[%.4f,%.4f] Trange=[%.4f,%.4f] finite=%d\n', ...
    numel(T),elapsed,min(Vm),max(Vm),min(Ca),max(Ca),min(S),max(S), ...
    all(isfinite([Vm(:);Ca(:);S(:)])));
fprintf('SL observed range=[%.3f,%.3f] end=%.3f\n',min(params.SL_init(params.SL_init>0)),max(params.SL_init),params.SL_init(find(params.SL_init>0,1,'last')));
save('smoke_static_105_dt003_output.mat','T','Vm','Ca','S','elapsed','-v7');
