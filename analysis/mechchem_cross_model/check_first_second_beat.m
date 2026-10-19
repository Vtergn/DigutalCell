% Check whether the no-large-prepace B cohort's first beat is stationary.
here=fileparts(mfilename('fullpath')); addpath(here); cd(here);
load('smoke_output.mat','Y1','Y','T','Vm','Ca','S');
first_t=T; first_vm=Vm; first_ca=Ca; first_s=S;
ep=Y1(end,:); mech=Y(end,:);
clear Y1 Y T Vm Ca S
Param; Variable; global params Vars
sl=2100; dt=.002; times=0:dt:1000; nt=numel(times);
params.CL=1000;params.computeS=1;params.SSL=sl;
params.SL0=sl;params.Afterload=70;
params.SL_init=zeros(nt+5,1);params.SL_init(1)=sl;
params.HH=zeros(nt+5,1);
Vars.S_titin=zeros(nt+5,1);Vars.Sx=zeros(nt+5,params.leng);
Vars.SL=zeros(nt+5,1);
[T,Y1,Y]=Ode2Alt_EM('model_modif_bidir_BAR','SVarDot_cai', ...
    times,ep,mech,dt,1e-3,1000,0,0);
vm=Y1(:,1);ca=1000*Y1(:,6);
s=Vars.Sx(1:numel(T),end)+Vars.S_titin(1:numel(T));
grid=(0:999)';
one=interp1(first_t,[first_vm,first_ca,first_s],grid);
two=interp1(T,[vm,ca,s],grid);
fprintf('BEAT1 APD=%d Ca=%.4f T=%.3f; BEAT2 APD=%d Ca=%.4f T=%.3f; rel_changes Ca=%.3f T=%.3f\n', ...
    sum(one(:,1)>-67),max(one(:,2)),max(one(:,3)), ...
    sum(two(:,1)>-67),max(two(:,2)),max(two(:,3)), ...
    (max(two(:,2))-max(one(:,2)))/max(one(:,2)), ...
    (max(two(:,3))-max(one(:,3)))/max(one(:,3)));
save('beat_stability.mat','grid','one','two','-v7');
