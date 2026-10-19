% Compare the differentiable SAC gain against the native MATLAB feedback.
here=fileparts(mfilename('fullpath'));
root=fileparts(fileparts(here));
addpath(fullfile(root,'Data','human_SAC_reference'));
addpath(fullfile(root,'Data','7fb','7fb-959277c48267', ...
    'Land_model_matlab','release'));
addpath(fileparts(here));
source=load(fullfile(here,'prepace_state_v5.mat'),'state');
par=[0.2203,10.3203,12.8208,1.4110,0.8742];
isac=[4.47,0.001248];
gains=[0.5,0.8,1.0,1.2];
dy=zeros(50,numel(gains));
for i=1:numel(gains)
    feedback=struct('g_ionic',gains(i));
    dy(:,i)=human_bidirectional_rhs(0.5,source.state,1.08,true,true, ...
        par,isac,[2.3,-2.4],feedback);
end
save(fullfile(here,'gain_reference.mat'),'gains','dy','-v7');
