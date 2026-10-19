% Native MATLAB trajectory and derivative reference for the PyTorch port.
here = fileparts(mfilename('fullpath'));
root = fileparts(fileparts(here));
addpath(fullfile(root,'Data','human_SAC_reference'));
addpath(fullfile(root,'Data','7fb','7fb-959277c48267', ...
    'Land_model_matlab','release'));
addpath(fileparts(here));
source = load(fullfile(here,'prepace_state_v5.mat'),'state');
par = [0.2203,10.3203,12.8208,1.4110,0.8742];
isac = [4.47,0.001248]; beta = [2.3,-2.4];
times = (0:0.02:2)';
lengths = [1.00,1.08];
states = zeros(numel(times),50,numel(lengths));
derivs = zeros(size(states));
options = odeset('RelTol',1e-10,'AbsTol',1e-12,'MaxStep',0.01);
for j = 1:numel(lengths)
    ell = lengths(j);
    fun = @(t,y) human_bidirectional_rhs(t,y,ell,true,true, ...
        par,isac,beta);
    [~,yy] = ode15s(fun,times,source.state,options);
    states(:,:,j) = yy;
    for k = 1:numel(times)
        derivs(k,:,j) = fun(times(k),yy(k,:)')';
    end
end
save(fullfile(here,'trajectory_reference.mat'), ...
    'times','lengths','states','derivs','-v7');
