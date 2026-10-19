function check_flux_invariance()
% Verify the new feedback fields preserve the unmodified RHS and that the
% residual is exactly silent under no imposed stretch.
here=fileparts(mfilename('fullpath'));
root=fileparts(here);
addpath(fullfile(root,'Data','human_SAC_reference'));
addpath(fullfile(root,'Data','7fb','7fb-959277c48267', ...
    'Land_model_matlab','release'));
s=load(fullfile(here,'human_bidirectional_output','prepace_state.mat'), ...
    'state','par','isac_factor');
base=struct('g_ionic',1.2);
explicit=base;explicit.sac_gate=1;explicit.sac_residual_slope=0;
a=human_bidirectional_rhs(250,s.state,1.05,true,true, ...
    s.par,s.isac_factor,[2.3,-2.4],base);
b=human_bidirectional_rhs(250,s.state,1.05,true,true, ...
    s.par,s.isac_factor,[2.3,-2.4],explicit);
assert(max(abs(a-b))<1e-12);
changed=base;changed.sac_gate=.2;changed.sac_residual_slope=8;
c=human_bidirectional_rhs(250,s.state,1,true,true, ...
    s.par,s.isac_factor,[2.3,-2.4],base);
d=human_bidirectional_rhs(250,s.state,1,true,true, ...
    s.par,s.isac_factor,[2.3,-2.4],changed);
assert(max(abs(c-d))<1e-12);
fprintf('flux invariance checks passed\n');
end
