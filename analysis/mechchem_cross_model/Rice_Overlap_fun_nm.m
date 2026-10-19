function [Overlap, m_end, z_end, Ov_a] = Rice_Overlap_fun_nm(SL)
% Rice_Overlap_fun(SL) = Length of the single overlap region in nanometers.
% This formulation is taken from Rice et al. 2008. 
global params

% SLn = SL*1000;
SLn = SL;

nsamples    = params.leng;
bare        = params.bare;
m_length    = params.m_length;
thin_length = params.thin_length;

z_end = min((m_length/2), (SLn/2));
m_end = max((SLn/2 - (SLn - thin_length)), (bare/2));
Ov    = z_end - m_end;
Ovs   = Ov.*(Ov>=0);

s           = params.s;
window = [1:1:nsamples];
x      = (m_length/2)*(window/nsamples);
Ov_a   = ((tanh(s*(x-m_end)) - tanh(s*(x-z_end)))/2).*(Ov>=0);

Overlap = round(Ovs);