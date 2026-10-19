function Param

global params

params.SSL         = [2100]; % nm
params.SL0         = [2100];
params.SL_init     = [2100];
params.HH          = 0;
params.AF         = [];
params.leng        = 50; % nm   leng is the length of the half myosin
% Assumed sarcomere dimensions
params.bare        = 100; % nm
params.m_length    = 1600; % nm
params.thin_length = 1200; % nm
params.tspan       = linspace(0,1,1300); % s

% Fitted model parameters
params.v_max       = 5767; % um/s 
params.F_iso       = 0.29*5;%0.29*5; % *5  *7 % 0.08 0.29; %kPa/nm
params.C_s         = 0.02/5;%0.02/5; % /5   /7   0.0134  0.02; % 1/kPa
params.a           =  3.25/1.3; %/1.3; %/1.9; 2.9 %2.5; %3.25; % uM/um
params.b           =  8.56/1.1; %/1.1; %/1.9; 2.9 %8.56; % uM

params.n           = 3;
params.h_SE0       = 10; % 10 nm
params.s           = 0.05;
params.F_afterload = [17 30 42 55 66 79 100];
% Calcium attach and detach rates from Rice et al 2008
params.Ca_at       =  50*10*10; %*10*10  x 3 50*100; % initally 50; 50*4  *30
params.Ca_de       =  250*10; % 10 initially 250;   250/4    /3
%Landesberg and Sideman 1994 
params.f_DA        = 40; %40;
params.f_AD        = 12; %12;
% Calcium tranisent
params.t_start     = 0.05;
params.tau_1       = 0.02;
params.tau_2       = 0.11;
params.Ca_amp      = 1.45;
params.Ca_dias     = 0.09;

params.Afterload   = [];




