% Local parameterized adaptation of the downloaded Land2016.m model.
% Original: Data/7fb/7fb-959277c48267/Land_model_matlab/release/Land2016.m
% Changes: function name and optional theta struct for cohort generation.
% Land et al. 2016: Human contraction at 37 degrees C
% Arguments:
% t: time in ms
% y: state variables
% Cai: calcium (micromolar). Can give a value (at current time) or a vector (1 ms spacing will be assumed)
% mode = "intact".  "skinned" or "intact"
% lambda = 1: extension ratio SL/resting SL
% dlambda_dt = 0: velocity (/ms)
% 
% Output:
% dydt: 
% T: Ta+Tp
% Ta: active tension
% Tp: passive tension
%
% Call with no arguments to get a reasonable (though non-paced) initial condition
function [dydt, T, Ta, Tp] = Land2016_variable(t,y,Cai,mode,lambda,dlambda_dt,theta)

%-------------------------------------------------------------------------------
% Initial conditions and input defaults
%-------------------------------------------------------------------------------
if nargin==0 || isempty(t)  
   dydt=[0 0 0 1 0 0 0];
  return
end

if nargin < 6 % no velocity data -> fixed length
  dlambda_dt = 0;
end

if nargin < 5 % no length data -> fixed resting length
  lambda = 1;
end

if length(Cai) > 1 % Cai is a vector: assume it is 1 ms spaced time data
  Cai = interp1(0:length(Cai),[Cai Cai(1)], mod(t,length(Cai)) );
end

dydt = zeros(size(y));
%-------------------------------------------------------------------------------
% Parameters
%-------------------------------------------------------------------------------

perm50 = 0.35;
TRPN_n = 2;
koff    = 0.1;
dr = 0.25;
wfrac = 0.5;

TOT_A = 25;
ktm_unblock = 1;

beta_1 = -2.4;
beta_0 = 2.3;
gamma = 0.0085;
gamma_wu = 0.615;
phi = 2.23;  

if strcmp(mode,'skinned')
  nperm = 2.2;
  ca50 = 2.5;
  Tref = 40.5;
  nu = 1;
  mu = 1;
else
  nperm = 5;
  ca50 = 0.805;
  Tref = 120;
  nu = 7;
  mu = 3;
end

if nargin>=7 && ~isempty(theta)
  fields = fieldnames(theta);
  allowed = {'perm50','TRPN_n','koff','dr','wfrac','TOT_A', ...
    'ktm_unblock','beta_0','beta_1','gamma','gamma_wu','phi', ...
    'nperm','ca50','Tref','nu','mu'};
  for idx = 1:numel(fields)
    field = fields{idx};
    if ~ismember(field,allowed)
      error('Unsupported Land parameter: %s',field);
    end
    value = theta.(field);
    if ~isscalar(value) || ~isfinite(value)
      error('Invalid Land parameter %s',field);
    end
    switch field
      case 'perm50', perm50 = value;
      case 'TRPN_n', TRPN_n = value;
      case 'koff', koff = value;
      case 'dr', dr = value;
      case 'wfrac', wfrac = value;
      case 'TOT_A', TOT_A = value;
      case 'ktm_unblock', ktm_unblock = value;
      case 'beta_0', beta_0 = value;
      case 'beta_1', beta_1 = value;
      case 'gamma', gamma = value;
      case 'gamma_wu', gamma_wu = value;
      case 'phi', phi = value;
      case 'nperm', nperm = value;
      case 'ca50', ca50 = value;
      case 'Tref', Tref = value;
      case 'nu', nu = value;
      case 'mu', mu = value;
    end
  end
end

k_ws = 0.004 * mu;
k_uw = 0.026  * nu;


cdw = phi * k_uw * (1-dr)*(1-wfrac) /  ((1-dr)*wfrac);
cds = phi * k_ws * (1-dr)*wfrac / dr;


k_wu = k_uw * (1/wfrac - 1) - k_ws;
k_su = k_ws * (1/dr - 1) * wfrac; 
A = (0.25 * TOT_A) / ((1-dr)*wfrac + dr) * (dr/0.25);

 

% State Variables
XS   = max(0,y(1));
XW   = max(0,y(2));
TRPN = max(0,y(3));
TmBlocked = y(4);
ZETAS = y(5);
ZETAW= y(6);



%-------------------------------------------------------------------------------
% XB model
%-------------------------------------------------------------------------------

lambda = min(1.2,lambda);
Lfac =  max(0, 1 + beta_0 * (lambda + min(0.87,lambda) - 1.87) );

XU   = (1 - TmBlocked) - XW - XS; % unattached available xb = all - tm blocked - already prepowerstroke - already post-poststroke - no overlap

xb_ws = k_ws * XW;
xb_uw = k_uw * XU ;
xb_wu = k_wu * XW;
xb_su = k_su * XS;

gamma_rate  = gamma * max( (ZETAS > 0) .* ZETAS, (ZETAS < -1) .* (-ZETAS-1) );
xb_su_gamma = gamma_rate * XS;

gamma_rate_w  = gamma_wu * abs(ZETAW); % weak xbs don't like being strained
xb_wu_gamma = gamma_rate_w * XW;      

dydt(1)  = xb_ws - xb_su - xb_su_gamma;
dydt(2)  = xb_uw - xb_wu - xb_ws - xb_wu_gamma;

ca50 = ca50 + beta_1*min(0.2,lambda - 1);
dydt(3)  = koff * ( (Cai/ca50)^TRPN_n * (1-TRPN) - TRPN); % untouched

XSSS = dr * 0.5;
XWSS = (1-dr) * wfrac * 0.5;
ktm_block = ktm_unblock * (perm50^nperm) *  0.5 / (0.5 - XSSS - XWSS);

dydt(4)  = ktm_block * min(100, (TRPN^-(nperm/2))) * XU  - ktm_unblock * (TRPN^(nperm/2)) *  TmBlocked;

%-------------------------------------------------------------------------------
% Velocity dependence -- assumes distortion resets on W->S
%-------------------------------------------------------------------------------

dydt(5) = A * dlambda_dt - cds * ZETAS;% - gamma_rate * ZETAS;
dydt(6) = A * dlambda_dt - cdw * ZETAW;% - gamma_rate_w * ZETAW;

%-------------------------------------------------------------------------------
% Passive model
%-------------------------------------------------------------------------------

par_k = 7;
b = 9.1;
eta_l = 200;
eta_s = 20;
a = 2.1;

Cd = y(7);
C = lambda - 1;

if (C - Cd) < 0
 eta = eta_s;
else
 eta = eta_l;
end
    
dCd_dt = par_k * (C - Cd) / eta; % F2=Fd
dydt(7) = dCd_dt;

Fd = eta * dCd_dt;
F1 = (exp( b * C) - 1);
Tp = a * (F1 + Fd);

%-------------------------------------------------------------------------------
% Active and Total Force
%-------------------------------------------------------------------------------

Ta = Lfac * (Tref/dr) * ( (ZETAS+1) * XS + (ZETAW) * XW );
T = Ta + Tp;
