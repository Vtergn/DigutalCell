function dy= SVarDot_cai(t, y, cai, BAR,count)
% The function SVarDot finds the derivatives of the state variables P, AF,
% and hi. P represents the probability that a binding site along the thin
% filament is in the unblocked position. AF represents the probability of
% the formation of a strong, force-generating cross-bridge. The state
% variable hi represents the intrinsic half sarcomere length. 

% Code from Dupuis et al. 2018.

global params
global Vars

%% Get Model Parameters
%fitted parameters
C_s         = params.C_s; % units inverse of tension
a           = params.a; % units uM um^-1
b           = params.b; % units uM
n           = params.n; % unitless
v_max       = params.v_max; % units nm s^-1
% Calcium attach and detach rates from Rice et al 2008
Ca_at       = params.Ca_at; % units s^-1 
Ca_de       = params.Ca_de; % units s^-1 
%Landesberg and Sideman 1994 use values f_DA = 40, f_AD = 12.
f_DA        = params.f_DA; % units s^-1
f_AD        = params.f_AD; % units s^-1

h_SE0       = params.h_SE0; % nm.
F_iso       = params.F_iso; % force by a single cross-bridge (in Newtons)
SL0         = params.SL0;
% External length clamp for cross-model tests. The published cross-bridge
% equations below are unchanged; only the imposed isometric boundary moves.
if isfield(params,'length_pulse')
    p=params.length_pulse;
    ms=t*1000;
    up=max(0,min(1,(ms-p.onset_ms)/p.ramp_ms));
    down=max(0,min(1,(p.onset_ms+2*p.ramp_ms+p.hold_ms-ms)/p.ramp_ms));
    SL0=SL0*(1+(p.lambda_peak-1)*min(up,down));
end

%% Set the initial conditions for each iteration
% SL_init     = params.SL_init(end);
SL_init     = params.SL_init(count);
if isfield(params,'length_pulse')
    SL_init=SL0;
end

% Determine the overlap function
[Overlap, m_end, z_end, Ov_a] = Rice_Overlap_fun_nm(SL_init);
% determine the length of the initial conditions vector
leng        = params.leng;
dx          = 800/leng;
% determine the initial state variable vectors for each iteration
P           = y(1:leng,:); % unitless 
AF          = y(leng+1:2*leng,:); % unitless
hi          = y(2*leng+1); % nm

%% Determine the sarcomere length (and related variables)
AXX         = (cumsum(AF)*dx);
AX          = AXX(end);
S_titin     = 4.413*(exp(2.873*((max(1.845, SL_init/1000))-1.845))-1);
% Find the cumulative sum of AF at the proper index used to calculate h_short 
h_short     = (((params.Afterload-S_titin)*h_SE0)/(F_iso*AX)) + hi;
h_iso       =  SL0/2;
h_s         = min(h_short, h_iso);
SL          = h_s*2;
h_se        = h_s - hi;

% Parameter K_Ca depends on sarcomere length
K_Ca        = (-a*SL/1000 + b); % units uM

K_CaFac = 1.5; %1.2 
if BAR == 1, K_Ca = K_Ca * K_CaFac; end %1.15 %2
params.K_CaFac = K_CaFac;

% H determines the ratio of the length of the series elastic element to its
% length when fully extended (isometric).
H           = (h_s - hi)/h_SE0; % unitless

%% Calculated variables
% Determine the tension
F_XB        = F_iso*AF*H; % units Newtons
Sx          =0.5*(cumsum(F_XB))*dx;%+S_titin;
Sx(2:end,:) =Sx(1:end-1,:)+Sx(2:end,:);
% Determine the normlaized calcium concentration
Ca_transient = cai;
Ca_transient = Ca_transient * 1000;%from mMol to microMol
Ca           = Ca_transient/K_Ca;

% Determine the length of the single overlap region
[Overlap, m_end, z_end, Ov_a] = Rice_Overlap_fun_nm(SL);

%% Determine the derivatives of the state variables

dP  = (1 - P)*(Ca^n)*Ca_at - Ca_de*P.*exp(-n*C_s*Sx);
dAF = (1 - AF)*f_DA.*P.*Ov_a' - AF*f_AD;
dhi = (H - 1)*v_max; 
dy  = [dP; dAF; dhi];

ind = count;

if params.computeS == 1
    Vars.S_titin(ind,:)      = S_titin';
    Vars.Sx(ind,:)           = Sx';
end
params.SL_init(ind+1,:)    = SL;
params.HH(ind,:)         = H;

