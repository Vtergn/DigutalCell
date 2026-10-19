%% Electromechanical model runner 
% model_modif_bidir_BAR: electrical model
% SVarDot_cai: mechanical model 
% Y1 = state variables vector for electrics
% Y = state variables vector for mechanics

Param;

global params

params.CL = CL;
params.prepaced = prepaced;
params.computeS = computeS; 

SarcL = []; S = []; Ca_tran = []; 
JREL = []; JUP = []; ICAL = []; ICaL=[]; Jup=[]; Jrel=[]; XB_AX = []; INA = []; IKS = []; INAK = []; IKB = []; FJUPP = [];

SSL = Sarcomere_length;
params.SSL = Sarcomere_length;
params.SL_init = Sarcomere_length;
params.SL0 = Sarcomere_length;
params.SL0 = SSL;
params.Afterload = F_afterload;

Variable;
global Vars

SL = SSL;
h_s = SL/2; % half sarcomere length in nm
h_SE0 = params.h_SE0;

%% Set initial conditions

leng = params.leng;
init(1:2*leng) = 0;
init(2*leng+1) = h_s - h_SE0; %

tspan = []; elecTS = 0.001; mechaTS = 0.001; % Set the electrical and mechanical time steps, can be different
tspan_beat = [0:0.001:100 100.01:elecTS:CL];
for k = 1:nb_beat
    tspan = [tspan tspan_beat+(k-1)*CL];
end

params.tspan = tspan;
numtp = length(tspan);

%% Initialize vectors
if params.computeS == 1;
    Vars.S_titin      = zeros(numtp, 1);
    Vars.SL           = zeros(numtp, 1);
    Vars.Sx           = zeros(numtp, params.leng);
end

params.SL_init    = zeros(numtp, 1);
params.SL_init(1) = Sarcomere_length;
params.HH         = zeros(numtp, 1);

%% Solve the differential equations for the state variables

if params.prepaced == 0 % compute the EM state from scratch
%     X0ORD = TOR_X0_bidir(CL,BAR); % compute the EP state 
    load(['X0ORD_CL' num2str(CL)]) % load a pre-computed EP state 
    [T, Y1, Y] = Ode2Alt_EM('model_modif_bidir_BAR', 'SVarDot_cai', tspan, X0ORD, init, mechaTS,1E-3, CL, BAR,CL*(nb_beat - 5));
end

if params.prepaced == 1 

    load(prepace_state,'Y','Y1') % load the prepaced state variables
    ep_state = Y1(end,:); mech_state = Y(end, :);
    [T, Y1, Y] = Ode2Alt_EM('model_modif_bidir_BAR', 'SVarDot_cai', tspan, ep_state, mech_state, mechaTS,1E-3, CL, BAR,CL*(nb_beat - 5));
    
end

%% Retrieve variables 

fluxes = [];
if flux
    for i=[1:size(Y1,1)];
        IsJs=model_modif_bidir_BAR(T(i),Y1(i,:),[],BAR,[],0);
        INa(i)=IsJs(1);
        %     INaL(i)=IsJs(2);
        %     Ito(i)=IsJs(3);
        ICaL(i)=IsJs(4);
        %     IKr(i)=IsJs(5);
        IKs(i)=IsJs(6);
        %     IK1(i)=IsJs(7);
        %     INaCa_i(i)=IsJs(8);
        %     INaCa_ss(i)=IsJs(9);
        INaK(i)=IsJs(10);
        IKb(i)=IsJs(11);
        %     INab(i)=IsJs(12);
            ICab(i)=IsJs(13);
        %     IpCa(i)=IsJs(14);
        %     Jdiff(i)=IsJs(15);
        %     JdiffNa(i)=IsJs(16);
        %     JdiffK(i)=IsJs(17);
        Jup(i)=IsJs(18);
            Jleak(i)=IsJs(19);
        %     Jtr(i)=IsJs(20);
        Jrel(i)=IsJs(21);
        %     CaMKa(i)=IsJs(22);
        %     Istim(i)=IsJs(23);
        %     fJupp(i) = IsJs(24);
    end
    cansr=Y1(:,8);
    fluxes = [INa; ICaL; INaK; IKb; ICab; Jup; Jleak; Jrel; cansr'];
end

tspan = T;
count = length(tspan);

if params.computeS == 1
    
    Vars.S_titin      = Vars.S_titin(1:count,:);
    Vars.Sx           = Vars.Sx(1:count,:);
    Vars.SL           = Vars.SL(1:count,:);
    
    S_titin = Vars.S_titin;
    S = [(Vars.Sx(:, leng)+S_titin), S];
    SarcL = Vars.SL;
end

Ca = Y1(:,6) * 1000; Vm = Y1(:,1); CaTn = Y1(:,42);


%% Save data
fname = 'EMsimulation';
save(fname,saved_variables{:},'-v7.3')

clearvars -except Sarcomere_length nb_beat saveVar flux saved_variables CL BAR ab
