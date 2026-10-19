function X0 = ORd_X0_bidir(flag, varargin)

%initial conditions for state variables
if flag == 0
    v=-87;
    nai=7;
    nass=nai;
    ki=145;
    kss=ki;
    cai= 1.0e-4; %3.5e-4; %1.0e-4;
    cass=cai;
    cansr=1.2;
    cajsr=cansr;
    m=0;
    hf=1;
    hs=1;
    j=1;
    hsp=1;
    jp=1;
    mL=0;
    hL=1;
    hLp=1;
    a=0;
    iF=1;
    iS=1;
    ap=0;
    iFp=1;
    iSp=1;
    d=0;
    ff=1;
    fs=1;
    fcaf=1;
    fcas=1;
    jca=1;
    nca=0;
    ffp=1;
    fcafp=1;
    xrf=0;
    xrs=0;
    xs1=0;
    xs2=0;
    xk1=1;
    Jrelnp=0;
    Jrelp=0;
    CaMKt=0;
    CaTn = 1.0e-4; %1.0e-3;
    CaCalm = 1.0e-4;% 1.0e-3;
    %X0 is the vector for initial sconditions for state variables
    X0=[v nai nass ki kss cai cass cansr cajsr m hf hs j hsp jp mL hL hLp a iF iS ap iFp iSp d ff fs fcaf fcas jca nca ffp fcafp xrf xrs xs1 xs2 xk1 Jrelnp Jrelp CaMKt CaTn CaCalm]';
else
    %% Get to an already stable stystem
    X0 = varargin{1}; CL = varargin{2};
    if length(varargin)>2, BAR = varargin{3}; else BAR = 0; end
    
    beats = 700; output = [];   %700 beats
    for B=[1:beats]
        [time X]=ode15s(@model_modif_orig_BAR,[0 CL],X0,[],BAR,1); %model_modif_bidir(t,X,P, flag_ode)
%         if B<2, output = [output; time, X]; end % >690
        X0=X(size(X,1),:);
        %     B; %output beat number to the screen to monitor runtime progress
    end
    
end
