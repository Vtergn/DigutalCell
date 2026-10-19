function dy = human_bidirectional_rhs(t,y,lam,trpn_on,sac_on,par,isac_factor,beta,feedback)
% 43 ToR-ORd/SAC states + 7 Land states with optional dynamic TnC feedback.
% feedback.g_ionic scales the entire stretch-sensitive ionic increment.
% It is an effective feedback strength, not an isolated SAC conductance.
if nargin < 9 || isempty(feedback), feedback = struct('g_ionic',1); end
if ~isfield(feedback,'ionic_scale'), feedback.ionic_scale=[1,1,1]; end
if ~isfield(feedback,'land_params'), feedback.land_params=[]; end
if ~isfield(feedback,'gain_exponent'), feedback.gain_exponent=0; end
if ~isfield(feedback,'sac_saturation'), feedback.sac_saturation=0; end
if ~isfield(feedback,'trpn_capacity_scale'), feedback.trpn_capacity_scale=1; end
if ~isfield(feedback,'sac_gate'), feedback.sac_gate=1; end
if ~isfield(feedback,'sac_residual_slope'), feedback.sac_residual_slope=0; end
if isa(lam,'function_handle')
    mechanical = lam(t);
    current_lambda = mechanical(1);
    dlambda_dt = mechanical(2);
else
    current_lambda = lam;
    dlambda_dt = 0;
end
if sac_on, ionic_lambda = current_lambda; else, ionic_lambda = 1; end
if sac_on && feedback.sac_saturation>0
    % Alternative saturating length-to-ionic-input law anchored at lambda=1.10.
    % It changes the model form while preserving the calibration stimulus.
    d=current_lambda-1;
    ionic_lambda=1+d*(1+0.1/feedback.sac_saturation)/ ...
        (1+abs(d)/feedback.sac_saturation);
end
if sac_on && (feedback.g_ionic ~= 1 || feedback.gain_exponent ~= 0 || ...
        feedback.sac_gate ~= 1 || feedback.sac_residual_slope ~= 0) && ionic_lambda ~= 1
    de0 = electrical_rhs(t,y,1,par,isac_factor,feedback.ionic_scale);
    de1 = electrical_rhs(t,y,ionic_lambda,par,isac_factor,feedback.ionic_scale);
    gain=feedback.g_ionic*feedback.sac_gate* ...
        (abs(ionic_lambda-1)/0.1)^feedback.gain_exponent* ...
        exp(feedback.sac_residual_slope*(0.1-abs(ionic_lambda-1)));
    gain=min(3,max(0,gain));
    de = de0 + gain*(de1-de0);
else
    de = electrical_rhs(t,y,ionic_lambda,par,isac_factor,feedback.ionic_scale);
end
if isempty(feedback.land_params)
    [dl,~,~,~] = Land2016(t,y(44:50),1000*y(6),'intact', ...
        current_lambda,dlambda_dt,beta);
else
    [dl,~,~,~] = Land2016_variable(t,y(44:50),1000*y(6),'intact', ...
        current_lambda,dlambda_dt,feedback.land_params);
end
de = de(:); dl = dl(:);
if trpn_on
    % Undo the SAC model's instantaneous TRPN buffer, then use Land's
    % dynamic TRPN flux. cai is mM, Land Cai is uM, TRPN is a fraction.
    cai = y(6); cmdnmax=.05; kmcmdn=.00238;
    trpnmax=.07; kmtrpn=.0005;
    other_buffer = cmdnmax*kmcmdn/(kmcmdn+cai)^2;
    static_trpn_buffer = trpnmax*kmtrpn/(kmtrpn+cai)^2;
    B_old = 1/(1+other_buffer+static_trpn_buffer);
    B_new = 1/(1+other_buffer);
    de(6) = B_new*(de(6)/B_old - ...
        feedback.trpn_capacity_scale*trpnmax*dl(3));
end
dy = [de;dl];
end

function de=electrical_rhs(t,y,lambda,par,isac_factor,ionic_scale)
if all(ionic_scale==1)
    de=model_SAC(t,y(1:43),1,lambda,par,isac_factor,-1,-1,0);
else
    de=model_SAC_hetero(t,y(1:43),1,lambda,par,isac_factor,-1,-1,0,ionic_scale);
end
end
