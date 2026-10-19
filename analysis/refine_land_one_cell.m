function refine_land_one_cell(input_mat,output_mat,row,gamma)
% Test-time true-Land calibration, initialized by training-cell ridge.
% Only T(1.0), T(1.1), and measured Ca enter optimization.
if nargin<4, gamma=.1; end
s=load(input_mat);
ca=double(s.Cai_uM(row,:));
cal=double(squeeze(s.Tcal_kPa(row,:,:)));
if isfield(s,'Ttarget_kPa')
    target=double(s.Ttarget_kPa(row,:));
else
    target=[];  % blind confirmation: target is absent from MATLAB input
end
lo=double(s.lo(5:12)); hi=double(s.hi(5:12));
x0=double((s.theta_start(row,5:12)-lo)./(hi-lo));
fit_index=[1 2 4 5 6];  % Tref, ca50, nperm, beta0, beta1
sample=1:10:601;
observed=cal(:,sample);
denominator=50*sqrt(numel(observed));
fun=@(z) residual(z,x0,fit_index,lo,hi,ca,observed,sample,denominator,gamma);
options=optimoptions('lsqnonlin','Display','off','MaxIterations',5, ...
    'MaxFunctionEvaluations',45,'FiniteDifferenceType','forward', ...
    'FiniteDifferenceStepSize',.01,'FunctionTolerance',1e-5,'StepTolerance',1e-4);
[zfit,resnorm,~,exitflag,output]=lsqnonlin(fun,x0(fit_index), ...
    zeros(size(fit_index)),ones(size(fit_index)),options);
xfit=x0; xfit(fit_index)=zfit;
theta_before=lo+x0.*(hi-lo);
theta_after=lo+xfit.*(hi-lo);
before=land_tension_given_ca(ca,theta_before,[1 1.1 1.125]);
after=land_tension_given_ca(ca,theta_after,[1 1.1 1.125]);
if isempty(target)
    true_target=nan(size(sample));
else
    true_target=target(sample);
end
result=struct('cell_index_zero_based',s.cell_index_zero_based(row), ...
    'split_code',s.split_code(row),'gamma',gamma,'fit_index',fit_index, ...
    'theta_before',theta_before,'theta_after',theta_after, ...
    'before_target_MAE_kPa',mean(abs(before.Ta_kPa(3,sample)-true_target)), ...
    'after_target_MAE_kPa',mean(abs(after.Ta_kPa(3,sample)-true_target)), ...
    'before_cal_MAE_kPa',mean(abs(before.Ta_kPa(1:2,sample)-observed),'all'), ...
    'after_cal_MAE_kPa',mean(abs(after.Ta_kPa(1:2,sample)-observed),'all'), ...
    'before_target_waveform',before.Ta_kPa(3,:), ...
    'after_target_waveform',after.Ta_kPa(3,:), ...
    'exitflag',exitflag,'function_evaluations',output.funcCount,'resnorm',resnorm);
save(output_mat,'result','-v7.3');
fprintf('cell %.0f gamma %.3f cal %.3f -> %.3f target %.3f -> %.3f, %d calls\n', ...
    result.cell_index_zero_based,gamma,result.before_cal_MAE_kPa, ...
    result.after_cal_MAE_kPa,result.before_target_MAE_kPa, ...
    result.after_target_MAE_kPa,result.function_evaluations);
end

function r=residual(z,x0,fit_index,lo,hi,ca,observed,sample,denominator,gamma)
x=x0; x(fit_index)=z;
try
    prediction=land_tension_given_ca(ca,lo+x.*(hi-lo),[1 1.1]);
    difference=(prediction.Ta_kPa(:,sample)-observed)/denominator;
    r=[difference(:);sqrt(gamma)*(z(:)-x0(fit_index)')];
catch
    r=ones(numel(observed)+numel(fit_index),1)*100;
end
end
