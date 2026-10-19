function land_eight_direction_ensemble(input_mat,audit_mat,output_mat,n_draws)
% Exploratory full-Land compatible twins. No withheld target outcome is read.
% The local Jacobian proposes candidates; every candidate is checked with
% the original nonlinear Land solver at both observed mechanical loads.
s=load(input_mat);
a=load(audit_mat);
n_cells=numel(a.cell_index_zero_based);
sample=1:10:601;
target_lambdas=[1.125 1.15];
all_lambdas=[1 1.1 target_lambdas];
prediction=nan(n_cells,2,301,n_draws);
theta_draw=nan(n_cells,8,n_draws);
cal_rmse=nan(n_cells,n_draws);
weights=nan(n_cells,n_draws);
candidate_failure=zeros(n_cells,1);
reflected_fraction=zeros(n_cells,1);
point=nan(n_cells,2,301);
prior_sd=1/sqrt(12);
effective_sample_count=6; % explicit working likelihood, not an exact posterior
rng(20260929);
for row=1:n_cells
    theta=double(a.theta_center(row,:));
    lo=double(a.lo(:)'); hi=double(a.hi(:)');
    z=(theta-lo)./(hi-lo);
    ca=double(s.Cai_uM(row,:));
    observed=double(squeeze(s.Tcal_kPa(row,:,:)));
    J=double(squeeze(a.J_observed(row,:,:)));
    sigma=double(a.sigma_effective(row));
    [~,S,V]=svd(J/sigma,'econ');
    singular=diag(S);
    sd=1./sqrt(singular.^2+1/prior_sd^2);
    raw=z;
    for q=1:n_draws
        if q>1
            raw=z+(V*(sd.*randn(8,1)))';
        end
        % Repeated reflection retains the candidate within the prior box
        % without accumulating mass exactly on the bounds.
        zz=mod(raw,2);
        zz=min(zz,2-zz);
        reflected_fraction(row)=reflected_fraction(row)+any(abs(zz-raw)>1e-12);
        th=lo+zz.*(hi-lo);
        try
            forward=land_tension_given_ca(ca,th,all_lambdas);
            residual=forward.Ta_kPa(1:2,sample)-observed(:,sample);
            cal_rmse(row,q)=sqrt(mean(residual(:).^2));
            theta_draw(row,:,q)=th;
            prediction(row,:,:,q)=forward.Ta_kPa(3:4,1:2:601);
            if q==1,point(row,:,:)=forward.Ta_kPa(3:4,1:2:601);end
        catch
            candidate_failure(row)=candidate_failure(row)+1;
        end
    end
    good=isfinite(cal_rmse(row,:));
    if ~any(good),error('No valid candidate for cell %d',row);end
    % A transparent soft compatibility weight based on observed-load fit.
    % This is deliberately called an ensemble, not a Bayesian posterior.
    logw=-0.5*effective_sample_count*(cal_rmse(row,good)/sigma).^2;
    logw=logw-max(logw);
    w=exp(logw); w=w/sum(w);
    weights(row,good)=w; weights(row,~good)=0;
    reflected_fraction(row)=reflected_fraction(row)/n_draws;
    fprintf('eight-parameter ensemble row %d/%d, valid %d, ESS %.1f\n', ...
        row,n_cells,nnz(good),1/sum(w.^2));
end
cell_index_zero_based=a.cell_index_zero_based;
save(output_mat,'prediction','theta_draw','cal_rmse','weights', ...
    'candidate_failure','reflected_fraction','point','prior_sd', ...
    'effective_sample_count','target_lambdas','cell_index_zero_based','-v7.3');
end
