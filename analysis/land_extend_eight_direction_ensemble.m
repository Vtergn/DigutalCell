function land_extend_eight_direction_ensemble(input_mat,audit_mat,old_mat,out_mat,n_extra)
% Append independently seeded candidates without repeating existing Land runs.
% Target outcomes/true parameters are absent from all inputs.
s=load(input_mat);
a=load(audit_mat);
old=load(old_mat);
n_cells=numel(a.cell_index_zero_based);
n_old=size(old.prediction,4);
n_new=n_old+n_extra;
sample=1:10:601;
all_lambdas=[1 1.1 1.125 1.15];
prediction=nan(n_cells,2,301,n_new);
prediction(:,:,:,1:n_old)=old.prediction;
theta_draw=nan(n_cells,8,n_new);
theta_draw(:,:,1:n_old)=old.theta_draw;
cal_rmse=nan(n_cells,n_new);
cal_rmse(:,1:n_old)=old.cal_rmse;
weights=nan(n_cells,n_new);
point=old.point;
candidate_failure=old.candidate_failure;
reflected_fraction=nan(n_cells,1);
prior_sd=double(old.prior_sd);
effective_sample_count=double(old.effective_sample_count);
target_lambdas=old.target_lambdas;
cell_index_zero_based=old.cell_index_zero_based;
for row=1:n_cells
    rng(20261002+row);
    theta=double(a.theta_center(row,:));
    lo=double(a.lo(:)');hi=double(a.hi(:)');
    z=(theta-lo)./(hi-lo);
    ca=double(s.Cai_uM(row,:));
    observed=double(squeeze(s.Tcal_kPa(row,:,:)));
    J=double(squeeze(a.J_observed(row,:,:)));
    sigma=double(a.sigma_effective(row));
    [~,S,V]=svd(J/sigma,'econ');
    singular=diag(S);
    sd=1./sqrt(singular.^2+1/prior_sd^2);
    count_reflect=old.reflected_fraction(row)*n_old;
    for q=n_old+1:n_new
        raw=z+(V*(sd.*randn(8,1)))';
        zz=mod(raw,2);zz=min(zz,2-zz);
        count_reflect=count_reflect+any(abs(zz-raw)>1e-12);
        th=lo+zz.*(hi-lo);
        try
            forward=land_tension_given_ca(ca,th,all_lambdas);
            residual=forward.Ta_kPa(1:2,sample)-observed(:,sample);
            cal_rmse(row,q)=sqrt(mean(residual(:).^2));
            theta_draw(row,:,q)=th;
            prediction(row,:,:,q)=forward.Ta_kPa(3:4,1:2:601);
        catch
            candidate_failure(row)=candidate_failure(row)+1;
        end
    end
    good=isfinite(cal_rmse(row,:));
    logw=-.5*effective_sample_count*(cal_rmse(row,good)/sigma).^2;
    logw=logw-max(logw);
    w=exp(logw);w=w/sum(w);
    weights(row,good)=w;weights(row,~good)=0;
    reflected_fraction(row)=count_reflect/n_new;
    fprintf('extended ensemble row %d/%d, valid %d, ESS %.1f\n', ...
        row,n_cells,nnz(good),1/sum(w.^2));
end
save(out_mat,'prediction','theta_draw','cal_rmse','weights','point', ...
    'candidate_failure','reflected_fraction','prior_sd', ...
    'effective_sample_count','target_lambdas','cell_index_zero_based','-v7.3');
end
