function land_local_posterior_pilot(input_mat, refined_dir, output_mat, n_cells, n_draws)
% Exploratory conditional five-parameter Laplace approximation.
% AP/electrical and three unrefined Land parameters remain fixed; no
% held-out mechanical target is loaded by this function.
s=load(input_mat);
target_lambdas=[1.125 1.15];
fit_index=[1 2 4 5 6];
sample=1:10:601;
draws=nan(n_cells,numel(target_lambdas),301,n_draws);
point=nan(n_cells,numel(target_lambdas),301);
covariance=nan(n_cells,5,5);
singular_values=nan(n_cells,5);
sigma_effective=nan(n_cells,1);
clipped_fraction=nan(n_cells,1);
failed_draws=zeros(n_cells,1);
cell_index_zero_based=double(s.cell_index_zero_based(1:n_cells));
prior_sd=.20; % prior-scaled parameter uncertainty, fixed before target scoring
rng(20260929);
for row=1:n_cells
    r=load(fullfile(refined_dir,sprintf('land_refine_test_%02d.mat',row)));
    theta=double(r.result.theta_after(:)');
    ca=double(s.Cai_uM(row,:));
    observed=double(squeeze(s.Tcal_kPa(row,:,:)));
    lo=double(s.lo(5:12)); hi=double(s.hi(5:12));
    z=(theta-lo)./(hi-lo);
    base=land_tension_given_ca(ca,theta,[1 1.1]);
    resid=base.Ta_kPa(:,sample)-observed(:,sample);
    % Noise floor is an explicit assumption. Additional residual RMS is
    % treated as model discrepancy; this is not externally calibrated.
    sigma_effective(row)=sqrt(.5^2+mean(resid(:).^2));
    J=zeros(numel(resid),5);
    for k=1:5
        j=fit_index(k);
        zp=z; zm=z;
        zp(j)=min(1,z(j)+.01);
        zm(j)=max(0,z(j)-.01);
        tp=land_tension_given_ca(ca,lo+zp.*(hi-lo),[1 1.1]);
        tm=land_tension_given_ca(ca,lo+zm.*(hi-lo),[1 1.1]);
        delta=(tp.Ta_kPa(:,sample)-tm.Ta_kPa(:,sample))/(zp(j)-zm(j));
        J(:,k)=delta(:);
    end
    [~,D,~]=svd(J/sigma_effective(row),'econ');
    singular_values(row,:)=diag(D)';
    C=inv((J'*J)/sigma_effective(row)^2+eye(5)/prior_sd^2);
    covariance(row,:,:)=C;
    R=chol(C+eye(5)*1e-12,'lower');
    original=land_tension_given_ca(ca,theta,target_lambdas);
    point(row,:,:)=original.Ta_kPa(:,1:2:601);
    countclip=0;
    for q=1:n_draws
        zz=z;
        raw=z(fit_index)'+R*randn(5,1);
        clipped=min(1,max(0,raw));
        countclip=countclip+any(raw~=clipped);
        zz(fit_index)=clipped;
        try
            forward=land_tension_given_ca(ca,lo+zz.*(hi-lo),target_lambdas);
            draws(row,:,:,q)=forward.Ta_kPa(:,1:2:601);
        catch
            failed_draws(row)=failed_draws(row)+1;
        end
    end
    clipped_fraction(row)=countclip/n_draws;
    fprintf('posterior row %d sigma %.3f clip %.2f failures %d\n', ...
        row,sigma_effective(row),clipped_fraction(row),failed_draws(row));
end
save(output_mat,'draws','point','covariance','singular_values', ...
    'sigma_effective','clipped_fraction','failed_draws', ...
    'target_lambdas','cell_index_zero_based','prior_sd','-v7.3');
end
