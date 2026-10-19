function land_eight_direction_audit(input_mat,refined_dir,output_mat,n_cells,include_target,fast_jac)
% Eight mechanical-parameter Jacobians; no hidden-load outcome is loaded.
if nargin<5,include_target=true;end
if nargin<6,fast_jac=false;end
s=load(input_mat);
sample=1:10:601;
if include_target,target_lambdas=[1.125 1.15];else,target_lambdas=[];end
all_lambdas=[1 1.1 target_lambdas];
J_observed=nan(n_cells,2*numel(sample),8);
if include_target,J_target=nan(n_cells,2*numel(sample),8);else,J_target=[];end
theta_center=nan(n_cells,8);
observed_residual=nan(n_cells,2*numel(sample));
sigma_effective=nan(n_cells,1);
cell_index_zero_based=double(s.cell_index_zero_based(1:n_cells));
lo=double(s.lo(5:12));hi=double(s.hi(5:12));
for row=1:n_cells
    r=load(fullfile(refined_dir,sprintf('land_refine_test_%02d.mat',row)));
    theta=double(r.result.theta_after(:)');
    ca=double(s.Cai_uM(row,:));
    obs=double(squeeze(s.Tcal_kPa(row,:,:)));
    z=(theta-lo)./(hi-lo);
    base=land_tension_given_ca(ca,theta,all_lambdas);
    residual=base.Ta_kPa(1:2,sample)-obs(:,sample);
    observed_residual(row,:)=residual(:)';
    sigma_effective(row)=sqrt(.5^2+mean(residual(:).^2));
    theta_center(row,:)=theta;
    for k=1:8
        zp=z;
        if fast_jac
            if z(k)>.99,zp(k)=z(k)-.01;else,zp(k)=z(k)+.01;end
            tp=land_tension_given_ca(ca,lo+zp.*(hi-lo),all_lambdas);
            d=(tp.Ta_kPa(:,sample)-base.Ta_kPa(:,sample))/(zp(k)-z(k));
        else
            zm=z;
            zp(k)=min(1,z(k)+.01);
            zm(k)=max(0,z(k)-.01);
            tp=land_tension_given_ca(ca,lo+zp.*(hi-lo),all_lambdas);
            tm=land_tension_given_ca(ca,lo+zm.*(hi-lo),all_lambdas);
            d=(tp.Ta_kPa(:,sample)-tm.Ta_kPa(:,sample))/(zp(k)-zm(k));
        end
        J_observed(row,:,k)=reshape(d(1:2,:),1,[]);
        if include_target,J_target(row,:,k)=reshape(d(3:4,:),1,[]);end
    end
    fprintf('eight-direction Jacobian row %d sigma %.3f\n',row,sigma_effective(row));
end
save(output_mat,'J_observed','J_target','theta_center','observed_residual', ...
    'sigma_effective','cell_index_zero_based','target_lambdas','lo','hi','-v7.3');
end
