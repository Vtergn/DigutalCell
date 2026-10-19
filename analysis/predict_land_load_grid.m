function predict_land_load_grid(input_mat, output_mat, first_row, last_row)
% Forward-only evaluation. Input has no true parameters or target traces.
s=load(input_mat);
lam=double(s.target_lambdas(:)');
n=last_row-first_row+1;
prediction=nan(n,numel(lam),601);
failed=false(n,1);
for row=first_row:last_row
    try
        q=land_tension_given_ca(double(s.Cai_uM(row,:)), ...
            double(s.theta_after(row,:)),lam);
        prediction(row-first_row+1,:,:)=q.Ta_kPa(:,1:601);
    catch err
        failed(row-first_row+1)=true;
        fprintf('prediction failed row %d: %s\n',row,err.message);
    end
    if mod(row,8)==0, fprintf('predicted through row %d\n',row); end
end
cell_index_zero_based=s.cell_index_zero_based(first_row:last_row);
save(output_mat,'prediction','failed','cell_index_zero_based','lam','-v7.3');
end
