function generate_land_load_grid_truth(input_mat, output_mat, first_row, last_row)
% Separate truth run; no predictions or fitted parameters are loaded here.
s=load(input_mat);
lam=double(s.target_lambdas(:)');
n=last_row-first_row+1;
truth=nan(n,numel(lam),601);
failed=false(n,1);
for row=first_row:last_row
    try
        q=land_tension_given_ca(double(s.Cai_uM(row,:)), ...
            double(s.theta_true(row,:)),lam);
        truth(row-first_row+1,:,:)=q.Ta_kPa(:,1:601);
    catch err
        failed(row-first_row+1)=true;
        fprintf('truth failed row %d: %s\n',row,err.message);
    end
    if mod(row,8)==0, fprintf('truth through row %d\n',row); end
end
cell_index_zero_based=s.cell_index_zero_based(first_row:last_row);
save(output_mat,'truth','failed','cell_index_zero_based','lam','-v7.3');
end
