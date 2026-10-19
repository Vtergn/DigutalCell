function refine_land_batch(input_mat,output_dir,first_row,last_row,gamma)
% Independent-cell batch wrapper for the frozen Land refinement protocol.
if ~exist(output_dir,'dir'), mkdir(output_dir); end
for row=first_row:last_row
    output_mat=fullfile(output_dir,sprintf('land_refine_test_%02d.mat',row));
    refine_land_one_cell(input_mat,output_mat,row,gamma);
end
end
