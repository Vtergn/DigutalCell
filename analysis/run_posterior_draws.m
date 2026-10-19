function run_posterior_draws(input_csv, output_mat, first_row, last_row)
% Re-run posterior parameter draws through the actual ToR-ORd and Land ODEs.
% Row limits permit independent workers without sharing mutable output files.
if nargin < 3 || isempty(first_row), first_row = 1; end
if nargin < 4 || isempty(last_row), last_row = inf; end
tab = readtable(input_csv);
last_row = min(last_row,height(tab));
if first_row > last_row, error('Empty requested row range'); end
rows = first_row:last_row;
n = numel(rows);
Ta_kPa = nan(n,1001,'single');
peak_kPa = nan(n,1);
ok = false(n,1);
error_message = strings(n,1);
cell_index_zero_based = tab.cell_index_zero_based(rows);
mode = tab.mode(rows);
draw_index = tab.draw_index(rows);
for local = 1:n
    i = rows(local);
    theta = table2array(tab(i,4:15));
    try
        result = torord_land_given_theta(theta,1.125,12);
        Ta_kPa(local,:) = single(result.Ta_kPa(1,:));
        peak_kPa(local) = max(result.Ta_kPa(1,:));
        ok(local) = true;
    catch ME
        error_message(local) = string(ME.message);
    end
    if mod(local,20)==0 || local==n
        fprintf('forward rows %d-%d: %d/%d complete, %d succeeded\n', ...
            first_row,last_row,local,n,sum(ok(1:local)));
    end
end
save(output_mat,'Ta_kPa','peak_kPa','ok','error_message', ...
    'cell_index_zero_based','mode','draw_index','rows','-v7.3');
end
