function run_local_jacobian(input_csv, output_mat, row)
% Central finite differences of the true ToR-ORd/Land simulator.
% Derivatives are with respect to each unit-prior-width parameter.
tab=readtable(input_csv);
theta=table2array(tab(row,2:13));
lo=table2array(tab(row,14:25));
hi=table2array(tab(row,26:37));
if any(theta-lo < .01*(hi-lo)) || any(hi-theta < .01*(hi-lo))
    error('Parameter too close to a prior boundary');
end
cell_index_zero_based=tab.cell_index_zero_based(row);
stretches=[1.0 1.1 1.125];
time_index=1:10:601;  % 0-600 ms, ten-millisecond cadence
Jv=zeros(numel(time_index),12);
Jca=Jv;
Jt=zeros(3,numel(time_index),12);
delta_unit=.01;
for j=1:12
    h=delta_unit*(hi(j)-lo(j));
    plus=theta; plus(j)=plus(j)+h;
    minus=theta; minus(j)=minus(j)-h;
    yp=torord_land_given_theta(plus,stretches,12);
    ym=torord_land_given_theta(minus,stretches,12);
    Jv(:,j)=(yp.Vm_mV(time_index)-ym.Vm_mV(time_index))/(2*delta_unit);
    Jca(:,j)=(yp.Cai_uM(time_index)-ym.Cai_uM(time_index))/(2*delta_unit);
    Jt(:,:,j)=(yp.Ta_kPa(:,time_index)-ym.Ta_kPa(:,time_index))/(2*delta_unit);
    fprintf('cell %d derivative %d/12 complete\n',cell_index_zero_based,j);
end
save(output_mat,'Jv','Jca','Jt','theta','lo','hi','stretches', ...
    'time_index','cell_index_zero_based','delta_unit','-v7.3');
end
