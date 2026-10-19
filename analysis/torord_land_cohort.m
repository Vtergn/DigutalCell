function torord_land_cohort(nCells, output_dir, nBeats, checkpointEvery, seed)
% Paired AP/Ca/active-tension virtual cardiomyocytes at seven isometric loads.
% All conditions of a virtual cell share exactly the same twelve parameters.
% First use: torord_land_cohort(6)
if nargin < 1, nCells = 6; end
if nargin < 2 || isempty(output_dir)
    output_dir = fullfile(fileparts(mfilename('fullpath')),'virtual_cardiomyocyte_pilot');
end
if nargin < 3, nBeats = 12; end
if nargin < 4, checkpointEvery = 100; end
if nargin < 5, seed = 20260928; end
if nCells < 2 || nCells ~= floor(nCells), error('nCells must be integer >= 2'); end
if ~exist(output_dir,'dir'), mkdir(output_dir); end
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'Data','torord_official','matlab'));
addpath(fileparts(mfilename('fullpath')));
rng(seed,'twister');

% Ranges are deliberately moderate around the original model defaults.
% They define this synthetic benchmark prior, not a human-population prior.
names = {'ICaL_Multiplier','Jup_Multiplier','INaCa_Multiplier', ...
    'IKr_Multiplier','Tref','ca50','TRPN_n','nperm', ...
    'beta_0','beta_1','phi','perm50'};
lo = [0.75 0.75 0.75 0.75, 90 0.65 1.6 1.3, 1.6 -3.1 1.6 0.27];
hi = [1.25 1.25 1.25 1.25,150 0.96 2.4 3.0, 3.0 -1.7 2.9 0.44];
theta = zeros(nCells,numel(names));
for p=1:numel(names)
    u = ((randperm(nCells)'-1)+rand(nCells,1))/nCells;
    theta(:,p) = lo(p)+(hi(p)-lo(p))*u;
end
theta_table = array2table(theta,'VariableNames',names);
theta_table.cell_id = (1:nCells)';
split_code = zeros(nCells,1,'uint8'); % 0=train, 1=validation, 2=test
ordering = randperm(nCells);
nTrain = floor(0.70*nCells);
nVal = floor(0.15*nCells);
split_code(ordering(nTrain+1:nTrain+nVal)) = 1;
split_code(ordering(nTrain+nVal+1:end)) = 2;
theta_table.split_code = split_code;
theta_table = movevars(theta_table,'cell_id','Before',1);
writetable(theta_table,fullfile(output_dir,'theta.csv'));

lambda = [0.975 1.000 1.025 1.050 1.075 1.100 1.125];
time_ms = (0:1000)';
V = nan(nCells,numel(time_ms),'single');
Cai_uM = V;
Ta_kPa = nan(nCells,numel(lambda),numel(time_ms),'single');
Tp_kPa = Ta_kPa;
success = false(nCells,1);
calcium_cycle_drift = nan(nCells,1);
electrical_beats_used = nan(nCells,1);
land_cycle_drift = nan(nCells,numel(lambda));
mechanical_cycles_used = nan(nCells,numel(lambda));
failure_reason = strings(nCells,1);
checkpoint_path = fullfile(output_dir,'cohort_checkpoint.mat');
first_cell = 1;
if exist(checkpoint_path,'file')
    old = load(checkpoint_path);
    if ~isequal(size(old.theta),size(theta)) || max(abs(old.theta(:)-theta(:)))>1e-12
        error('Checkpoint parameter design differs from requested cohort');
    end
    V=old.V; Cai_uM=old.Cai_uM; Ta_kPa=old.Ta_kPa; Tp_kPa=old.Tp_kPa;
    success=old.success; calcium_cycle_drift=old.calcium_cycle_drift;
    electrical_beats_used=old.electrical_beats_used;
    land_cycle_drift=old.land_cycle_drift;
    mechanical_cycles_used=old.mechanical_cycles_used;
    failure_reason=old.failure_reason;
    first_cell=old.next_cell;
    fprintf('Resuming cohort from cell %d\n',first_cell);
end
tstart = tic;
for i=first_cell:nCells
  try
    p = struct('bcl',1000,'model',@model_Torord,'verbose',false, ...
        'cellType',0,'ICaL_Multiplier',theta(i,1), ...
        'Jup_Multiplier',theta(i,2), ...
        'INaCa_Multiplier',theta(i,3),'IKr_Multiplier',theta(i,4));
    x0 = getStartingState('Torord_endo');
    beats_used = 0;
    while true
      block_beats = nBeats;
      if beats_used > 0, block_beats = 8; end
      [bt,bx] = modelRunner(x0,[],p,block_beats,block_beats-2);
      beats_used = beats_used+block_beats;
      c_prev = 1000*interp1(bt{1},bx{1}(:,6),time_ms,'pchip');
      c = 1000*interp1(bt{2},bx{2}(:,6),time_ms,'pchip');
      v = interp1(bt{2},bx{2}(:,1),time_ms,'pchip');
      calcium_cycle_drift(i) = max(abs(c-c_prev))/max(c);
      if calcium_cycle_drift(i)<=0.01 || beats_used>=60, break; end
      x0 = bx{2}(end,:);
    end
    electrical_beats_used(i) = beats_used;
    if any(~isfinite(c)) || any(c<=0) || any(~isfinite(v)) ...
            || calcium_cycle_drift(i)>0.01 || max(v)<0
        error('Electrical quality/convergence check failed');
    end
    mt = struct('Tref',theta(i,5),'ca50',theta(i,6), ...
        'TRPN_n',theta(i,7),'nperm',theta(i,8), ...
        'beta_0',theta(i,9),'beta_1',theta(i,10), ...
        'phi',theta(i,11),'perm50',theta(i,12));
    V(i,:) = single(v);
    Cai_uM(i,:) = single(c);
    for j=1:numel(lambda)
      lam = lambda(j);
      ca_at = @(s) interp1(time_ms,c,min(1000,mod(s,1000)),'pchip');
      rhs = @(s,y) Land2016_variable(s,y,ca_at(s),'intact',lam,0,mt);
      for mechanical_cycles = [5,7]
        [lt,ly] = ode15s(rhs,[0 1000*mechanical_cycles],Land2016_variable());
        ty = interp1(lt,ly,1000*(mechanical_cycles-1)+time_ms,'pchip');
        ty_prev = interp1(lt,ly,1000*(mechanical_cycles-2)+time_ms,'pchip');
        aa = zeros(size(time_ms)); pp = aa; aa_prev = aa;
        for k=1:numel(time_ms)
          [~,~,aa(k),pp(k)] = Land2016_variable(1000*(mechanical_cycles-1)+time_ms(k),ty(k,:),c(k),'intact',lam,0,mt);
          [~,~,aa_prev(k),~] = Land2016_variable(1000*(mechanical_cycles-2)+time_ms(k),ty_prev(k,:),c(k),'intact',lam,0,mt);
        end
        land_cycle_drift(i,j) = max(abs(aa-aa_prev))/max(aa);
        if land_cycle_drift(i,j)<=0.01, break; end
      end
      mechanical_cycles_used(i,j) = mechanical_cycles;
      if any(~isfinite(aa)) || max(aa)<=0 || land_cycle_drift(i,j)>0.01
        error('Land quality/convergence check failed at lambda %.3f',lam);
      end
      Ta_kPa(i,j,:) = single(aa);
      Tp_kPa(i,j,:) = single(pp);
    end
    success(i) = true;
    fprintf('cell %d/%d OK; %d electrical beats; Ca drift %.4g; max Land drift %.4g; elapsed %.1f s\n', ...
        i,nCells,beats_used,calcium_cycle_drift(i),max(land_cycle_drift(i,:)),toc(tstart));
  catch ME
    warning('Cell %d failed: %s',i,ME.message);
    failure_reason(i) = string(ME.message);
    V(i,:) = nan; Cai_uM(i,:) = nan;
    Ta_kPa(i,:,:) = nan; Tp_kPa(i,:,:) = nan;
  end
  if mod(i,checkpointEvery)==0
    next_cell=i+1;
    save(checkpoint_path,'V','Cai_uM','Ta_kPa','Tp_kPa', ...
      'theta','names','lambda','time_ms','success','split_code', ...
      'calcium_cycle_drift','electrical_beats_used', ...
      'land_cycle_drift','mechanical_cycles_used','failure_reason', ...
      'lo','hi','next_cell','-v7.3');
  end
end
save(fullfile(output_dir,'cohort.mat'),'V','Cai_uM','Ta_kPa','Tp_kPa', ...
    'theta','names','lambda','time_ms','success','split_code', ...
    'calcium_cycle_drift','electrical_beats_used', ...
    'land_cycle_drift','mechanical_cycles_used','failure_reason', ...
    'lo','hi','-v7.3');
if exist(checkpoint_path,'file'), delete(checkpoint_path); end
feature = table((1:nCells)',success,calcium_cycle_drift, ...
    'VariableNames',{'cell_id','success','calcium_cycle_drift'});
feature.split_code = split_code;
feature.failure_reason = failure_reason;
feature.electrical_beats_used = electrical_beats_used;
feature.Vpeak_mV = max(double(V),[],2);
feature.Cai_peak_uM = max(double(Cai_uM),[],2);
for j=1:numel(lambda)
    tag = strrep(sprintf('%.3f',lambda(j)),'.','p');
    feature.(['Ta_peak_' tag '_kPa']) = max(double(Ta_kPa(:,j,:)),[],3);
end
writetable(feature,fullfile(output_dir,'features.csv'));
manifest = struct('seed',seed,'nCells',nCells,'nBeats',nBeats, ...
    'checkpoint_every',checkpointEvery, ...
    'adaptive_electrical_prepacing',true,'max_electrical_beats',60, ...
    'mechanical_cycles',5,'adaptive_mechanical_cycles',7, ...
    'electrical_cycle_drift_limit',0.01, ...
    'mechanical_cycle_drift_limit',0.01,'lambda',lambda, ...
    'theta_names',{names},'lower_bounds',lo,'upper_bounds',hi, ...
    'split_code_meanings',{{'train','validation','test'}}, ...
    'coupling','one-way Ca to active tension', ...
    'calibration_loads',[1.0,1.1], ...
    'in_range_test_loads',[1.025,1.05,1.075], ...
    'outside_range_test_loads',[0.975,1.125]);
fid = fopen(fullfile(output_dir,'manifest.json'),'w');
fprintf(fid,'%s',jsonencode(manifest,PrettyPrint=true)); fclose(fid);
fprintf('%d/%d virtual cells passed; total %.1f s\n',sum(success),nCells,toc(tstart));
end
