function [ti_s,fisave,fi2save]=Ode2Alt_EM(DfDtFu,DfDtFu2,ti,fp,fp2,skip,timescale, CL, BAR, startsave)

%Adaptation 180913: 
% INPUTS:
% "DfDtFu": handle to ORD EP model
% "DfDtFu2": handle to MechChem model (SVarDot_cai)
% ti: row vector of time points
% fp: row vector of state variables defines initial condition at time ti(1)
% for the ORD model
% fp2: row vector of state variables defines initial condition at time
% ti(1)for the MechChem model
% "skip" reflects time interval (in ms) between consecutive calls to the DfDtFu2 (MechChem model)
% timescale: conversion of timescale between ORD and MechChem (1e-3)
% CL: cycle length 
% BAR: boolean of whether beta adrenergic stimulation is on or not
% startsave: nb of beats that are actually recorded (to allow for
% preprocessing: for example CL*(nb_beat - 5) to save 5 beats)
% OUTPUTS: ti_s=column vector of time points
% fisave: matrix, each rows of statevariables represents successive time
% points for ORD
% fi2save: matrix, each rows of statevariables represents successive time
% points for MechChem


% function [ti,fi]=Ode2(DfDtFu,ti,fp)
% Solving system of ordinary differential equation with given steps, 2nd order
% 1 calculation of derivative per time step
% DfDtFu= string pointing to function df/dt
% ti= row vector of time points
% fp= row vector of state variables defines initial condition at time ti(1)
% OUTPUT: t=column vector of time points
% fi: matrix, each rows of statevariables represents successive time points
% dfdt=(DfDtFu); % setting derivative function
disp('Initializing');

nt=numel(ti); % number of time points
np=numel(fp); % number of state variables
np2=numel(fp2);
numtimepointssave = length(startsave:skip:ti(end));
ti_s = zeros(numtimepointssave+2,1);
fisave=zeros(numtimepointssave+2,np); % reserve memory space for output
fi2save=zeros(numtimepointssave+2,np2);

fi=zeros(2,np); % reserve memory space for output
fi2=zeros(2,np2);


lastCallToFunc2 = ti(2);
dt     = ti(2)-ti(1); % time step

if ~isempty(DfDtFu)
  dfdt=str2func(DfDtFu); % setting derivative function
  h      = fp; % h=output, set to initial condition
  fi(1,:)= h ; % copy to output matrix
  fisave(1,:)= h ;
  df1    = dfdt(ti(1),h',fp2(1:50),BAR,1)'; % fp2(1:80) ?
  f      = h + df1*dt;
  df     = dfdt(ti(1),f',fp2(1:50),BAR,2)';
  g      = 0.5*(df+df1);
  h      = h + g*dt; % value of state variable, 2nd order approx
  fi(2,:)= h; % copy to output matrix
  fisave(2,:)= h;
else
  h = 0;
end

if ~isempty(DfDtFu2)
  dfdt2=str2func(DfDtFu2);
  h2 = fp2;
  fi2(1,:) =  h2;
  fi2save(1,:) =  h2;
  df12 = dfdt2(ti(1)*timescale,h2',fi(1,6),BAR,1)';
  f2 = h2 + df12*(dt*timescale);
  df2 = dfdt2(ti(1)*timescale,f2',fi(1,6),BAR,2)';
  g2 = 0.5 * (df2+df12);
  h2 = h2 + g2*dt*timescale;
  fi2(2,:) = h2;
  fi2save(2,:) = h2;
end

ti_s(1) = ti(1);
ti_s(2) = ti(2);

count = 3;
countsave = 3;

for it=3:nt % step by step integration
  if mod(it, floor(nt/20)) == 0
    disp(['Step ', num2str(it) , ' out of ', num2str(nt)]);
  end
  
  if ~isempty(DfDtFu)
    dt=ti(it)-ti(it-1); % time step
    f = h + g*dt;
    df= dfdt(mod(ti(it), CL),f',df2(1:50),BAR,count)';
    g = (g+2*df)/3; % updated derivative
    h = h + g*dt; % value of state variable, 2nd order approx
  end
  
  if ti(it) - lastCallToFunc2 >= skip
    
    if ~isempty(DfDtFu2)
      dt2=ti(it)-lastCallToFunc2; % time step
      f2 = h2 + g2*dt2*timescale;
      df2= dfdt2(ti(it)*timescale,f2',h(6),BAR, count)'; % derivative calculation
      g2 = (g2+2*df2)/3; % updated derivative
      h2 = h2 + g2*dt2*timescale; % value of state variable, 2nd order approx
    end
    if ti(it) >= startsave
        fisave(countsave,:)=h;
        fi2save(countsave,:)=h2;
        ti_s(countsave) = ti(it);
        countsave = countsave + 1;
    end
    count = count + 1;
    lastCallToFunc2 = ti(it);
  end
end
if startsave > 0
    ti_s = ti_s(3:countsave-1);
    fisave = fisave(3:countsave-1,:);
    fi2save = fi2save(3:countsave-1,:);
else
    ti_s = ti_s(1:countsave-1,:);
    fisave = fisave(1:countsave-1,:);
    fi2save = fi2save(1:countsave-1,:);
end
end