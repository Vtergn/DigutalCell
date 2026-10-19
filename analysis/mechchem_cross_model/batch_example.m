%% Example batch to run the electromechanical model

clear all

% SIMULATION SETTINGS
nb_beat = 1; % Nb of beats to simulate
Sarcomere_length = 2100;
F_afterload = 70; 
CL = 1000; % Cycle length (ms)
BAR = 0; % Beta adrenergic stimulation, 0 = control case, 1 = presence of beta adrenergic stimulation

flux = 0; % Boolean to compute the current variables
computeS = 1; % Boolean to compute tension S
saved_variables = {'T','S','Ca','Y1','Y','params'}; % List of variables that are saved after the simultion 
prepaced = 1; % 1 = loading of an already prepaced state (prepace_state), 0 = start from scratch 
prepace_state = 'prepaced_SL2100_BAR0_CL1000_500beats.mat';

EMmodel_runner; 

%% Plots 

load('EMsimulation.mat','T','Ca','S')
subplot(2,2,1), plot(T,Ca,'k','LineWidth',2), title('Intracellular calcium'), ylim([0 0.6])
subplot(2,2,2), plot(T,S,'k','LineWidth',2), title('Tension')