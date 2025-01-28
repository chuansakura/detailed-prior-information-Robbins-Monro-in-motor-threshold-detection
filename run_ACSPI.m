clear all;

step=30; %step number to be implemented

if ~exist('Total_Subject_Count', 'var')
    Total_Subject_Count = 25e3;    % 25k
end

file_name = fullfile(sprintf('Subjects_%d_reprocessed.mat', Total_Subject_Count));
load( file_name, 'Subjects', 'Total_Subject_Count', 'y_thresh');

params.step_number = step;   
params.y_thresh = y_thresh;
params.opts = optimset('Display', 'off', 'MaxFunEvals', 500000, 'FunValCheck', 'on', 'MaxIter', 10000, 'TolFun', 1e-6, 'TolX', 1e-8);

note=randi([1, 25000]); %pick a random patient from 1 to 25000

params.subj_parameters = Subjects(note).subj_parameters;
params.thresh_x = Subjects(note).relative_frequency.p50_lin; 

lb = 0;        
ub = 1.3;        
params.start_amplitude = lb + (ub - lb) * rand(1);


rr = StochasticApproximation(params, 19, true, 1, 1, false, false,0.17,0.15);
% penultimate parameter is stepsize a0, last parameter is c

dev_ACSPI = rr.abs_err*100;  %threshold estimation deviation record for 30 steps (in %MSO)

