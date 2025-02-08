clear all;

step_number=30; %step number to be implemented
Total_Subject_Count = 25e3;    % 25k virtual patients

file_name = fullfile(sprintf('Subjects_%d_reprocessed.mat', Total_Subject_Count)); % load the data of virtual patient
load( file_name, 'Subjects', 'Total_Subject_Count', 'y_thresh');
 
y_thresh = 5.0000e-05;  %MEP threshold
opts = optimset('Display', 'off', 'MaxFunEvals', 500000, 'FunValCheck', 'on', 'MaxIter', 10000, 'TolFun', 1e-6, 'TolX', 1e-8); % setting of optimization

note=randi([1, 25000]); %pick a random patient from 1 to 25000

subj_parameters = Subjects(note).subj_parameters; % parameter of the virtual patient
thresh_x = Subjects(note).relative_frequency.p50_lin;  % the threshold of machine stimulation of this virtual subject

lb = 0;        
ub = 1.3;        
start_amplitude = lb + (ub - lb) * rand(1); % starting stimulating amlitude

is_analog=true; % digital or analog

% for parameter 'version':
% 1 represents nonadaptive without prior, 3 represents adaptive without prior, 
% 19 represents nonadaptive with prior, 39 represents adaptive with prior
version=3; 

start_ctrl_seqs=0.15; % this is a_0, step size
SD_coeff=0.15; % this is c, standard deviation of RM distribution


%% below are optimization steps, don't mind

if exist("SD_coeff", "var")
    [start_ctrl_seqs, SD_coeff] = ndgrid(start_ctrl_seqs, SD_coeff);
end


control_sequence = NaN(step_number, 1);
amplitude_list = NaN(step_number+1, 1);
response_bin = false(step_number, 1);
if is_analog
    response_list = NaN(step_number, 1);
end

delta_Y = zeros(step_number, 1);
stochastic_mode = true(step_number, 1);
amplitude_list(1, :) = start_amplitude;


switch version
    case {3, 39}   %%本来是{3,4, 11,12, 13,14, 39}  {1,19,3,4, 11,12, 13,14, 39}
        number_of_sign_changes = zeros(1, 1);
end

% load prior
if (version == 19) || (version == 39)
    file_name = fullfile('Prior_RM.mat');
    load(file_name, 'prior_pdf_vec', 'x_vec'); 
end


for step_cnt = 1 : step_number
    T_start = tic;
    response =  virtstimulate(amplitude_list(step_cnt, :), subj_parameters) ;
    % response > 0; real function is already taken in virtstimulate
    response_bin(step_cnt, :) = logical(response >= y_thresh);
    if is_analog
        response_list(step_cnt, :) = response;
        delta_Y(step_cnt, :) = log10(response) - log10(y_thresh); % in log-space
    else
        delta_Y(step_cnt, :) = 2 * response_bin(step_cnt, :) - 1; % convert from logical [0,1] to [-1,1]
    end

    
    % Calculate control sequence and next amplitude
    switch version
        case {1,19}       % ~1/i, original version IX, X %%本来是{1,2,19}  {2}
            control_sequence(step_cnt, :) = start_ctrl_seqs(:)' / step_cnt;
        case {3,39}      % overall: ~1/i, only in case of sign change, original version I, II %%本来是{3,4,39}  {1,19,3,4,39} 
            if step_cnt == 1
                control_sequence(step_cnt, :) = start_ctrl_seqs(:)';
            else        % step_cnt > 1
                sign_change = xor(response_bin(step_cnt, :), response_bin(step_cnt-1, :));  % logical, 0: no sign change, 1: sign change
                control_sequence(step_cnt, ~sign_change) = control_sequence(step_cnt-1, ~sign_change);
                
                number_of_sign_changes = number_of_sign_changes + sign_change;              % none-zero for those with sign changes
                control_sequence(step_cnt,  sign_change) = control_sequence(step_cnt-1,  sign_change) .* number_of_sign_changes(sign_change)./ (number_of_sign_changes(sign_change) + 1);
            end
    end

    % Finding next amplitude
    if stochastic_mode(step_cnt)
        if (version == 19) || (version == 39) % prior RM
            for ii = 1:1
                prior_mu = amplitude_list(step_cnt, ii) - control_sequence(step_cnt, ii) * delta_Y(step_cnt, ii);
                normal_pdf = normpdf(x_vec, prior_mu , SD_coeff(ii)/step_cnt);
                product = prior_pdf_vec .* normal_pdf;       % Calculate the product
                [~, max_ind] = max(product);            % Find the argmax                
                amplitude_list(step_cnt+1, ii) = x_vec(max_ind);
            end
        else
            % x_(i+1) = x_i - a_i * (y_i - y_th)
            amplitude_list(step_cnt+1, :) = amplitude_list(step_cnt, :) - ...
                                            control_sequence(step_cnt, :) .* delta_Y(step_cnt, :);
        end
        amplitude_list(step_cnt+1, :) = min(max(0, amplitude_list(step_cnt+1, :)), 1.3); % no negative stimuli, maximum clipped to 130% MSO
    else
        amplitude_list(step_cnt+1) = lin_estim(step_cnt);
    end
end

amplitude_list = squeeze(reshape(amplitude_list, [step_number+1, 1, 1]));
abs_err = amplitude_list - thresh_x;               % absolute error (0-1)


%% this is the threshold estiamtion deviation of this virtual subject for each step (in %machine output)
dev=abs_err*100; 
