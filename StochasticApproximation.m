%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   StochasticApproximation, Robbins-Monro, fully log, all versions
%   Version 1, 2:       non-adaptive 1/i, 1st & 2nd order, ACS1-H, DCS1-H, ACS2-H
%   Version 3, 4:       adaptive 1/i, 1st & 2nd order, ACS1-HA, DCS1-HA, ACS2-HA
%   Version 5, 6:       adaptive 2^(-i/4), 1st & 2nd order
%   Version 7, 8:       adaptive 2^(-i/2), 1st & 2nd order, ACS1-GA, DCS1-GA, ACS2-GA
%   Version 9, 10:      adaptive 2^(-i/2), 1st & 2nd order, de- and increasing steps
%   Version 11, 12:     adaptive 1/i, 1st & 2nd order, step increases if plateau reached
%   Version 13:         Stochastic Newton, starts as RM (adaptive 1/i, 1st order) until robust estimation with linear regression (not used in preprint, orginal version snewta)
%   Version 5-3:        adaptive 2^(-i/4) switches to 1/i, 1st order i.e., rapid approach switching to a.s. convergence.
%                       Transition: max step stize < +/- 0.015 = 1.5% MSO. ACS1-GHA
%   Version 7-3:        adaptive 2^(-i/2) switches to 1/i, 1st order i.e., rapid approach switching to a.s. convergence.
%                       Transition: max step stize < +/- 0.015 = 1.5% MSO. ACS1-GHA
%   Version 19,39:      Prior RM, ver19 corresponds to ver.1(digital). ver39 corresponds to ver.3 (analog). (LSW, 2024.1.10)

%   StochasticApproximation(params, version, is_analog, num_start_conditions, num_second_order_weights, run_lin, run_MLE, default_start_ctrl_seqs, SDcoeff)
%   Default argin after params: 1, true, 1, 1, false, false, 0.20/(log10(1e-2)-log10(1e-5)), 0.11

%   Estimate derivative with robust linear regression
%   Run MLE in parallel

% The methods we would modify (to include prior) are:
% version = 3 with is_analog = true     ACS-HA
% version = 1 with is_analog = false    DCS-H (non-adaptive)
% analog is cts y，digital is binary y（1 / -1）

function result = StochasticApproximation(...
    params, version, is_analog, ...
    num_start_conditions, num_second_param, ...
    run_lin, run_MLE,...
    default_start_ctrl_seqs, default_SD_coeff)

% I added 2 variables in the function input (default_start_ctrl_seqs and SDcoeff) 
% default_start_ctrl_seqs is a_0 (initial stepsize), SDcoeff is the coefficient of the standard deviation of RM distribution (used in prior RM)

if nargin < 2
    version =  1;
end
if nargin < 3
    is_analog = true;
end
if nargin < 4
    num_start_conditions = 1;
end

%1:d,3:a

% original
% if nargin < 8
%     if version ==1
%         default_start_ctrl_seqs = 0.2;  % 0.18 for 0-130, 0.17 for 20-100
% 
%     elseif version== 3 || version== 39
%         default_start_ctrl_seqs = 0.15;
% 
%     elseif version ==19
%         default_start_ctrl_seqs = 0.17;
% 
%     else 
%         default_start_ctrl_seqs = 0.0667;
%     end
% end
% 
% 
% if nargin < 9
%     if version== 19
%         default_SD_coeff = 0.15;  % 0.18 for 0-130, 0.17 for 20-100
% 
%     elseif version ==39
%         default_SD_coeff = 0.14;
% 
%     else 
%         default_start_ctrl_seqs = 0.0667;
%     end
% end

% prior setting
if nargin < 8
    if version == 1 && is_analog == true %ACS
        default_start_ctrl_seqs = 0.0216 ;  

    elseif version== 19 && is_analog == true %ACSPI
        default_start_ctrl_seqs = 0.0412;

    elseif version ==3 &&  is_analog == true % ACSA
        default_start_ctrl_seqs = 0.0862;

    elseif version ==39 &&  is_analog == true %ACSAPI
        default_start_ctrl_seqs = 0.0676;

    elseif version == 1 && is_analog == false %DCS
        default_start_ctrl_seqs = 0.0158 ;  

    elseif version== 19 && is_analog == false %DCSPI
        default_start_ctrl_seqs = 0.0302;

    elseif version ==3 &&  is_analog == false %DCSA
        default_start_ctrl_seqs = 0.1081;

    elseif version ==39 &&  is_analog == false %DCSAPI
        default_start_ctrl_seqs = 0.0826;

    else 
        default_start_ctrl_seqs = 0.0667;
    end
end


if nargin < 9
    if version == 19 && is_analog == true %ACSPI
        default_SD_coeff = 0.1683; 

    elseif version == 39 && is_analog == true %ACSAPI
        default_SD_coeff = 0.1540;  

    elseif version == 19 && is_analog == false %DCSPI
        default_SD_coeff = 0.1697;  

    elseif version == 39 && is_analog == false %DCSAPI
        default_SD_coeff = 0.1615;  

    else 
        default_start_ctrl_seqs = 0.0667;
    end
end


% if nargin < 9
%     if version == 39
%         default_SD_coeff = 0.14; 
%     elseif version == 19
%         default_SD_coeff = 0.14;
%     else 
%         default_SD_coeff = 0.14;  
%     end
% end

if num_start_conditions == 1
    start_ctrl_seqs = default_start_ctrl_seqs;   
    % starting value is approx. cotangent (ai*dy = dx => ai = dx/dy)
    % Overestimate slightly??
else
    num_start_conditions = fix(num_start_conditions/2)*2 + 1; % Make odd 
    start_ctrl_seqs = default_start_ctrl_seqs * logspace(-0.75, 0.75, num_start_conditions);        % 31 starting positions: step 0.05
end
if (nargin < 5) || (mod(version, 2) == 1 && ~(version == 19 || version == 39))
    num_second_param = 1;
end
if num_second_param == 1
    if mod(version, 2) == 0 
        second_order_weights = -0.10;
    elseif (version == 19) || (version == 39)
        SD_coeff = default_SD_coeff;
    end
else
    if mod(version, 2) == 0
        num_second_param = fix(num_second_param/2)*2 + 1; % Make odd 
        second_order_weights = linspace(-1, 1, num_second_param);
    elseif (version == 19) || (version == 39)
        SD_coeff = default_SD_coeff + linspace(-0.04, 0.15, num_second_param);   % 20 SD coefficients: 0.01:0.01:0.20 
       
    end
end
if exist("second_order_weights", "var")
    [start_ctrl_seqs, second_order_weights] = ndgrid(start_ctrl_seqs, second_order_weights);
elseif exist("SD_coeff", "var")
    [start_ctrl_seqs, SD_coeff] = ndgrid(start_ctrl_seqs, SD_coeff);
end
num_conditions = num_start_conditions * num_second_param;

if nargin < 6 || (num_conditions > 1)
    run_lin = false;
end
if nargin < 7 || (num_conditions > 1)
    run_MLE = false;
end


y_thresh = params.y_thresh;
step_number = params.step_number;


control_sequence = NaN(step_number, num_conditions);
amplitude_list = NaN(step_number+1, num_conditions);
response_bin = false(step_number, num_conditions);
if is_analog
    response_list = NaN(step_number, num_conditions);
end
run_time = NaN(step_number, 1);

% delta_ctrl_seq = zeros(1, num_conditions);
delta_Y = zeros(step_number, num_conditions);

stochastic_mode = true(step_number, 1);

amplitude_list(1, :) = params.start_amplitude;

% (compensating that saturation reduces the error term and thus the drive to go back to the threshold)
cutoff_low = 10e-6;            % 10 µV. below that, the step size is increased again 
cutoff_high = 10e-3;           % 10 mV. above that, the step size is increased again

if run_lin
    lin_estim = NaN(1, step_number);
    min_lin = 5;
end
if run_MLE
    MLE_t = NaN(1, step_number);
    MLE_s = NaN(1, step_number);
    MLE_opts = params.opts;
end

switch version
    case {3,4, 11,12, 13,14, 39}   %%本来是{3,4, 11,12, 13,14, 39}  {1,19,3,4, 11,12, 13,14, 39}
        number_of_sign_changes = zeros(1, num_conditions);
    case {5,6}
        b_i = (2)^(1/3);
    case {7,8, 9,10}
        b_i = sqrt(2);
    case {53,54}
        b_i = (2)^(1/3);
        as_convergence = false(1, num_conditions);   % false: geometric; true: harmonic
        number_of_sign_changes = zeros(1, num_conditions);
    case {73,84}
        b_i = sqrt(2);
        as_convergence = false(1, num_conditions);   % false: geometric; true: harmonic
        number_of_sign_changes = zeros(1, num_conditions);
end

% load prior
if (version == 19) || (version == 39)

    file_name = fullfile('Prior_RM.mat');
    load(file_name, 'prior_pdf_vec', 'x_vec'); 
end



for step_cnt = 1 : step_number
    T_start = tic;
    response =  virtstimulate(amplitude_list(step_cnt, :), params.subj_parameters) ;
    % response > 0; real function is already taken in virtstimulate
    response_bin(step_cnt, :) = logical(response >= y_thresh);
    if is_analog
        response_list(step_cnt, :) = response;
        delta_Y(step_cnt, :) = log10(response) - log10(y_thresh); % in log-space
    else
        delta_Y(step_cnt, :) = 2 * response_bin(step_cnt, :) - 1; % convert from logical [0,1] to [-1,1]
    end
    if run_lin
        %%%%%%%%%%%%%%%%%%%%%%
        % Estimate Derivative:
        if (step_cnt > min_lin)
            [pcoeff, ~] = robustfit(amplitude_list(1:step_cnt), log10(response_list(1:step_cnt)), 'bisquare');
            % Estimate threshold from this fit:
            lin_estim(step_cnt) = (log10(y_thresh) - pcoeff(1)) / pcoeff(2);
            % a + b*x = y -> x = (y-a)/b
        end
    end
    if run_MLE
        %%%%%%%%%%%%%%%%%%%%%%
        % MLEstimator with data
        stim_vec = cat(2, zeros(1, num_conditions)', amplitude_list(1:step_cnt, :)'); % added 0-stimulation (with 0 response)
        response_vec = cat(2, false(1, num_conditions)', response_bin(1:step_cnt, :)');

        fun_MLE_min = @(theta) -1 * loglikelyhood(stim_vec, response_vec, theta(1), theta(2));
        MLE_init = [0.5, 0.2];
        [MLE, ~, MLE_exitflag] = fminsearch(fun_MLE_min, MLE_init, MLE_opts);	% find max via min of negative
        if ~MLE_exitflag            % Optimization did not converge, do it again:
            MLE_init = [rand *1, rand *0.15];
            [MLE, ~, MLE_exitflag] = fminsearch(fun_MLE_min, MLE_init, MLE_opts);
            while ~MLE_exitflag        % do it again:
                MLE_init = [rand *1, rand *0.1];
                [MLE, ~, MLE_exitflag] = fminsearch(fun_MLE_min, MLE_init, MLE_opts);
            end
        end
        if MLE_exitflag
            MLE_t(step_cnt) = MLE(1);
            MLE_s(step_cnt) = MLE(2);
        end
    end
    
    % Calculate control sequence and next amplitude
    switch version
        case {1,2,19}       % ~1/i, original version IX, X %%本来是{1,2,19}  {2}
            control_sequence(step_cnt, :) = start_ctrl_seqs(:)' / step_cnt;
        case {3,4,39}      % overall: ~1/i, only in case of sign change, original version I, II %%本来是{3,4,39}  {1,19,3,4,39} 
            if step_cnt == 1
                control_sequence(step_cnt, :) = start_ctrl_seqs(:)';
            else        % step_cnt > 1
                sign_change = xor(response_bin(step_cnt, :), response_bin(step_cnt-1, :));  % logical, 0: no sign change, 1: sign change
                control_sequence(step_cnt, ~sign_change) = control_sequence(step_cnt-1, ~sign_change);
                
                number_of_sign_changes = number_of_sign_changes + sign_change;              % none-zero for those with sign changes
                control_sequence(step_cnt,  sign_change) = control_sequence(step_cnt-1,  sign_change) .* number_of_sign_changes(sign_change)./ (number_of_sign_changes(sign_change) + 1);
            end
        case {5,6,7,8}      % overall: ~2^(-i/3) or ~2^(-i/2), only in case of sign change, original version III, IV
            if step_cnt == 1
                control_sequence(step_cnt, :) = start_ctrl_seqs(:)';
            else % step_cnt > 1
                sign_change = xor(response_bin(step_cnt, :), response_bin(step_cnt-1, :));      % logical, 0: no sign change, 1: sign change
                control_sequence(step_cnt, :) = control_sequence(step_cnt-1, :) ./ (b_i.^sign_change);  %  b_i^-1 for sign change
            end
        case {9,10}     % overall: ~2^(-i/2), only in case of sign change, original version VII, VIII
%             delta_ctrl_seq = (b_i    - 1) .*  xor(response_bin(step_cnt, :), response_bin(step_cnt-1, :)) + ...
%                              (b_i^-1 - 1) .* ~xor(response_bin(step_cnt, :), response_bin(step_cnt-1, :)) ;
            if step_cnt == 1
                control_sequence(step_cnt, :) = start_ctrl_seqs(:)';
            else % step_cnt > 1
                sign_change = xor(response_bin(step_cnt, :), response_bin(step_cnt-1, :)) * 2 -1;   % -1: no sign change, 1: sign change
                control_sequence(step_cnt, :) = control_sequence(step_cnt-1, :) ./ (b_i.^sign_change);
            end
        case {11,12}    % overall: ~1/i, only in case of sign change & not in plateau regions, original version XXI
            if step_cnt == 1
                control_sequence(step_cnt, :) = start_ctrl_seqs(:)';
            else        % step_cnt > 1
                sign_change = xor(response_bin(step_cnt, :), response_bin(step_cnt-1, :)) & ...
                                ( response_list(step_cnt, :) > cutoff_low  )  & ...
                                ( response_list(step_cnt, :) < cutoff_high );       % logical, 1: sign change, counted only outside plateau, 0: no sign change or sign change but in plateau
                control_sequence(step_cnt, ~sign_change) = control_sequence(step_cnt-1, ~sign_change);
                
                number_of_sign_changes = number_of_sign_changes + sign_change;                     
                control_sequence(step_cnt,  sign_change) = control_sequence(step_cnt-1,  sign_change) .* number_of_sign_changes(sign_change)./ (number_of_sign_changes(sign_change) + 1);
            end
        case {13,14} % Stochastic Newton, starts as RM (adaptive 1/n, 1st order) until robust estimation with linear regression, original version swneta
            % Linear approximation considered stable as soon as
            % derivative positive and steeper than initial derivative
            % as well as estimated threshold between 10% and 100%
            % amplitude => Newton-Raphson-like descent using the
            % linearized approximation
            if (step_cnt > min_lin) && (pcoeff(2) > 1/start_ctrl_seqs) && ...
               (lin_estim(step_cnt)>0.1) && (lin_estim(step_cnt)<1.0)
                stochastic_mode(step_cnt) = false;
            else    % Normal adaptive RM until robust estimate
                if step_cnt == 1
                    control_sequence(step_cnt, :) = start_ctrl_seqs(:)';
                else        % step_cnt > 1
                    sign_change = xor(response_bin(step_cnt, :), response_bin(step_cnt-1, :));  % logical, 0: no sign change, 1: sign change
                    control_sequence(step_cnt, ~sign_change) = control_sequence(step_cnt-1, ~sign_change);
                    
                    number_of_sign_changes = number_of_sign_changes + sign_change;              % none-zero for those with sign changes
                    control_sequence(step_cnt,  sign_change) = control_sequence(step_cnt-1,  sign_change) .* number_of_sign_changes(sign_change)./ (number_of_sign_changes(sign_change) + 1);
                end
            end
            
        case {53,54, 73,84}	% Controlling sequence in the beginning: ~2^(-i/3) or ~2^(-i/2); only in case of sign change)
            % switches to 1/n as soon as max expected step size < +/- 1.5% MSO
            if step_cnt == 1
                control_sequence(step_cnt, :) = start_ctrl_seqs(:)';
            else        % step_cnt > 1
                sign_change = xor(response_bin(step_cnt, :), response_bin(step_cnt-1, :));  % logical, 0: no sign change, 1: sign change
                control_sequence(step_cnt, ~sign_change) = control_sequence(step_cnt-1, ~sign_change);
                
                max_step_size = control_sequence(step_cnt-1, :) .* ...
								(    log10(max(response_list, [], 1, 'omitnan') ) - ...
								 min(log10(min(response_list, [], 1, 'omitnan') ), log10(y_thresh)) );
                as_convergence = as_convergence | (max_step_size <  1.5e-2);            % switch mode and stick to a.s.; false: geometric; true: harmonic
                
                ind_sign_change_NOT_as = sign_change & ~as_convergence;
                control_sequence(step_cnt,  ind_sign_change_NOT_as) = control_sequence(step_cnt-1,  ind_sign_change_NOT_as) / (b_i);
                
                ind_sign_change_AND_as = sign_change &  as_convergence;
                number_of_sign_changes = number_of_sign_changes + sign_change;              % none-zero for those with sign changes
                control_sequence(step_cnt,  ind_sign_change_AND_as) = control_sequence(step_cnt-1,  ind_sign_change_AND_as) .* number_of_sign_changes(ind_sign_change_AND_as)./ (number_of_sign_changes(ind_sign_change_AND_as) + 1);
            end
    end

    % Finding next amplitude
    if stochastic_mode(step_cnt)
        if (version == 19) || (version == 39) % prior RM
            for ii = 1:num_conditions
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
            if ~mod(version, 2) && (step_cnt >= 2)   % Even versions, add second order term
                % x_(i+1) = x_i - a_i * ( (y_i - y_th) + w * (y_(i-1) - y_th) )
                amplitude_list(step_cnt+1, :) = amplitude_list(step_cnt+1, :) - ...
                                                control_sequence(step_cnt, :) .* (second_order_weights(:)' .* delta_Y(step_cnt-1, :) ) ; 
            end
        end
        amplitude_list(step_cnt+1, :) = min(max(0, amplitude_list(step_cnt+1, :)), 1.3); % no negative stimuli, maximum clipped to 130% MSO
    else
        amplitude_list(step_cnt+1) = lin_estim(step_cnt);
    end
    run_time(step_cnt) = toc(T_start)/num_conditions;
end

result.run_time = run_time;
result.control_sequence = squeeze(reshape(control_sequence, [step_number, num_start_conditions, num_second_param]));
result.amplitude_list = squeeze(reshape(amplitude_list, [step_number+1, num_start_conditions, num_second_param]));
result.step_size = squeeze(reshape(cat(1, zeros(1, num_conditions), diff(amplitude_list)), [step_number+1, num_start_conditions, num_second_param]));

result.abs_err = result.amplitude_list - params.thresh_x;               % absolute error (0-1)
result.rel_err = result.abs_err ./ params.thresh_x *100;                % relative error (0%-100%)

if is_analog
    result.response_list = squeeze(reshape(response_list, [step_number, num_start_conditions, num_second_param]));
else
    result.response_bin = squeeze(reshape(response_bin, [step_number, num_start_conditions, num_second_param]));
end
if run_lin
    result.lin_estim = squeeze(reshape(lin_estim, [step_number, num_start_conditions, num_second_param]));
    result.lin_abs_err = result.lin_estim - params.thresh_x;            % absolute error (0-1)
    result.lin_rel_err = result.lin_abs_err ./ params.thresh_x *100;    % relative error (0%-100%)
end
if run_MLE
    result.MLE_t = squeeze(reshape(MLE_t, [step_number, num_start_conditions, num_second_param]));
    result.MLE_s = squeeze(reshape(MLE_s, [step_number, num_start_conditions, num_second_param]));
    result.MLE_abs_err = result.MLE_t - params.thresh_x;                % absolute error (0-1)
    result.MLE_rel_err = result.MLE_abs_err ./ params.thresh_x *100;    % relative error (0%-100%)
end
if version == 13
    result.stochastic_mode = squeeze(reshape(stochastic_mode, [step_number, num_start_conditions, num_second_param]));
end

% amplitude_deviation_list_excluding_last = abs(amplitude_list(1:end-1, :)- params.thresh_x * ones(step_number, num_conditions)); %0.6569 is thresh_x
% result.all_devi = amplitude_deviation_list_excluding_last;
% total_devi = sum(amplitude_deviation_list_excluding_last, 2);
% result.last_step_devi = total_devi(end);

end