function [strong_class, wc_error, upper_bound] = adaboost(X, y, num_steps)
% [strong_class, wc_error, upper_bound] = adaboost(X, y, num_steps)
%
% Trains an AdaBoost classifier
%
%   Parameters:
%       X [K x N] - training samples, KxN matrix, where K is the number of 
%            weak classifiers and N the number of data points
%
%       y [1 x N] - training samples labels (1xN vector, contains -1 or 1)
%
%       num_steps - number of iterations
%
%   Returns:
%       strong_class - a structure containing the found strong classifier
%           .wc [1 x num_steps] - an array containing the weak classifiers
%                  (their structure is described in findbestweak())
%           .alpha [1 x num_steps] - weak classifier coefficients
%
%       wc_error [1 x num_steps] - error of the best weak classifier in
%             each iteration
%
%       upper_bound [ 1 x num_steps] - upper bound on the training error
%

%% initialisation
%soze of  X must be taken instead of dummy example
[~,N] = size(X);

% prepare empty arrays for results
strong_class.wc = [];
strong_class.alpha = zeros(1, num_steps);

%% AdaBoost

%prepare data to intake in findbeastweak
l_output = y == 1;
l_output_2 = y ~= 1;
result = sum(l_output);
P = N-result;
D = ones(1,N);
%training sample weights 
D(l_output) = D(l_output)/(2*result);
D(l_output_2) = D(l_output_2)/(2*P);
upper_bound(1) = 1;

for t = 1:num_steps
    %find best weak takes Input, labels and D, 
    [wct, wct_error] = findbestweak(X, y, D);
    %better than flip a coin
    if wct_error >= 0.5
        break
    end

    alpha = 0.5*log((1-wct_error)/wct_error);
    
    weak = sign(wct.parity*(X(wct.idx,:)-wct.theta));

    D = D.*exp(-alpha.*y.*weak);
    Z_out = sum(D);
    D = D./Z_out;

    strong_class.wc(t).parity = wct.parity;
    strong_class.wc(t).idx = wct.idx;
    strong_class.wc(t).theta = wct.theta;
    strong_class.alpha(t) = alpha;
    wc_error(t) = wct_error;
    
    upper_bound(t+1) = upper_bound(t)*Z_out;

end
upper_bound = upper_bound(2:end);

end
