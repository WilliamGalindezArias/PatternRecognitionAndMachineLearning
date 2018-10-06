function errors = compute_error(strong_class, X, y)
% errors = compute_error(strong_class, X, y)
%
% Computes the error on data X for *all lengths* of the given strong
% classifier
%
%   Parameters:
%       strong_class - the structure returned by adaboost()
%
%       X [K x N] - samples, K is the number of weak classifiers and N the
%            number of data points
%
%       y [1 x N] - sample labels (-1 or 1)
%
%   Returns:
%       errors [1 x T] - error of the strong classifier for all lenghts 1:T
%            of the strong classifier
%
T = length(strong_class.wc);
N = size(X,2);

for iterator = 1:T
    temp_parity = strong_class.wc(iterator).parity;
    temp_idx = strong_class.wc(iterator).idx;
    temp_theta = strong_class.wc(iterator).theta;
    temp_alpha = strong_class.alpha(iterator);
    parity(iterator) = temp_parity;
    idx(iterator) = temp_idx;
    theta(iterator) = temp_theta;
    alpha(iterator) = temp_alpha;
end

weak_classif = repmat(alpha',1,N).*sign(repmat(parity',1,N).*(X(idx,:)-repmat(theta',1,N)));
strong_one = sign(cumsum(weak_classif));

%errors [1 x T] - error of the strong classifier for all lenghts 1:T
%            of the strong classifier
errors = sum((strong_one ~= repmat(y,T,1)),2)./N;
errors = errors';
