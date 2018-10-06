function classify = adaboost_classify(strong_class, X)
% classify = adaboost_classify(strong_class, X)
%
% Applies the strong classifier to the data X and returns the
% classification labels
%
%   Parameters:
%       strong_class - the structure returned by adaboost()
%
%       X [K x N] - training samples, K is the number of weak classifiers
%            and N the number of data points
%
%   Returns:
%       classify [1 x N] - the labels of the input data X as predicted by
%            the strong classifier strong_class
%

N = size(X,2);
T = length(strong_class.wc);

for iterator = 1:T
 
    parity(iterator) = strong_class.wc(iterator).parity;
    idx(iterator) = strong_class.wc(iterator).idx;
    t_0(iterator) = strong_class.wc(iterator).theta;
    alpha(iterator) = strong_class.alpha(iterator);
end

parity = repmat(parity',1,N);
t_0 = repmat(t_0',1,N);
alpha = repmat(alpha',1,N);
idx = idx';

weak_classifier = alpha.*sign(parity.*(X(idx,:)-t_0));
strong_one = sign(cumsum(weak_classifier));

classify = strong_one(end,:);