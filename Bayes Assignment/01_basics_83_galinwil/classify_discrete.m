function labels = classify_discrete(img, q)
% label = classify_discrete(imgs, q)
%
%   Classify images using discrete measurement and strategy q.
%
%   Parameters:
%       images - test set images, <h x w x n>
%       q - strategy <1 × 21> vector of 1 or 2
%
%   Return:
%       label - image labels, <1 x n>


x_feature_per_image = (compute_measurement_lr_discrete(img))

x_normalized = -10:1:10
q
twos_in_q = q ==2;
ones_in_q = q ==1;

twos_in_q;
ones_in_q;


c_finder = (x_normalized(twos_in_q))

a_finder = (x_normalized(ones_in_q))

clasification_c= x_feature_per_image == c_finder
clasification_a = x_feature_per_image == a_finder

labels = 2*sum(clasification_c,2)+ sum (clasification_a,2)



%labels = 
