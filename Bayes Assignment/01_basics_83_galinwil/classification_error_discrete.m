function error = classification_error_discrete(data, labels, q)
% error = classification_error_discrete(images, labels, q)
%
%   Compute classification error for a discrete strategy q.
%
%   Parameters:
%       data    <1 x n> vector
%       labels  <1 x n> vector of values 1 or 2
%       q       <1 × m> vector of 1 or 2
%
%   Returns:
%       error - classification error as a fraction of false samples
%               scalar in range <0, 1>


x_feature_per_image = (compute_measurement_lr_discrete(data))

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

my_labels = 2*sum(clasification_c,2)+ sum (clasification_a,2)
labels
dif_matrix = (labels' - my_labels)
elements_dif_than_zero = find(dif_matrix ~= 0)

error = length(elements_dif_than_zero)/length(dif_matrix)


