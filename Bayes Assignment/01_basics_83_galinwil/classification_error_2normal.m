function error = classification_error_2normal(images, labels, q)
% error = classification_error_2normal(images, labels, q)
%
%   Compute classification error of a strategy q in a test set.
%
%   Parameters:
%       images - test set images, <h x w x n>
%       labels - test set labels <1 x n>
%       q - strategy
%               q.t1 q.t2 - two descision thresholds 
%               q.decision - 3 decisions for intervals (-inf, t1>, (t1, t2>, (t2, inf)
%                            shape <1 x 3>
%
%   Return:
%       error - classification error in range <0, 1>






%my_labels_cont =  
% % Results x intervals 
my_labels = classify_2normal(images,q)
dif_matrix_cont = (labels - my_labels)
elements_dif_than_zero_cont = find(dif_matrix_cont ~= 0)

error = length(elements_dif_than_zero_cont)/length(dif_matrix_cont)
