function label = classify_2normal(imgs, q)
% label = classify_2normal(imgs, q)
%
%   Classify images using continuous measurement and strategy q.
%
%   Parameters:
%       images - test set images, <h x w x n>
%       q - strategy
%               q.t1 q.t2 - two descision thresholds 
%               q.decision - 3 decisions for intervals (-inf, t1>, (t1, t2>, (t2, inf)
%                            shape <1 x 3>
%
%   Return:
%       label - image labels, <1 x n>


x_feature_per_image_cont = (compute_measurement_lr_cont(imgs))



interval_1 = x_feature_per_image_cont <= q.t1

interval_2 = x_feature_per_image_cont <= q.t2 & x_feature_per_image_cont > q.t1

interval_3 = x_feature_per_image_cont > q.t2

% clasification
if q.decision(1) == 1 
    x_feature_per_image_cont(interval_1) = 1
else 
    x_feature_per_image_cont(interval_1) = 2
end 
   
if q.decision(2) == 1
      x_feature_per_image_cont(interval_2) = 1
else 
    x_feature_per_image_cont(interval_2) = 2
end
    
if q.decision(3) == 1
      x_feature_per_image_cont(interval_3) = 1
else 
      x_feature_per_image_cont(interval_3) = 2
end
   

label = x_feature_per_image_cont
