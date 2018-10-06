function R = bayes_risk_2normal(distribution1, distribution2, q)
% R = bayes_risk_2normal(distribution1, distribution2, q)
%
%   Compute bayesian risk of a strategy q for 2 normal distributions and zero-one loss function.
%
%   Parameters:
%       distribution1 - parameters of the normal dist. distribution1.Mean, distribution1.Sigma
%       distribution2 - the same as distribution1
%       q - strategy
%               q.t1 q.t2 - two descision thresholds 
%               q.decision - 3 decisions for intervals (-inf, t1>, (t1, t2>, (t2, inf)
%                            shape <1 x 3>
%
%   Return:
%       R - bayesian risk, scalar

priorA = distribution1.Prior
priorC = distribution2.Prior
% 
% % Results x intervals 
%   first interval -inf to t1
%   first_interval = integral_1
% second interval t1 to t2
% areas of the graph
% area calculations
% area_1 = normcdf(q.t, distribution1.Mean, distribution.Sigma)
% area_2 = (normcdf(q.t,distribution.Mean, distribution.Sigma) - normcdf(q.t,distribution.Mean, distribution.Sigma))
% area_3 = (1 - normcdf(q.t, distribution.Mean, distribution.Sigma))

% integral_1 = normcdf(q.t1, distribution1.Mean, distribution1.Sigma)* priorA
% integral_2 = normcdf(q.t2, distribution1.Mean, distribution1.Sigma) * priorC
% integral_3 = normcdf(q.t2, distribution2.Mean, distribution2.Sigma)* priorC

%second_interval = (normcdf(q.t2,distribution2.Mean, distribution2.Sigma) - normcdf(q.t1,distribution2.Mean, distribution2.Sigma))*priorC
%case  [1,2,1]

% first_interval = normcdf(q.t1, distribution1.Mean, distribution1.Sigma)* priorA
% second_interval = (normcdf(q.t2,distribution2.Mean, distribution2.Sigma) - normcdf(q.t1,distribution2.Mean, distribution2.Sigma))*priorC
% third_interval = (1 - normcdf(q.t2, distribution1.Mean, distribution1.Sigma))*priorA
% sum_of_integrals = first_interval + second_interval + third_interval

%case  [1,1,2]


%another interval

% selection

%Case 111

if q.decision(1)== 1
    prob_1 = (normcdf(q.t1, distribution1.Mean, distribution1.Sigma))*priorA
else
   prob_1 =  (normcdf(q.t1, distribution2.Mean, distribution2.Sigma))*priorC
end
   
 % Second interval   
if q.decision(2) == 1
    prob_2 = (normcdf(q.t2,distribution1.Mean, distribution1.Sigma) - normcdf(q.t1,distribution1.Mean, distribution1.Sigma))*priorA
    
else
    prob_2 = (normcdf(q.t2,distribution2.Mean, distribution2.Sigma) - normcdf(q.t1,distribution2.Mean, distribution2.Sigma))*priorC
end

% Interval
if q.decision(3) == 1
    prob_3 = (1 - normcdf(q.t2, distribution1.Mean, distribution1.Sigma))*priorA
    
else 
    prob_3 = (1 - normcdf(q.t2, distribution2.Mean, distribution2.Sigma))*priorC
end
    
   
   
% case 

R = 1- sum([prob_1, prob_2, prob_3])
    
