function R = bayes_risk_discrete(distribution1, distribution2, W, q)
% R = bayes_risk_discrete(distribution1, distribution2, W, q)
%
%   Compute bayesian risk for a discrete strategy q
%
%   Parameters:
%           distribution1.Prob      pXk(x|A) given as a <1 × n> vector
%           distribution1.Prior 	prior probability pK(A)
%           W                       cost function matrix
%                                   dims: <states x decisions>
%                                   (nr. of states and decisions is fixed to 2)
%           q                       strategy - <1 × n> vector, values 1 or 2
%
%   Return:
%           R - bayesian risk, <1 x 1>


% discreteA, discreteC, W, q_discrete
% pXK : X � K ? R be the joint probability that the object is in the state k and the observation x is made.

posterior_a = (distribution1.Prob*distribution1.Prior);
posterior_b = (distribution2.Prob*distribution2.Prior);

post_matrix = [posterior_a ; posterior_b]' * W;

%%%%%

comp_matrix(post_matrix(:,1)<= post_matrix(:,2)) = 1;

comp_matrix_2 = 2*(post_matrix(:,1)> post_matrix(:,2))';

%q = comp_matrix+ comp_matrix_2;

one_col_two_col = post_matrix([q==1; q==2]');

R = sum(one_col_two_col);



