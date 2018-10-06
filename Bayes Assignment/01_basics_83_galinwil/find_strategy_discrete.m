function q = find_strategy_discrete(distribution1, distribution2, W)
% q = find_strategy_discrete(distribution1, distribution2, W)
%
%   Find bayesian strategy for 2 discrete distributions.
%   
%   Parameters:
%       distribution1.Prob      pXk(x|A) given as a <1 × n> vector
%       distribution1.Prior 	prior probability pK(A)
%       distribution2.Prob      ...
%       distribution2.Prior 	...
%       W - cost function matrix, <states x decisions>
%                                (nr. of states is fixed to 2)
%
%   Return: 
%       q - optimal strategy <1 x n>

posterior_a = (distribution1.Prob*distribution1.Prior);
posterior_b = (distribution2.Prob*distribution2.Prior);

%this is the risk
post_matrix = [posterior_a ; posterior_b]' * W;

comp_matrix(post_matrix(:,1)<= post_matrix(:,2)) = 1;
%comp_matrix_2 = zeros(1, 21)
comp_matrix_2 = 2*(post_matrix(:,1)> post_matrix(:,2))';

%with the risk i can find the optimal q for which risk is lowest
q = comp_matrix+ comp_matrix_2;



