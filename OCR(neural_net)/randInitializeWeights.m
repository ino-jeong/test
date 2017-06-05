% This is Machine Learning Online Class from Coursera, Exercise 4
%
% codes implemented by applicant as assignment of online course are :
%
%     sigmoidGradient.m
%     nnCostFunction.m
%
% randInitializeWeights() function randomly initialize the weights of a layer with L_in / L_out size
% note that W should be set to a matrix of size(L_out, 1 + L_in) as the column row of W handles the "bias" terms


function W = randInitializeWeights(L_in, L_out)

% You need to return the following variables correctly 
W = zeros(L_out, 1 + L_in);

% ====================== YOUR CODE HERE ======================
% Instructions: Initialize W randomly so that we break the symmetry while
%               training the neural network.
%
% Note: The first row of W corresponds to the parameters for the bias units
%

epsilon_init = 0.12;
W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;

% =========================================================================

end
