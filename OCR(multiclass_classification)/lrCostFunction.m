% This is Machine Learning Online Class from Coursera, Exercise 3
%
% codes implemented by applicant as assignment of online course are :
%
%     lrCostFunction.m (logistic regression cost function)
%     oneVsAll.m
%     predictOneVsAll.m
%
% lrCostFunction() Compute cost and gradient for logistic regression with regularization

function [J, grad] = lrCostFunction(theta, X, y, lambda)

% Initialize some values
m = length(y);              % number of training examples
grad = zeros(size(theta));
theta1 = theta([2:end],:);

% ====================== YOUR CODE HERE (Coursera) ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%


% compute z first
z = X * theta;


% get cost 'j'
J = -1/m *  ( y .* log(sigmoid(z)) + (1-y) .* log(1-sigmoid(z)) );
J = sum(J);
J = J + lambda / (2*m)*(theta1'*theta1);    % add regularization term

                        
% get gradient
grad = (1/m) * (X' * ( sigmoid(z) - y ) );
grad(2:end) = grad(2:end) + lambda/m .* theta1;     % add regularization term except bias term


% =============================================================
end
