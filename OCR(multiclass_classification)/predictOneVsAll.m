% This is Machine Learning Online Class from Coursera, Exercise 3
%
% codes implemented by applicant as assignment of online course are :
%
%     lrCostFunction.m (logistic regression cost function)
%     oneVsAll.m
%     predictOneVsAll.m
%
% predictOneVsAll() predicts the label for a trained one-vs-all classifier.


function p = predictOneVsAll(all_theta, X)

% initializing some useful variables
m = size(X, 1);
num_labels = size(all_theta, 1);

X = [ones(m, 1) X];     % Add ones to the X data matrix (bias term)

% ====================== YOUR CODE HERE (Coursera) ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%

total_z = X * all_theta';
total_h = sigmoid(total_z);     % put multiplication into activatioin function (sigmoid)

% check which one has maximum probability.
% and return its label (p_index)
[max_val, p_index] = max(total_h,[],2);
p = p_index;

% =========================================================================
end
