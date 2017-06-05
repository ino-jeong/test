% This is Machine Learning Online Class from Coursera, Exercise 4
%
% codes implemented by applicant as assignment of online course are :
%
%     sigmoidGradient.m
%     nnCostFunction.m
%
% nnCostFunction implements the neural network cost function for a two layer
% neural network which performs classification


function [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
delta1 = zeros(size(Theta1));
delta2 = zeros(size(Theta2));

X_ = [ones(size(X,1),1) X];     % Add bias term in X


% ====================== YOUR CODE HERE (Coursera) ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% for each i_th training set, calculate cost and adding it to itself.
for t = 1:m

    % perform forward propagation first to get errors
    z2 = Theta1 * X_(t,:)';
    a2 = sigmoid(z2);
    a2 = [ones(1,size(a2,2)) ; a2];     % add bias term in a2

    z3 = Theta2 * a2;
    h = sigmoid(z3);  % h == a3

    % get true label y of this training set
    yt = zeros(size(Theta2,1),1);
    yt(y(t,1),1) = 1;

    % compute cost
    J = J + sum( -yt.*log(h) - (1-yt).*log(1-h) );

    % calculate errors at each layer (back propagation)
    er3 = h - yt;
    er2 = Theta2(:,2:end)' * er3 .* sigmoidGradient(z2);    % note that bias term is removed from Theta2

    delta2 = delta2 + er3 * a2';
    delta1 = delta1 + er2 * X_(t,:);
end


% add regularization term
reg = sum( sum(Theta1(:,2:end) .* Theta1(:,2:end)) ) + sum( sum(Theta2(:,2:end) .* Theta2(:,2:end)) );
J = J/m + lambda/(2*m) * reg;

Theta2_grad = 1/m * delta2;     % Theta2_grad = D2 = 1/m * delta2, without regularization
Theta1_grad = 1/m * delta1;     % Theta1_grad = D1 = 1/m * delta1, without regularization

Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m * Theta2(:,2:end);   % (add regularization term)
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m * Theta1(:,2:end);   % (add regularization term)

% =========================================================================
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
