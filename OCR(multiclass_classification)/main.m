% This is Machine Learning Online Class from Coursera, Exercise 3
%
% codes implemented by applicant as assignment of online course are :
%
%     lrCostFunction.m (logistic regression cost function)
%     oneVsAll.m
%     predictOneVsAll.m


% Initialization
clear ; close all; clc

% Setup the parameters you will use for this part of the exercise
input_layer_size = 400;  % 20x20 Input Images of Digits
num_labels = 10;         % 10 labels (digits), from 1 to 10
                         % (note that we have mapped "0" to label 10)


%  =========== Part 1: Loading and Visualizing Data =============
% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('ex3data1.mat');  % training data stored in arrays X, y
m = size(X, 1);        % m = number of training set

% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);
displayData(sel);

fprintf('Program paused. Press enter to continue.\n');
pause;


%  ============ Part 2: Vectorize Logistic Regression ============
fprintf('\nTraining One-vs-All Logistic Regression...\n')

% 'oneVsAll()' function will execute training with data array X and label y
% and it will return trained weights named 'all_theta'
lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;


%  ================ Part 3: Predict for One-Vs-All ================
%  prediction by trained result
pred = predictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


%  ========== Part 4: Checking Prediction result visually ==========
% randomly permute examples
rp = randperm(m);

% display random 5 examples with prediction result
for i = 1:5
    % Display
    fprintf('\nDisplaying Example Image\n');
    displayData(X(rp(i), :));

    pred = predictOneVsAll(all_theta, X(rp(i),:));
    fprintf('\nMulticlass classification Prediction: %d (digit %d)\n', pred, mod(pred, 10));

    % Pause
    fprintf('Program paused. Press enter to continue.\n');
    pause;
end


