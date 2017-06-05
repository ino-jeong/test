% This is Machine Learning Online Class from Coursera, Exercise 7
%
% codes implemented by applicant as assignment of online course are :
%
%     computeCentroids.m
%     findClosestCentroids.m
%     kMeansInitCentroids.m
%
% kMeansInitCentroids() function initializes K centroids that are to be used in K-Means on the dataset X


function centroids = kMeansInitCentroids(X, K)

% ====================== YOUR CODE HERE (Coursera) ======================
% Instructions: You should set centroids to randomly chosen examples from
%               the dataset X

% get random permutation index of x first,
% and pick k data(x) as initial centroids

randidx = randperm(size(X,1));      % random index permutation
centroids = X(randidx(1:K), :);     % return first k data as centroids

% =============================================================

end

